import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected, conv2d, conv2d_transpose
from time import time
from utils import *
from tensorflow.python.ops.parallel_for.gradients import jacobian

def adverserial_loss(yhat, y):

    from discriminator import discriminator

    def gradient_penalty(d, yhat, y):
        bs = tf.shape(yhat)[0]
        alpha = tf.random_uniform(shape=[bs, 1, 1, 1], minval=0., maxval=1.)
        differences = yhat - y
        interpolates = y + (alpha * differences)
        gradients = tf.gradients(d(interpolates, reuse=True), [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2, 3]))
        gradient_penalty = tf.reduce_mean((slopes-1.)**2)

        return gradient_penalty

    d_hat = discriminator(yhat)
    d_real = discriminator(y, reuse=True)
    # gp = gradient_penalty(discriminator, yhat, y)

    D_loss = tf.reduce_mean(d_real) - tf.reduce_mean(d_hat) #+ 10*gp
    G_loss = -tf.reduce_mean(d_hat)# + 10*gp

    disc_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
    
    clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in disc_vars]

    return D_loss, G_loss, clip_D, disc_vars

def prep_test_data(inp_path, out_path, BS=8):
    test_inp_data = np.load(inp_path).astype(np.float32)[:BS]
    test_out_data = np.load(out_path).astype(np.float32)[:BS]

    # lose channel dimension as mp_decompose does not like it!
    test_inp_data = test_inp_data[:, :, :, 0]
    test_out_data = test_out_data[:, :, :, 0]

    # I use a dummy generator as I only need access to a pre-built CL
    # class without specifying class properties.
    dummygenerator = dataGenerator(inp_path, BS)
    test_inp_data = dummygenerator.cl.mp_decompose(
        test_inp_data)  # .sum(axis=-1, keepdims=True)
    test_out_data = dummygenerator.cl.mp_decompose(
        test_out_data)

    return test_inp_data, test_out_data

class SingleBigDiffeo():
    """ This network takes in x,y and box center to spit out
    the coordinates of the deformed network, and also which box of the input
    should the diffeo be calculated from
    """

    def __init__(self, size, box_center_npy, nhidden=128, nlayers=2):
        self.nh = nhidden
        self.nl = nlayers
        self.size = size
        self.bc = tf.convert_to_tensor(box_center_npy, dtype=tf.float32)
        print('Got here')

        self.grid = self.create_cartesian_grid(self.size)
        self.train_flag = tf.placeholder(tf.bool, shape=[])
        print('Made grid')
        print('setup flows')

    def estimate_wavespeed(self, x):

        with tf.variable_scope('wavespeed', reuse=tf.AUTO_REUSE):

            # z is 8x8x1
            h = fully_connected(x, 32, activation_fn=tf.nn.relu)
            h = fully_connected(h, 64, activation_fn=tf.nn.relu)
            h = fully_connected(h, 128, activation_fn=tf.nn.relu)
            h = fully_connected(h, 256, activation_fn=tf.nn.relu)
            h = fully_connected(h, 128, activation_fn=tf.nn.relu)
            h = fully_connected(h, 64, activation_fn=tf.nn.relu)
            h = fully_connected(h, 32, activation_fn=tf.nn.relu)
            h = fully_connected(h, 1, activation_fn=tf.nn.relu)

        return h + 1 # increase min of relu to 1


    def create_cartesian_grid(self, size):
        """Creates a Cartesian grid of size x size points"""
        sx, sy = size
        sx = sx+2
        sy = sy+2
        ax = tf.linspace(0.0, 1.0-1.0/sx, sx)
        ay = tf.linspace(0.0, 1.0-1.0/sy, sy)
        x, y = tf.meshgrid(ax, ay)
        x = tf.reshape(x, [-1])
        y = tf.reshape(y, [-1])
        orig = tf.stack((x, y), axis=1)
        return orig


    def net(self, boxid, scope='d1'):
        sx, sy = self.size
        sx += 2
        sy += 2
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            b = tf.tile(tf.reshape(self.bc[boxid], (1, 2)),
                        [sx*sy, 1])
            x = tf.concat((self.grid, b), axis=-1)
            h = x
            for d in range(self.nl):
                h = fully_connected(h, 2**d*self.nh, activation_fn=tf.nn.relu)
            y = fully_connected(h, 2, activation_fn=tf.identity)
            y += x[:,:2]
            ch = fully_connected(h, 2, activation_fn=tf.identity)
            # ch = tf.stack((ch[:, 0], tf.abs(ch[:, 1])), axis=-1)
            # ch = ch/(tf.norm(ch, axis=-1, keepdims=True) + 1e-12)
            y = tf.reshape(y, [1, sx, sy, 2])
            x = y[:, :, :, 0]
            y = y[:, :, :, 1]

            ch = tf.reshape(ch, [1, sx, sy, 2])

        return x, y, ch

    @staticmethod
    def calc_singvals_of_jacobian(inputs, outputs):
        """Inputs are Nx4, outputs are Nx4"""
        N = tf.shape(inputs)[0]
        idx = tf.range(N)
        sample_num = tf.cast(tf.cast(N, tf.float32)*0.01, tf.int32)
        ridxs = tf.random_shuffle(idx)[:sample_num]
        outputs = tf.gather(outputs, ridxs)

        grads = tf.gradients(outputs[:,0], inputs, stop_gradients=inputs)
        a, b = grads[0][:,0], grads[0][:,1]

        grads = tf.gradients(outputs[:,1], inputs, stop_gradients=inputs)
        c, d = grads[0][:,0], grads[0][:,1]

        S1 = a**2 + b**2 + c**2 + d**2
        S2 = tf.sqrt((a**2 + b**2 - (c**2 + d**2))**2 + 4*(a*c+b*d)**2)

        sig1 = tf.sqrt((S1+S2)/2.0)
        sig2 = tf.sqrt((S1-S2)/2.0)
        sig = tf.stack([sig1, sig2], axis=-1)
        sig = tf.gather(sig, ridxs, axis=0)

        return sig

    @staticmethod
    def calc_loss(sing_vals):
        return tf.reduce_sum(tf.square(sing_vals-1))

    def get_singvals_loss(self, inputs, outputs):
        return self.calc_loss(self.calc_singvals_of_jacobian(inputs, outputs))

    def load(self, sess, ckpt, saver):
        saver.restore(sess, ckpt)
        return


    def net_w_inp(self, x, sx=None, sy=None, scope='d1', return_sing_loss=True):
        if sx is None or sy is None:
            sx, sy = self.size

        sx += 2
        sy += 2
        loss = 0
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            h = x
            for d in range(self.nl):
                h = fully_connected(h, 2**d*self.nh, activation_fn=tf.nn.relu)
            y = fully_connected(h, 2, activation_fn=tf.identity)
            y += x[:, :2]
            ch = fully_connected(h, 2, activation_fn=tf.identity)
            # ch = tf.stack((ch[:, 0], tf.abs(ch[:, 1])), axis=-1)
            # ch = ch/(tf.norm(ch, axis=-1, keepdims=True) + 1e-12)

            if return_sing_loss:
                out = tf.concat([y, ch], axis=-1)
                d = y-x[:,:2]
                d_s = tf.split(d, 48, axis=0, name='split')
                d_s = tf.stack(d_s, axis=-1)

                loss += tf.reduce_mean(tf.linalg.norm(tf.reduce_mean(d_s, axis=0), axis=0)**2)

            y = tf.reshape(y, [-1, sx, sy, 2])
            x = y[:, :, :, 0]
            y = y[:, :, :, 1]

            ch = tf.reshape(ch, [-1, sx, sy, 2])

        return x, y, ch, loss


    def train_net(self, x, scope='d1'):
        sx, sy = self.size
        sx += 2
        sy += 2

        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            h = x
            for d in range(self.nl):
                h = fully_connected(h, 2**d*self.nh, activation_fn=tf.nn.relu)
            y = fully_connected(h, 2, activation_fn=tf.identity)
            y += x[:, :2]
            ch = fully_connected(h, 2, activation_fn=tf.identity)
            # ch = tf.stack((ch[:, 0], tf.abs(ch[:, 1])), axis=-1)
            # ch = ch/(tf.norm(ch, axis=-1, keepdims=True) + 1e-12)

            out = tf.concat((y, ch), axis=-1)

        return out

    @staticmethod
    def get_trainable_vars():
        return tf.trainable_variables(
            scope='d1') + tf.trainable_variables(scope='d2')

    def apply(self, imgs, x_s, y_s, name):
        """Apply diffeomorphism, move the separate channels to batch dimension
        so that diffeo can be applied in parallel"""

        batch_size = tf.shape(imgs)[0]
        height = tf.shape(imgs)[1]
        width = tf.shape(imgs)[2]
        nchannels = tf.shape(imgs)[3]

        _, h, w, nc = imgs.get_shape().as_list()

        # to have zero outside the image domain, 
        # this increases img_size by 2
        pad_imgs = tf.pad(
            imgs, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")

        imgs_ch_in_batch = move_ch_to_batch(pad_imgs)

        x_s = tf.tile(x_s, (batch_size, 1, 1))
        y_s = tf.tile(y_s, (batch_size, 1, 1))

        warp = bilinear_sampler(imgs_ch_in_batch, x_s, y_s)
        warp = move_ch_back(warp, nchannels)

        warp = warp[:, 1:height+1, 1:width+1, :]

        warp = tf.reshape(warp, (-1, h, w, nc))

        return warp

def transform(data):
    data[:, :2, :] = data[:, :2, :]/1000
    return data


def make_eval_data(idx):
    bv = np.load('box_vecs_unit.npy')[1:49]
    print('Choosing box %d for evaluation' % idx)
    bv = bv[idx].reshape(1, 2)

    x = np.linspace(0.2, 0.8, 10)
    X, Y = np.meshgrid(x, x)
    grid = np.stack((X.ravel(), Y.ravel()), axis=-1)
    bv = np.tile(bv, [len(grid), 1])

    inputs = np.concatenate((grid, bv), axis=-1)

    return inputs.astype(np.float32)


def make_plot(inputs, out1, out2):
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))

    ax[0].scatter(inputs[:, 0], inputs[:, 1], c='b')
    ax[0].scatter(out1[:, 0], out1[:, 1], c='k', marker='+', s=20)
    ax[0].arrow(0.5, 0.5, 0.1*inputs[0, 2], 0.1*inputs[0, 3], color='r')

    ax[1].scatter(inputs[:, 0], inputs[:, 1], c='b')
    ax[1].scatter(out2[:, 0], out2[:, 1], c='k', marker='+', s=20)

    plt.savefig('change.png')
    plt.close()


def train_yxi2xeta():

    IS = 128
    box_npy = np.load('box_vecs_unit.npy')
    diffnet = SingleBigDiffeo((IS, IS), box_npy, nhidden=32, nlayers=5)

    data_d1 = transform(np.load(
        '../HJ_flow/d1_gl_yxi2xeta.npy').astype(np.float32))
    data_d1_t = tf.convert_to_tensor(data_d1, dtype=tf.float32)

    data_d2 = transform(np.load(
        '../HJ_flow/d2_gl_yxi2xeta.npy').astype(np.float32))
    data_d2_t = tf.convert_to_tensor(data_d2, dtype=tf.float32)


    bs = 1024
    dataset = tf.data.Dataset.from_tensor_slices((data_d1_t, data_d2_t))
    dataset = dataset.prefetch(3*bs)
    dataset = dataset.shuffle(3*bs)
    dataset = dataset.repeat()
    dataset = dataset.batch(bs)

    data = dataset.make_one_shot_iterator().get_next()
    x1 = data[0][:, :, 0]
    y1 = data[0][:, :, 1]

    x2 = data[1][:, :, 0]
    y2 = data[1][:, :, 1]

    BS = tf.shape(x1)[0]

    logdir = 'hj_flow_yxi2xeta_restanh_ws_estimate/'
    try:
        os.mkdir(logdir)
    except:
        print('%s exists!' % logdir)
        pass

    yhat1 = diffnet.train_net(x1, scope='d1')
    yhat2 = diffnet.train_net(x2, scope='d2')
    
    wavespeeds = diffnet.estimate_wavespeed(diffnet.grid)

    origin = tf.zeros((1, 2), dtype=tf.float32)

    def get_val_index(arr, IS):
        return tf.cast(tf.clip_by_value(tf.floor(arr*(IS+2)), 0, IS+1), tf.int32)

    origin_c = diffnet.estimate_wavespeed(origin)

    c_y = diffnet.estimate_wavespeed(x1[:,:2])
    c_x = diffnet.estimate_wavespeed(yhat1[:,:2])
    
    xi_norm = tf.norm(x1[:,2:], axis=-1)
    eta_norm = tf.norm(yhat1[:,2:], axis=-1)

    d1_hamil_loss = tf.reduce_sum(tf.square(c_x*xi_norm - c_y*eta_norm))

    c_y = diffnet.estimate_wavespeed(x2[:,:2])
    c_x = diffnet.estimate_wavespeed(yhat2[:,:2])

    xi_norm = tf.norm(x2[:,2:], axis=-1)
    eta_norm = tf.norm(yhat2[:,2:], axis=-1)

    d2_hamil_loss = tf.reduce_sum(tf.square(c_x*xi_norm - c_y*eta_norm))

    ref_loss = 100*tf.reduce_sum(tf.square(origin_c - 1))

    img_inp = tf.convert_to_tensor(
        np.load('inp_wp_test.npy'), dtype=tf.float32)
    img_out = tf.convert_to_tensor(
        np.load('out_wp_test.npy'), dtype=tf.float32)

    x_s_d1 = []
    y_s_d1 = []
    x_s_d2 = []
    y_s_d2 = []
    for i in range(1, 49):
        x, y, c = diffnet.net(i, 'd1')
        x_s_d1.append(x)
        y_s_d1.append(y)

        x, y, c = diffnet.net(i, 'd2')
        x_s_d2.append(x)
        y_s_d2.append(y)
        print(i)

    x_s_d1 = tf.concat(x_s_d1, axis=0)
    y_s_d1 = tf.concat(y_s_d1, axis=0)

    x_s_d2 = tf.concat(x_s_d2, axis=0)
    y_s_d2 = tf.concat(y_s_d2, axis=0)




    # warp1 = diffnet.apply(img_inp[:, :, :, 1:49], x_s_d1, y_s_d1, 'warp_d1')
    # warp2 = diffnet.apply(img_inp[:, :, :, 1:49], x_s_d2, y_s_d2, 'warp_d2')

    w_t = tf.convert_to_tensor(np.array([100, 100, 100, 100]), dtype=tf.float32)

    loss_d1 = tf.reduce_sum(tf.square(y1 - yhat1)*w_t)/tf.cast(BS, tf.float32)
    loss_d2 = tf.reduce_sum(tf.square(y2 - yhat2)*w_t)/tf.cast(BS, tf.float32)

    hamil_loss = d1_hamil_loss + d2_hamil_loss + ref_loss
    loss = loss_d1 + loss_d2 + 1e-4*hamil_loss

    ws_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='wavespeed')

    print(ws_vars)
    print(sum([np.prod(v.get_shape().as_list())
               for v in tf.trainable_variables()]))

    train_step = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)
    train_step_ws = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(
        hamil_loss, var_list=ws_vars)
    run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer(), options=run_options)

        ckpt = tf.train.latest_checkpoint(logdir)
        if ckpt:
            saver.restore(sess, ckpt)
            print('checkpoint restored')
            # store the warped images
            # w1, w2 = sess.run([warp1, warp2])

            # np.save(logdir+'w1.npy', w1)
            # np.save(logdir+'w2.npy', w2)

        t = time()
        for i in range(100000):

            _, _, l_v, lh_v, y1_v, y2_v, yh1_v, yh2_v = sess.run([
                train_step, train_step_ws, loss, hamil_loss, y1, y2, yhat1, yhat2])

            if i % 100 == 0:
                print('That took %fs' % (time()-t))
                print('Loss at %d = %f, hamil_loss = %f' % (i, l_v, lh_v))
                t = time()

            if i % 1000 == 0:
                print('Required d1 ')
                print(y1_v[0])

                print('d1 estimate')
                print(yh1_v[0])

                print('Required d2 ')
                print(y2_v[0])

                print('d2 estimate')
                print(yh2_v[0])

            if i % 1000 == 0:
                x_s1, x_s2, y_s1, y_s2, c_v = sess.run(
                    [x_s_d1, x_s_d2, y_s_d1, y_s_d2, wavespeeds])
                np.save(logdir+'d1.npy', np.stack((x_s1, y_s1), axis=0))
                np.save(logdir+'d2.npy', np.stack((x_s2, y_s2), axis=0))
                np.save(logdir+'speed.npy', c_v)

                # t1h_v, t1_v = sess.run([t1hat, t1])
                # l_v = np.sum(np.linalg.norm(t1h_v - t1_v, axis=0))/100
                # print('Loss = %f'%l_v)
                # sh1_v, sh2_v = sess.run([shat1, shat2],
                #     feed_dict={t1: t1_eval, t2: t2_eval})

                # print('Eval')
                # print('Required source')
                # print(source_eval[:5])

                # print('d1 source estimate')
                # print(sh1_v[:5])

                # print('d2 source estimate')
                # print(sh2_v[:5])
                # make_plot(source_eval, sh1_v, sh2_v)
                saver.save(sess, logdir+'model', global_step=i)


def get_diff_grids(diffnet, box_npy, size=(8, 8)):
    with tf.variable_scope('check_diffs'):
        grid = diffnet.create_cartesian_grid(size)
        print('Grid shape')
        print(grid.get_shape().as_list())
        grids = []
        npts = tf.shape(grid)[0]
        N = len(box_npy)
        for i in range(1, N-1):
            bv = tf.convert_to_tensor(
                box_npy[i].reshape(1, 2), dtype=tf.float32)
            bgrid = tf.concat((grid, tf.tile(bv, (npts, 1))), axis=-1)
            grids.append(bgrid)

        grids = tf.concat(grids, axis=0)

    x_d1, y_d1, _, _ = diffnet.net_w_inp(grids, sx=size[0], sy=size[1], scope='d1')

    return (x_d1, y_d1)

def train_xt2xy():

    box_npy = np.load('box_vecs_unit.npy')
    diffnet = SingleBigDiffeo((128, 128), box_npy, nhidden=32, nlayers=3)

    data_d1 = np.load(
        '../HJ_flow/inp_seis_0p4.npy').astype(np.float32)
    data_d1_t = tf.convert_to_tensor(data_d1, dtype=tf.float32)

    data_d2 = np.load(
        '../HJ_flow/out_seis_0p4.npy').astype(np.float32)
    data_d2_t = tf.convert_to_tensor(data_d2, dtype=tf.float32)

    bs = 1024
    dataset = tf.data.Dataset.from_tensor_slices((data_d1_t, data_d2_t))
    dataset = dataset.prefetch(3*bs)
    dataset = dataset.shuffle(3*bs)
    dataset = dataset.repeat()
    dataset = dataset.batch(bs)

    data = dataset.make_one_shot_iterator().get_next()
    inp_diff = data[0]
    req_diff = data[1]

    BS = tf.shape(inp_diff)[0]

    logdir = 'hj_flow_xt/'
    try:
        os.mkdir(logdir)
    except:
        print('%s exists!' % logdir)
        pass

    yhat1 = diffnet.train_net(inp_diff, scope='d1')
    x_s_d1, y_s_d1 = get_diff_grids(diffnet, box_npy)
    w_t = tf.convert_to_tensor(np.array([100, 100, 1, 1]), dtype=tf.float32)

    loss = tf.reduce_sum(tf.square(req_diff - yhat1)*w_t)/tf.cast(BS, tf.float32)


    print(tf.trainable_variables())
    print(sum([np.prod(v.get_shape().as_list())
               for v in tf.trainable_variables()]))

    train_step = tf.train.AdamOptimizer(learning_rate=1e-5).minimize(loss)
    run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer(), options=run_options)

        ckpt = tf.train.latest_checkpoint(logdir)
        if ckpt:
            saver.restore(sess, ckpt)
            print('checkpoint restored')
            # # store the warped images
            x1_v, y1_v = sess.run([x_s_d1, y_s_d1])
            np.save(logdir+'d1.npy', np.stack([x1_v, y1_v], axis=-1))
            # np.save(logdir+'w1.npy', w1)

            # w1, w2 = sess.run([warp1, warp2])

            # np.save(logdir+'w1.npy', w1)
            # np.save(logdir+'w2.npy', w2)

        t = time()
        for i in range(100000):

            _, l_v, y_v, yh1_v = sess.run([
                train_step, loss, req_diff, yhat1])

            if i % 100 == 0:
                print('That took %fs' % (time()-t))
                print('Loss at %d = %f' % (i, l_v))
                t = time()

            if i % 1000 == 0:
                print('Required d1 ')
                print(y_v[0])

                print('d1 estimate')
                print(yh1_v[0])


            if i % 10000 == 0:
                x1_v, y1_v = sess.run([x_s_d1, y_s_d1])
                np.save(logdir+'d1.npy', np.stack([x1_v, y1_v], axis=0))
                # x_s1, x_s2, y_s1, y_s2 = sess.run(
                #     [x_s_d1, x_s_d2, y_s_d1, y_s_d2])
                # np.save(logdir+'x_s1.npy', x_s1)
                # np.save(logdir+'x_s2.npy', x_s2)
                # np.save(logdir+'y_s1.npy', y_s1)
                # np.save(logdir+'y_s2.npy', y_s2)

                # t1h_v, t1_v = sess.run([t1hat, t1])
                # l_v = np.sum(np.linalg.norm(t1h_v - t1_v, axis=0))/100
                # print('Loss = %f'%l_v)
                # sh1_v, sh2_v = sess.run([shat1, shat2],
                #     feed_dict={t1: t1_eval, t2: t2_eval})

                # print('Eval')
                # print('Required source')
                # print(source_eval[:5])

                # print('d1 source estimate')
                # print(sh1_v[:5])

                # print('d2 source estimate')
                # print(sh2_v[:5])
                # make_plot(source_eval, sh1_v, sh2_v)
                saver.save(sess, logdir+'model', global_step=i)



if __name__ == '__main__':
    train_yxi2xeta()
    # train_xt2xy()
