import numpy as np
import tensorflow as tf
from time import time
from utils import *
from tensorflow.contrib.layers import fully_connected, conv2d

class WPRN():
    """ 
    This network takes in the position and orientation of the 
    wave packet and spits out its position and orientation after 
    propagation over a given background. This is NOT used in the 
    FIONet. It is just used to generate training data for the actual
    routing network, ConvDiffeo which works on grids.
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
        loss = tf.constant([0.0], dtype=tf.float32)
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
                d_s = y-x[:,:2]
                d_s = tf.split(d_s, 48, axis=0, name='split')
                d_s = tf.stack(d_s, axis=-1)
                loss += tf.reduce_mean((tf.linalg.norm(tf.reduce_mean(d_s, axis=0), axis=0))**2)


            y = tf.reshape(y, [-1, sx, sy, 2])
            x = y[:, :, :, 0]
            y = y[:, :, :, 1]

            ch = tf.reshape(ch, [-1, sx, sy, 2])

        return x, y, loss


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

        N = self.bc.get_shape().as_list()[0] 

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

class convWPRN(WPRN):
    def __init__(self):
        super().__init__(size, box_center_npy, nhidden, nlayers)

    def net(self, x):
        ## random encoding to play well with the saver
        with tf.variable_scope('flow_field', reuse=tf.AUTO_REUSE):
            upsampler = tf.keras.layers.UpSampling2D(size=(2,2), interpolation='bilinear')
            c = fully_connected(x, 16, activation_fn = tf.nn.leaky_relu)
            c = fully_connected(c, 16*64, activation_fn = tf.nn.leaky_relu)

            c = tf.reshape(c, (-1, 4, 4, 64))

            h = upsampler(c)
            h = conv2d(h, 64, kernel_size=3)
            h = conv2d(h, 64, kernel_size=3)
            # [8x8]

            h = upsampler(h)
            h = conv2d(h, 64, kernel_size=3, activation_fn = tf.nn.leaky_relu)
            h = conv2d(h, 64, kernel_size=3, activation_fn = tf.nn.leaky_relu)
            # [16x16]

            h = upsampler(h)
            h = conv2d(h, 32, kernel_size=3, activation_fn = tf.nn.leaky_relu)
            h = conv2d(h, 32, kernel_size=3, activation_fn = tf.nn.leaky_relu)
            # [32x32]

            h = upsampler(upsampler(upsampler(h)))
            

            flows = conv2d(h, 8, kernel_size=1, activation_fn = tf.identity)

        return flows

 
        return 


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

    box_npy = np.load('box_vecs_unit.npy')
    diffnet = WPRN((128, 128), box_npy, nhidden=32, nlayers=3)

    data_d1 = transform(np.load(
        '../HJ_flow/d1_cons_yxi2xeta.npy').astype(np.float32))
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

    logdir = 'diff_dirs/hj_flow_yxi2xeta_ccbg/'
    try:
        os.mkdir(logdir)
    except:
        print('%s exists!' % logdir)
        pass

    yhat1 = diffnet.train_net(x1, scope='d1')
    yhat2 = diffnet.train_net(x2, scope='d2')
    xhat1 = diffnet.train_net(yhat1, scope='d2')
    xhat2 = diffnet.train_net(yhat2, scope='d1')

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


    w_t = tf.convert_to_tensor(np.array([100, 100, 100, 100]), dtype=tf.float32)

    loss_d1 = tf.reduce_sum(tf.square(y1 - yhat1)*w_t)/tf.cast(BS, tf.float32)
    loss_d2 = tf.reduce_sum(tf.square(y2 - yhat2)*w_t)/tf.cast(BS, tf.float32)

    loss_consistency = tf.reduce_sum(tf.square(x1 - xhat1)*w_t)/tf.cast(BS, tf.float32)
    loss_consistency += tf.reduce_sum(tf.square(x2 - xhat2)*w_t)/tf.cast(BS, tf.float32)

    loss = loss_d1 + loss_d2 + loss_consistency

    print(tf.trainable_variables())
    print(sum([np.prod(v.get_shape().as_list())
               for v in tf.trainable_variables()]))

    train_step = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)
    run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer(), options=run_options)

        ckpt = tf.train.latest_checkpoint(logdir)
        if ckpt:
            saver.restore(sess, ckpt)
            print('checkpoint restored')


        t = time()
        for i in range(100000):

            _, l_v, y1_v, y2_v, yh1_v, yh2_v = sess.run([
                train_step, loss, y1, y2, yhat1, yhat2])

            if i % 100 == 0:
                print('That took %fs' % (time()-t))
                print('Loss at %d = %f' % (i, l_v))
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
                x_s1, x_s2, y_s1, y_s2 = sess.run(
                    [x_s_d1, x_s_d2, y_s_d1, y_s_d2])
                np.save(logdir+'d1.npy', np.stack((x_s1, y_s1), axis=0))
                np.save(logdir+'d2.npy', np.stack((x_s2, y_s2), axis=0))

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

    x_d1, y_d1, _ = diffnet.net_w_inp(grids, sx=size[0], sy=size[1], scope='d1')

    return (x_d1, y_d1)

def train_rtm():
    box_npy = np.load('box_vecs_unit.npy')
    # theta = np.arccos(box_npy[:,0])*180/np.pi
    # cond = np.logical_and(theta>=20, theta<=160)

    ## this will keep the coarse and fine scales as they 
    ## will evaluate to pi/2 acc to the arccos metric
    # box_npy = box_npy[cond]
    N = len(box_npy)

    # theta = np.linspace(0,np.pi,50)
    # box_npy = np.stack((np.cos(theta), np.sin(theta)), axis=-1)
    
    diffnet = WPRN((128, 128), box_npy, nhidden=32, nlayers=3)
    data_d1 = np.load('../HJ_flow/d1_full_rtm_ext_gradient_2.npy').astype(np.float32)
    data_d1_t = tf.convert_to_tensor(data_d1, dtype=tf.float32)
    
    bs = 512
    dataset = tf.data.Dataset.from_tensor_slices(data_d1_t)
    dataset = dataset.prefetch(3*bs)
    dataset = dataset.shuffle(3*bs)
    dataset = dataset.repeat()
    dataset = dataset.batch(bs)

    data = dataset.make_one_shot_iterator().get_next()
    inp_diff = data[:,:,0]
    req_diff = data[:,:,1]

    BS = tf.shape(inp_diff)[0]

    logdir = 'hj_flow_full_rtm_ext_2/'
    try:
        os.mkdir(logdir)
    except:
        print('%s exists!' % logdir)
        pass

    # xi1tau = inp_diff[:,2:]/tf.linalg.norm(inp_diff[:,2:], axis=-1, keepdims=True)
    # inp_diff = tf.concat([inp_diff[:,:2], xi1tau], axis=-1)


    yhat1 = diffnet.train_net(inp_diff, scope='d1')
    x_s_d1, y_s_d1 = get_diff_grids(diffnet, box_npy, size=(128,128))
    w_t = tf.convert_to_tensor(np.array([100, 100, 1, 1]), dtype=tf.float32)
    loss = tf.reduce_sum(tf.square(req_diff - yhat1))/tf.cast(BS, tf.float32)


    bg_vec = tf.constant([[ 1.92812687, -1.18299809, -0.61859749]], dtype=tf.float32)
    bg_vec_sig = tf.concat([tf.nn.sigmoid(bg_vec), 
        tf.constant([[4.0/8.0, 1.0]], dtype=tf.float32)], axis=-1)
    params = tf.concat(
        (tf.tile(bg_vec_sig, [N-2, 1]), diffnet.bc[1:-1]), axis=-1)

    convdiff = convWPRN((128,128), box_npy, nhidden=32, nlayers=3)
    conv_grids = convdiff.net(params)

    conv_grids_x = conv_grids[:,64-1:192+1,64-1:192+1,0]
    conv_grids_y = conv_grids[:,64-1:192+1,64-1:192+1,1]

    loss_conv = tf.reduce_sum(tf.square(conv_grids_x - x_s_d1))/tf.cast(BS, tf.float32) + \
                tf.reduce_sum(tf.square(conv_grids_y - y_s_d1))/tf.cast(BS, tf.float32)

    diff_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='flow_field')
    print(diff_vars)

    # print(tf.trainable_variables())
    # print(sum([np.prod(v.get_shape().as_list())
    #            for v in tf.trainable_variables()]))

    train_step = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)
    train_step_conv = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(
        loss_conv, var_list=diff_vars)
    run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True)
    saver = tf.train.Saver(var_list=tf.trainable_variables())

    saver_diff = tf.train.Saver(diff_vars)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer(), options=run_options)

        ckpt = tf.train.latest_checkpoint(logdir)
        if ckpt:
            saver_diff.restore(sess, ckpt)
            print('checkpoint restored')
            # # store the warped images
            x1_v, y1_v, cg = sess.run([x_s_d1, y_s_d1, conv_grids])
            np.save(logdir+'d1.npy', np.stack([x1_v, y1_v], axis=-1))
            np.save(logdir+'dconv1.npy', cg)

        t = time()
        for i in range(40000):
            if i<10000:
                ops = [train_step, req_diff, yhat1, loss]
                _, y_v, yh1_v, l_v = sess.run(ops)
            else:
                ops = [train_step_conv, req_diff, yhat1, loss, loss_conv]
                _, y_v, yh1_v, l_v, lc_v = sess.run(ops)



            if (i+1) % 100 == 0:
                print('That took %fs' % (time()-t))
                print('Loss at %d = %f' % (i, l_v))
                if i > 10000:
                    print('Loss at %d = %f' % (i, lc_v))

                t = time()

            if (i+1) % 1000 == 0:
                print('Required d1 ')
                print(y_v[0])

                print('d1 estimate')
                print(yh1_v[0])

                ops = [inp_diff, req_diff]
                i_d, r_d = sess.run(ops)

                print(i_d[0])
                print(r_d[0])


            if (i+1) % 10000 == 0:
                x1_v, y1_v, cg = sess.run([x_s_d1, y_s_d1, conv_grids])
                np.save(logdir+'dconv1.npy', cg)
                np.save(logdir+'d1.npy', np.stack([x1_v, y1_v], axis=0))
                saver.save(sess, logdir+'model', global_step=i)
                saver_diff.save(sess, logdir+'diff', global_step=i)


def train_xt2xy():

    box_npy = np.load('box_vecs_unit.npy')
    diffnet = WPRN((128, 128), box_npy, nhidden=32, nlayers=3)

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
    # train_rtm()

