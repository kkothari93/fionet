import os
from .stn
import numpy as np
import tensorflow as tf
from time import time
from tensorflow.contrib.layers import fully_connected
from tensorflow.contrib.image import dense_image_warp


def bilinear_sampler(img, x, y, c):
    """
    Performs bilinear sampling of the input images according to the
    normalized coordinates provided by the sampling grid. Note that
    the sampling is done identically for each channel of the input.
    To test if the function works properly, output image should be
    identical to input image when theta is initialized to identity
    transform.
    Input
    -----
    - img: batch of images in (B, H, W, C) layout.
    - grid: x, y which is the output of affine_grid_generator.
    Returns
    -------
    - out: interpolated images according to grids. Same size as grid.
    """
    H = tf.shape(img)[1]
    W = tf.shape(img)[2]
    max_y = tf.cast(H - 1, 'int32')
    max_x = tf.cast(W - 1, 'int32')
    zero = tf.zeros([], dtype='int32')

    # rescale x and y to [0, W-1/H-1]
    x = tf.cast(x, 'float32')
    y = tf.cast(y, 'float32')
    x = x * tf.cast(max_x-1, 'float32')
    y = y * tf.cast(max_y-1, 'float32')

    # grab 4 nearest corner points for each (x_i, y_i)
    x0 = tf.cast(tf.floor(x), 'int32')
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), 'int32')
    y1 = y0 + 1

    # clip to range [0, H-1/W-1] to not violate img boundaries
    x0 = tf.clip_by_value(x0, zero, max_x)
    x1 = tf.clip_by_value(x1, zero, max_x)
    y0 = tf.clip_by_value(y0, zero, max_y)
    y1 = tf.clip_by_value(y1, zero, max_y)

    # get pixel value at corner coords
    Ia = get_pixel_value(img, x0, y0, c)
    Ib = get_pixel_value(img, x0, y1, c)
    Ic = get_pixel_value(img, x1, y0, c)
    Id = get_pixel_value(img, x1, y1, c)

    # recast as float for delta calculation
    x0 = tf.cast(x0, 'float32')
    x1 = tf.cast(x1, 'float32')
    y0 = tf.cast(y0, 'float32')
    y1 = tf.cast(y1, 'float32')

    # calculate deltas
    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    # add dimension for addition
    # wa = tf.expand_dims(wa, axis=3)
    # wb = tf.expand_dims(wb, axis=3)
    # wc = tf.expand_dims(wc, axis=3)
    # wd = tf.expand_dims(wd, axis=3)

    # compute output
    out = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])
    out = tf.expand_dims(out, axis=3)

    return out


def get_pixel_value(img, x, y, c):
    """
    Utility function to get pixel value for coordinate
    vectors x and y from a  4D tensor image.
    Input
    -----
    - img: tensor of shape (B, H, W, C)
    - x: flattened tensor of shape (B*H*W,)
    - y: flattened tensor of shape (B*H*W,)
    Returns
    -------
    - output: tensor of shape (B, H, W, C)
    """
    shape = tf.shape(img)
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]

    batch_idx = tf.range(0, batch_size)
    batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))
    b = tf.tile(batch_idx, (1, height, width))


    indices = tf.stack([b, y, x, c], 3)

    result = tf.gather_nd(img, indices)

    return result

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
        sx+=2
        sy+=2
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            b = tf.tile(tf.reshape(self.bc[boxid], (1, 2)), 
                [sx*sy, 1])
            x = tf.concat((self.grid, b), axis=-1)
            h = fully_connected(x, self.nh, activation_fn=tf.nn.relu)
            h = fully_connected(h, self.nh, activation_fn=tf.nn.relu)
            y = fully_connected(h, 2, activation_fn=tf.nn.sigmoid)
            ch = fully_connected(h, 2, activation_fn=tf.nn.sigmoid)
            ch = tf.stack((ch[:,0], tf.abs(ch[:,1])), axis=-1)
            # ch = tf.tensordot(ch, self.bc, axes=[[-1], [-1]])
            # ch = tf.argmax(ch, axis=-1)

        y = self.grid - y
        y = y*tf.convert_to_tensor(np.array([[sx, sy]], dtype=np.float32))

        return y, ch

    def load(self, sess, ckpt, saver):
        saver.restore(sess, ckpt)
        return

    def train_net(self, x, scope='d1'):
        sx, sy = self.size
        sx+=2
        sy+=2
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            h = fully_connected(x, self.nh, activation_fn=tf.nn.relu)
            h = fully_connected(h, self.nh, activation_fn=tf.nn.relu)
            y = fully_connected(h, 2, activation_fn=tf.nn.sigmoid)
            ch = fully_connected(h, 2, activation_fn=tf.nn.tanh)
            ch = tf.stack((ch[:,0], tf.abs(ch[:,1])), axis=-1)

            out = tf.concat((y,ch), axis=-1)

        return out

    @staticmethod
    def get_trainable_vars():
        return tf.trainable_variables(
            scope='d1') + tf.trainable_variables(scope='d2')

    def single_ch_apply(self, imgs, flow_vecs, name):
        """Apply diffeomorphism"""
        shape = tf.shape(imgs)
        batch_size = shape[0]
        imgs = tf.reshape(imgs, (batch_size,)+self.size+(1,))
        sx, sy = self.size
        sx = sx+2
        sy = sy+2

        pad_imgs = tf.pad(
            imgs, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
        # reshaping imgs
        imgs_for_api = tf.reshape(pad_imgs, (-1, sx, sy, 1))
        print(imgs_for_api.get_shape().as_list())

        # first add the batch_dimension
        flow_vecs = tf.expand_dims(flow_vecs, axis=0)

        # next repeat the flow vecs batch_times
        flow_vecs = tf.tile(flow_vecs, [batch_size, 1, 1])
        flow_vecs = tf.reshape(flow_vecs, (batch_size, sx*sy, 2))
        flow_vecs = tf.reshape(flow_vecs, (-1, sx, sy, 2))

        warp = dense_image_warp(
            imgs_for_api, flow_vecs,
            name='warp_diffnet_'+name)

        # get channels back in last axis
        warp = warp[:, 1:-1, 1:-1, :]
        warp = tf.reshape(warp, (batch_size, sx-2, sy-2, 1))

        return warp


    def apply_old(self, imgs, x_s, y_s, interp, name):
        """Apply diffeomorphism, move the separate channels to batch dimension
        so that diffeo can be applied in parallel"""

        batch_size = tf.shape(imgs)[0]

        interp = tf.abs(tf.tensordot(interp, self.bc, axes=[[-1],[-1]]))
        c = tf.math.argmax(interp, axis=-1, output_type=tf.int32, name='argmax')

        diagnostics = [c, interp]

        x_s = tf.tile(x_s, (batch_size, 1, 1))
        y_s = tf.tile(y_s, (batch_size, 1, 1))
        c = tf.tile(c, (batch_size, 1, 1))

        # to have zero outside the image domain, this increases
        # img_size by 2
        pad_imgs = tf.pad(
            imgs, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")

        warp = bilinear_sampler(pad_imgs, x_s, y_s, c)


def test_justdiffeo():
    from jspaceDataGenerator import get_iterator, dataGenerator
    from time import time
    # out_path = '/home/konik/Documents/FIO/data/explosion_thick_lines_0p2_final_p.npy'
    # inp_path = '/home/konik/Documents/FIO/data/explosion_thick_lines_0p2_source_p.npy'
    # inp_path = '/home/konik/Documents/FIO/data/wp_in_0p2_128.npy'
    # out_path = '/home/konik/Documents/FIO/data/wp_out_0p2_128.npy'
    out_path = '/mnt/ext6TB/fio/data/fisheye5k.npy'
    inp_path = '/mnt/ext6TB/fio/data/originals20k.npy'

    iterator = get_iterator(inp_path, out_path, 4, 2000)

    inp, req = iterator.get_next()
    print('Set up iterator')
    box_npy = np.load('box_vecs.npy')
    print('Loaded box vectors')

    diffnet = SingleBigDiffeo((128, 128), box_npy)

    # inp = tf.reduce_sum(inp, axis=-1, keepdims=True)
    # req = tf.reduce_sum(req, axis=-1, keepdims=True)

    flow_vecs1, interp1 = diffnet.net(0, scope='d1')
    flow_vecs2, interp2 = diffnet.net(0, scope='d2')
    out1 = diffnet.apply(inp, flow_vecs1, interp1, 'd1')
    out2 = diffnet.apply(inp, flow_vecs2, interp2, 'd2')

    out = out1 + out2

    loss = tf.reduce_sum(tf.square(out-req))

    train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        t = time()
        for i in range(10000):
            sess.run(train_step)
            if i % 10 == 0:
                print('That took %fs' % (time()-t))
                l = sess.run(loss)
                print('Loss at %d = %f' % (i, l))
                t = time()
            if i%100 == 0:
                xi, o, o1, o2, r = sess.run([inp, out, out1, out2, req])
                f1, i1 = sess.run([flow_vecs1, interp1])
                np.save('inp_sd.npy', xi)
                np.save('out_sd.npy', o)
                np.save('f1_sd.npy', f1)
                np.save('i1_sd.npy', i1)
                np.save('out1_sd.npy', o1)
                np.save('out2_sd.npy', o2)
                np.save('req_sd.npy', r)

def train_d1d2():

    def loss(yhat, y):
        poshat = yhat[:,:2]
        pos = y[:,:2]
        cond = tf.cast(tf.math.not_equal(pos, -1), tf.float32)
        pos_loss = tf.reduce_sum(cond*tf.square(pos-poshat))

        chhat = yhat[:, 2:4]
        ch = y[:, 2:4]
        # ch_loss = tf.reduce_sum(
        #     1 - tf.reduce_sum(ch*chhat, axis=-1))
        ch_loss = tf.reduce_sum(cond*tf.square(ch-chhat))

        return ch_loss + pos_loss

    box_npy = np.load('box_vecs.npy')
    diffnet = SingleBigDiffeo((512, 512), box_npy)
    x1 = tf.placeholder(tf.float32, [None, 4])
    x2 = tf.placeholder(tf.float32, [None, 4])
    y1 = tf.placeholder(tf.float32, [None, 4])
    y2 = tf.placeholder(tf.float32, [None, 4])

    d1_in = np.load('d1_in.npy')
    d2_in = np.load('d2_in.npy')
    d1_out = np.load('d1_out.npy')
    d2_out = np.load('d2_out.npy')
    N = len(d1_in)

    d1_in_test = d1_in[-500:]
    d2_in_test = d2_in[-500:]
    d1_out_test = d1_out[-500:]
    d2_out_test = d2_out[-500:]

    d1_in = d1_in[:-500]
    d2_in = d2_in[:-500]
    d1_out = d1_out[:-500]
    d2_out = d2_out[:-500]

    logdir = 'pretrain_diff_v1/'
    try:
        os.mkdir(logdir)
    except:
        print('%s exists!'%logdir)
        pass


    yhat1 = diffnet.train_net(x1, scope='d1')
    yhat2 = diffnet.train_net(x2, scope='d2')

    # loss1 = tf.reduce_sum(tf.square(y1-yhat1))
    # loss2 = tf.reduce_sum(tf.square(y2-yhat2))

    loss_d1 = loss(yhat1, y1)
    loss_d2 = loss(yhat2, y2)

    loss = loss_d1+loss_d2

    print(tf.trainable_variables())
    print(sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))

    train_step = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)
    run_options = tf.RunOptions(report_tensor_allocations_upon_oom = True)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer(), options=run_options)

        t = time()
        bs = 64
        for i in range(100000):
            k = i%(N//bs)
            
            feed_dict={
                x1:d1_in[k*bs:(k+1)*bs],
                y1:d1_out[k*bs:(k+1)*bs],
                x2:d2_in[k*bs:(k+1)*bs],
                y2:d2_out[k*bs:(k+1)*bs]
                }
            
            sess.run(train_step, feed_dict=feed_dict, options=run_options)
            
            if i % 100 == 0:
                print('That took %fs' % (time()-t))
                l = sess.run(loss, feed_dict=feed_dict)
                print('Loss at %d = %f' % (i, l))
                t = time()

            if i%1000==0:
                k = i%(500//bs)
                feed_dict={
                    x1:d1_in_test[k*bs:(k+1)*bs],
                    y1:d1_out_test[k*bs:(k+1)*bs],
                    x2:d2_in_test[k*bs:(k+1)*bs],
                    y2:d2_out_test[k*bs:(k+1)*bs]
                    }
                valhat, val = sess.run([yhat1, y1], feed_dict=feed_dict)
                print(valhat[0])
                print(val[0])

            if i%10000==0:
                saver.save(sess, logdir+'model', global_step=i)






def test_singlebigdiffeo():

    from jspaceDataGenerator import get_iterator, dataGenerator
    from time import time
    # out_path = '/home/konik/Documents/FIO/data/explosion_thick_lines_0p2_final_p.npy'
    # inp_path = '/home/konik/Documents/FIO/data/explosion_thick_lines_0p2_source_p.npy'
    inp_path = '/home/konik/Documents/FIO/data/wp_in_0p2_128.npy'
    out_path = '/home/konik/Documents/FIO/data/wp_out_0p2_128.npy'


    iterator = get_iterator(inp_path, out_path, 8, 1000)

    inp, req = iterator.get_next()
    print('Set up iterator')
    box_npy = np.load('box_vecs.npy')
    print('Loaded box vectors')

    diffnet = SingleBigDiffeo((128, 128), box_npy)

    net_out = []
    d1 = []
    d2 = []
    ch1 = []
    f1 = []
    for i in range(1, 49):
        flow_vec1, interp1 = diffnet.net(i, scope='d1')
        flow_vec2, interp2 = diffnet.net(i, scope='d2')

        d1_out = diffnet.apply(inp, flow_vec1, interp1, 'd1')
        d2_out = diffnet.apply(inp, flow_vec2, interp2, 'd2')

        d1.append(d1_out)
        d2.append(d2_out)

        ch1.append(interp1)
        f1.append(flow_vec1)

        k_out = d1_out + d2_out
        print(i)

        net_out.append(k_out)


    net_out = tf.concat(net_out, axis=-1)
    d1 = tf.concat(d1, axis=-1)
    d2 = tf.concat(d2, axis=-1)

    f1 = tf.stack(f1, axis=-1)
    ch1 = tf.stack(ch1, axis=-1)

    print(sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
    print(tf.trainable_variables())
    loss = tf.reduce_sum(tf.square(net_out-req[:, :, :, 1:49]))

    train_step = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)
    # run_options = tf.RunOptions(report_tensor_allocations_upon_oom = True)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())#, options=run_options)
        ckpt = tf.train.latest_checkpoint('./pretrain_diff_v1/')
        print(ckpt)
        diffnet.load(sess, ckpt, saver)
        print('preload successful')

        t = time()
        for i in range(10000):
            sess.run(train_step)#, options=run_options)
            if i % 10 == 0:
                print('That took %fs' % (time()-t))
                l = sess.run(loss)
                print('Loss at %d = %f' % (i, l))
                t = time()
            if i%100 == 0:
                xi, o, r, o1, o2, ch1_val, f1_val = sess.run([inp, net_out, req, d1, d2, ch1, f1])
                np.save('inp.npy', xi)
                np.save('out.npy', o)
                np.save('req.npy', r)
                np.save('o1.npy', o1)
                np.save('o2.npy', o2)
                np.save('ch1.npy', ch1_val)
                np.save('f1.npy', f1_val)



def EikonalNet():
    """This network decides where each wave coefficient should go

    architecture is as follows:

    Inputs
    ------
    One hot vector indicating v,k would go in
    and one-normalized (j1, j2) would also go in

    Outputs
    --------
    A softmax over differnt v,k s and a corresponding (j1,j2)
    normalized between 0,2

    Here if the output is greater than one, assume that that wavepacket
    left the domain 

    /TODO:
    Currently I have a fully connected architecture here
    but perhaps we need to put in more structure

    for each v,k I need a 2D output indicating the j index
    which means for each we need a 100D output or we can also input the 
    one hot vector for each box and ask for the j index there
    """

    def __init__(self, loss=None, nboxes=50, scope='eikonal'):
        self.nboxes = nboxes
        if loss is not None:
            self.loss = loss
        else:
            self.loss = 0

    def __call__(self, x):
        """x will be a one-hot encoding of the box along with
        2 scalars in the last two dimensions. The 2 scalars 
        indicated normalized j1, j2"""

        h = Dense(512, activation=tf.nn.relu)(x)
        h = Dense(128, activation=tf.nn.relu)(h)
        h = Dense(52, activation=tf.nn.relu)(h)

        return h

    def __new__(self):
        pass


if __name__ == '__main__':
    train_d1d2()
    test_singlebigdiffeo()
