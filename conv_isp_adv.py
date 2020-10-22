import os
import glob
import numpy as np
import tensorflow as tf
from time import time
from tqdm import tqdm
from tensorflow.contrib.layers import fully_connected, conv2d

from utils import *

from SingleBigDiffeo import convSingleBigDiffeo
from tfrecords import read_tfrecords

tf.reset_default_graph()
tf.random.set_random_seed(1)
np.random.seed(0)

FLAGS, unparsed = flags()

HEAD_START = FLAGS.head_start
DIFF_LOAD_DIR = FLAGS.diff_load_dir
LOG_DIR = FLAGS.log_dir
DATA_DIR = FLAGS.data_dir
DIFF_LR = FLAGS.diff_lr
GROUND_METRIC = FLAGS.ground_metric
W2_ALPHA = FLAGS.w2_alpha
TRAIN = FLAGS.train
D_ITER = FLAGS.diter

USE_ONLY_DIFF_BRANCH=False

def save_params():
    with open(os.path.join(LOG_DIR,'params.txt'), 'w+') as f:
        f.write(str(FLAGS))

def adverserial_loss(yhat, y, ret_summ_slopes=True, ground_metric='l2'):

    from discriminator import discriminator

    def gp(grads, interpolates):
        slopes = tf.sqrt(tf.reduce_sum(
            tf.square(grads), reduction_indices=[1, 2, 3]))
        if ret_summ_slopes:
            summ_slopes = tf.summary.histogram('slopes_summary', slopes)
        else:
            summ_slopes = None

        gradient_penalty = tf.reduce_mean((slopes-1.)**2)
        return gradient_penalty, summ_slopes 

    def gp_w2(grads, interpolates, nbr_rad = 7):
        """
        calculate the gradient in W2 metric in pixel space
        Algorithm 2 of http://proceedings.mlr.press/v97/dukler19a/dukler19a.pdf
        """
        if nbr_rad%2==0:
            nbr_rad += 1

        def compute_d(nbr_rad, IS):
            """Computes degree of each node"""
            from scipy.ndimage.filters import convolve

            N = nbr_rad
            
            a = np.arange(N)
            xx, yy = np.meshgrid(a,a)
            c = np.array([[xx[N//2,N//2], yy[N//2,N//2]]])
            
            kernel = np.linalg.norm(np.stack((xx.ravel(), yy.ravel()), axis=-1) - c, axis=-1)
            kernel /= kernel.sum()
            kernel = kernel.reshape(N,N)

            ones = np.ones((IS,IS))
            out = convolve(ones, kernel, mode='constant', cval=0)
            out /= out.sum()

            return out.reshape(1,IS,IS,1)

        def compute_kernels(nbr_rad):
            N = nbr_rad
            Nsqm1 = N**2 - 1

            a = np.arange(N**2)

            K = np.zeros((N, N, N**2))
            K[a%N,a//N, a] = -1 
            K[N//2,N//2, :] = 1
            # delete ii term
            K = np.delete(K, N**2//2, -1)
            
            M = np.zeros_like(K)
            M[a%N,a//N,a] = 0.5
            M[N//2,N//2,:] = 0.5
            # delete ii term
            M = np.delete(M, N**2//2, -1)
            
            # conv2d requires shape [filter_height, filter_width, in_channels, out_channels]
            return K.reshape(N,N,1,Nsqm1), M.reshape(N,N,1,Nsqm1)


        IS = grads.get_shape().as_list()[1] # get image size
        d = compute_d(nbr_rad, IS)
        K, M = compute_kernels(nbr_rad)

        K = tf.convert_to_tensor(K, dtype=tf.float32)
        M = tf.convert_to_tensor(M, dtype=tf.float32)
        d = tf.convert_to_tensor(d, dtype=tf.float32)

        # alpha D^T 11^T D
        DT_one = tf.reduce_sum(grads, axis=1, keepdims=True)
        oneT_D = tf.transpose(DT_one, [0,2,1,3])
        add_term = W2_ALPHA * DT_one * oneT_D
 
        H = tf.nn.conv2d(grads, K, [1,]*4, "SAME")
        V = tf.nn.conv2d(interpolates/d, M, [1,]*4, "SAME")
        
        H = H*H
        W = V*H

        W = tf.reduce_sum(W, reduction_indices=[1,2,3])
        slopes = tf.math.sqrt(W + add_term)
        if ret_summ_slopes:
            summ_slopes = tf.summary.merge([
                tf.summary.histogram('slopes_summary', slopes),
                tf.summary.scalar()])
        else:
            summ_slopes = None

        return tf.reduce_mean(tf.square(slopes-1)**2), summ_slopes

    def gradient_penalty(d, yhat, y):
        """WGAN-GP loss of I. Gulrajani et al (2017)"""
        bs = tf.shape(yhat)[0]
        alpha = tf.random_uniform(shape=[bs, 1, 1, 1], minval=0., maxval=1.)
        differences = yhat - y
        interpolates = y + (alpha * differences)
        gradients = tf.gradients(
            d(interpolates, reuse=True), [interpolates])[0]
        if ground_metric=='l2':
            gp_func = gp
        else:
            gp_func = gp_w2


        return gp_func(gradients, interpolates)

    d_hat = discriminator(yhat)
    d_real = discriminator(y, reuse=True)

    gploss, summ_slopes = gradient_penalty(discriminator, yhat, y)

    D_loss = tf.reduce_mean(d_hat) - tf.reduce_mean(d_real)  + 10*gploss
    G_loss = -tf.reduce_mean(d_hat)

    summ_slopes = tf.summary.merge([summ_slopes,
        tf.summary.scalar(
            'd_score', tf.reduce_mean(d_real)),
        tf.summary.scalar(
            'g_score', tf.reduce_mean(d_hat)),
        tf.summary.scalar(
            'WofW', tf.reduce_mean(d_real) - tf.reduce_mean(d_hat))])

    disc_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')

    return D_loss, G_loss, disc_vars, summ_slopes


def save_weights(ckpt, name='wts.npy'):
    """
    saves weight matrices, avoids optimizer weights
    """
    ckpt_vars = tf.train.list_variables(ckpt)
    ckpt_vars = [v for v in ckpt_vars if 'Adam' not in v[0]
                 and 'beta' not in v[0]]
    print(ckpt_vars)

    tvars = []
    for v, s in ckpt_vars:
        t = v.split('/')
        scope = '/'.join(t[:-1])
        with tf.variable_scope(scope, reuse=True):
            tvars.append(sess.run(tf.get_variable(t[-1])))

    dct = {}
    for k, (name, shape) in enumerate(ckpt_vars):

        nm_str = name.replace('/', '_')
        print(nm_str)
        dct[nm_str] = tvars[k]

    np.save(logdir + name, dct)


def main():

    def conv_part(x1, x2):
        from UNet import UNet

        inp_d1 = UNet(config={'nblocks': 5,
                              'start_ch': 32,
                              'out_ch': 50,
                              'kernel_size': 3}).net(x1, 'lr_d1')

        inp_d2 = UNet(config={'nblocks': 5,
                              'start_ch': 32,
                              'out_ch': 50,
                              'kernel_size': 3}).net(x2, 'lr_d2')

        return inp_d1, inp_d2


    """Diff grids"""
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

        x_d1, y_d1, loss_d1 = diffnet.net_w_inp(
            grids, scope='d1', return_sing_loss=True)
        x_d2, y_d2, loss_d2 = diffnet.net_w_inp(
            grids, scope='d2', return_sing_loss=True)

        return (x_d1, y_d1, loss_d1), (x_d2, y_d2, loss_d2)

    BS = 8

    step = tf.train.get_or_create_global_step()

    inp, req = read_tfrecords(
        DATA_DIR+'explosion_thick_lines_bg2_0p2/', batch_size=BS)

    # """Set up generalization data"""
    shapes_inp_path = DATA_DIR+'explosion_random_shapes_0p2_final_p_cl.npy'
    shapes_out_path = DATA_DIR+'explosion_random_shapes_0p2_source_p_cl.npy'
    shapes_in, shapes_out = np.load(shapes_inp_path)[
        :BS], np.load(shapes_out_path)[:BS]

    mnist_inp_path = DATA_DIR+'explosion_mnist_gen_0p2_final_p_cl.npy'
    mnist_out_path = DATA_DIR+'explosion_mnist_gen_0p2_source_p_cl.npy'
    mnist_in, mnist_out = np.load(mnist_inp_path)[
        :BS], np.load(mnist_out_path)[:BS]

    refl_inp_path = DATA_DIR+'explosion_reflectors_2_0p2_final_p_cl.npy'
    refl_out_path = DATA_DIR+'explosion_reflectors_2_0p2_source_p_cl.npy'
    refl_in, refl_out = np.load(refl_inp_path)[
        :BS], np.load(refl_out_path)[:BS]

    lines_inp_path = DATA_DIR+'explosion_thick_lines_gen_0p2_final_p_cl.npy'
    lines_out_path = DATA_DIR+'explosion_thick_lines_gen_0p2_source_p_cl.npy'
    lines_in, lines_out = np.load(lines_inp_path)[
        :BS], np.load(lines_out_path)[:BS]

    faces_inp_path = DATA_DIR+'explosion_face_refl_0p2_final_p_cl.npy'
    faces_out_path = DATA_DIR+'explosion_face_refl_0p2_source_p_cl.npy'
    faces_in, faces_out = np.load(faces_inp_path)[
        :BS], np.load(faces_out_path)[:BS]

    """Set up directories"""
    logdir = LOG_DIR
    griddir = logdir+'grids/'
    gendir_shapes = logdir+'gen_shapes/'
    gendir_lines = logdir+'gen_lines/'
    gendir_mnist = logdir+'gen_mnist/'
    gendir_refl = logdir+'gen_refl/'
    gendir_faces = logdir+'gen_faces/'

    safemkdir(logdir)
    safemkdir(griddir)
    safemkdir(gendir_shapes)
    safemkdir(gendir_mnist)
    safemkdir(gendir_lines)
    safemkdir(gendir_refl)
    safemkdir(gendir_faces)

    save_params()

    """Set up flows"""
    box_npy = np.load('box_vecs_unit.npy')
    diffnet = convSingleBigDiffeo((128, 128), box_npy, nhidden=32, nlayers=3)

    # This is a random code we begin with, for some reason setting the seed
    # would still change this vector everytime, so an explicit constant is
    # set here for reproducibility purposes.
    bg_vec = tf.constant([[ 1.92812687, -1.18299809, -0.61859749]], dtype=tf.float32)

    # the input z to routing network needs to be between 0 and 1
    bg_vec_sig = tf.concat([tf.nn.sigmoid(bg_vec), 
        tf.constant([[3.0/8.0, 1.0]], dtype=tf.float32)], axis=-1)

    # input vector to routing network
    params = tf.concat(
        (tf.tile(bg_vec_sig, [48, 1]), diffnet.bc[1:49]), axis=-1)

    # gives the grids
    flows = diffnet.net(params)

    """Conv portion of the newtork"""
    inp_d1, inp_d2 = conv_part(inp, inp)


    inp_sum = tf.reduce_sum(inp, keepdims=True, axis=-1)

    zero = tf.constant([0], tf.float32)

    x_s_d1 = flows[:, 64-1:192+1, 64-1:192+1, 0]
    y_s_d1 = flows[:, 64-1:192+1, 64-1:192+1, 1]

    x_s_d2 = flows[:, 64-1:192+1, 64-1:192+1, 4]
    y_s_d2 = flows[:, 64-1:192+1, 64-1:192+1, 5]

    """Give head start of e_0 epochs to routing network"""
    # HEAD_START is based on number of iterations, e_0 * dataset_size / batch_size
    b1, b2 = tf.cond(step<HEAD_START, lambda: (inp, inp), lambda: (inp_d1, inp_d2))

    warp1 = diffnet.apply(b1[:, :, :, 1:49],
                          x_s_d1, y_s_d1, 'warp_d1')

    warp2 = diffnet.apply(b2[:, :, :, 1:49],
                          x_s_d2, y_s_d2, 'warp_d2')

    net_out = warp1 + warp2

    # Calculated for diagnostic purposes.
    diff_img = diffnet.apply(inp[:, :, :, 1:49], x_s_d1, y_s_d1, 'warp_d1') + \
            diffnet.apply(inp[:, :, :, 1:49], x_s_d2, y_s_d2, 'warp_d2')

    # Compute the targets
    net_out_sum = tf.reduce_sum(net_out, axis=-1, keepdims=True)
    req_fio_sum = tf.reduce_sum(req[:, :, :, 1:49], axis=-1, keepdims=True)


    """Setup losses"""
    bs = tf.shape(req)[0]
    norm_net_out = tf.norm(tf.reshape(net_out, (-1,)), axis=-1)
    norm_req_fio = tf.norm(tf.reshape(req_fio_sum, (-1,)), axis=-1)

    loss_diff = tf.reduce_sum(tf.square(net_out_sum-req_fio_sum)) / \
        tf.cast(bs, tf.float32)

    D_loss, G_loss, disc_vars, summ_slopes = adverserial_loss(
        tf.abs(net_out_sum), tf.abs(req_fio_sum))

    loss = loss_diff

    """Collect different variables"""
    d1_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='lr_d1')
    d2_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='lr_d2')
    print('\ntrainable_variables: lr\n')
    print(d1_vars + d2_vars)

    diff_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='flow_field')

    print('\ntrainable_variables: diff\n')
    print(diff_vars)

    """Setup optimizers"""
    def build_optimizer(loss, gvars, step,
        optimizer=None, 
        lr=1e-4,
        gen_summary=False):

        ret = {}
        if len(gvars) == 0:
            return {'op': tf.no_op(), 'grad_summ_op': tf.no_op()}

        
        if optimizer is None:
            opt = tf.train.AdamOptimizer(learning_rate=lr)
        else:
            opt = opt(learning_rate=lr)

        ret['optimizer'] = opt
        ret['grads'] = opt.compute_gradients(loss, var_list=gvars)

        if gen_summary:
            grad_summ_op = tf.summary.merge([tf.summary.histogram(
                "%s-grad" % g[1].name, g[0]) for g in ret['grads']])
            ret['grad_summ_op']  = grad_summ_op

        if step is None:
            ret['op'] = opt.apply_gradients(ret['grads'])
        else:
            ret['op'] = opt.apply_gradients(ret['grads'], global_step=step)
        
        return ret

    """Set up otimizers"""
    # two different learning rates are required for routing and the low rank (lr) network
    # we also set up summaries for the low rank (lr) part
    lr_opt = build_optimizer(loss, d1_vars + d2_vars, None, lr=1e-4, gen_summary=True)
    train_step_lr, lr_grad_summ = lr_opt['op'], lr_opt['grad_summ_op']

    diff_opt = build_optimizer(loss, diff_vars, step, lr=DIFF_LR, gen_summary=True)
    train_step_diff, diff_grad_summ = diff_opt['op'], diff_opt['grad_summ_op']

    # D refers to the critic network
    D_solver = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(
        D_loss, var_list=disc_vars)
    
    # G refers to the wave packet routing network
    G_solver = tf.train.AdamOptimizer(learning_rate=DIFF_LR).minimize(
        G_loss, var_list=diff_vars, global_step=step)

    """set up savers"""
    saver_diff = tf.train.Saver(var_list=diff_vars)
    saver_net = tf.train.Saver()

    """set up loss summaries"""
    scalar_summ = tf.summary.merge([tf.summary.scalar('D_loss', D_loss),
        tf.summary.scalar('G_loss', G_loss),
        tf.summary.scalar('loss', loss)])

    summary_op = tf.summary.merge_all()

    do_diff_load = True

    def data_save(sess, gvars, dirname, data_in, data_out, grids_d1, grids_d2):
        """Save the data from the ops"""
        feed_dict = {}

        if grids_d1 is not None:
            xsd1 = grids_d1[0]
            ysd1 = grids_d1[1]

            xsd2 = grids_d2[0]
            ysd2 = grids_d2[1]

            grid_dict = {x_s_d1: xsd1, 
                        y_s_d1: ysd1,
                        x_s_d2: xsd2,
                        y_s_d2: ysd2}

            feed_dict.update(grid_dict)

        if data_in is not None:
            feed_dict.update({inp: data_in, req: data_out})

        if USE_ONLY_DIFF_BRANCH:
            feed_dict.update({step:0})

        xi, xi_lr1, xi_lr2, o, w1, w2, r = sess.run(
                gvars, feed_dict=feed_dict)
        diffeo_img = sess.run(diff_img, feed_dict=feed_dict)

        np.save(dirname+'inp.npy', xi)
        np.save(dirname+'inp_lr1.npy', xi_lr1)
        np.save(dirname+'inp_lr2.npy', xi_lr2)
        np.save(dirname+'out.npy', o)
        np.save(dirname+'req.npy', r)
        np.save(dirname+'o1.npy', w1)
        np.save(dirname+'o2.npy', w2)
        np.save(dirname+'diff_img.npy', diffeo_img)

        return None

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    summ_writer = tf.summary.FileWriter(logdir)


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())  # , options=run_options)

        # preload various parts of the network
        ckpt = tf.train.latest_checkpoint(logdir)
        if ckpt:
            saver_net.restore(sess, ckpt)
            print('Loaded network successfully!')
            print(ckpt)
            do_diff_load = False

        ckpt = tf.train.latest_checkpoint(DIFF_LOAD_DIR)
        if ckpt and do_diff_load:
            diffnet.load(sess, ckpt, saver_diff)
            print('Preload of diffeo successful')
            print(ckpt)


        ops_to_save = [inp, inp_d1, inp_d2,
                       net_out, warp1, warp2, req]
        gridops = [x_s_d1, y_s_d1, x_s_d2, y_s_d2]

        def save_grids(sess, i, gridops, griddir):
            # save the grids
            xsd1, ysd1, xsd2, ysd2 = sess.run(
                gridops)

            np.save(griddir + 'd1_%d.npy' %
                    i, np.stack((xsd1, ysd1), axis=0))
            np.save(griddir + 'd2_%d.npy' %
                    i, np.stack((xsd2, ysd2), axis=0))

        t = time()
        if not TRAIN:
            diff_dir = os.path.join(DIFF_LOAD_DIR, 'grids/')
            if os.path.exists(diff_dir):
                def extract_num(fname):
                    """fname format d1_{}.npy"""
                    fname = fname.split('/')[-1]
                    return int(fname[3:].split('.')[0])
                
                d1 = max(glob.glob(diff_dir+'d1_*.npy'), key=extract_num)
                d2 = max(glob.glob(diff_dir+'d2_*.npy'), key=extract_num)

            grids_d1 = np.load(d1)
            grids_d2 = np.load(d2)

            griddir = logdir+'grids_test/'
            gendir_shapes = logdir+'gen_shapes/'
            gendir_lines = logdir+'gen_lines/'
            gendir_mnist = logdir+'gen_mnist/'
            gendir_faces = logdir+'gen_faces/'
            gendir_refl = logdir+'gen_refl/'
            safemkdir(griddir)
            safemkdir(gendir_shapes)
            safemkdir(gendir_mnist)
            safemkdir(gendir_lines)
            safemkdir(gendir_faces)
            safemkdir(gendir_refl)

            data_save(sess, ops_to_save, gendir_shapes, shapes_in, shapes_out, grids_d1, grids_d2)
            data_save(sess, ops_to_save, gendir_faces, faces_in, faces_out, grids_d1, grids_d2)
            data_save(sess, ops_to_save, gendir_mnist, mnist_in, mnist_out, grids_d1, grids_d2)
            data_save(sess, ops_to_save, gendir_refl, refl_in, refl_out, grids_d1, grids_d2)
            data_save(sess, ops_to_save, gendir_lines, lines_in, lines_out, grids_d1, grids_d2)
            save_grids(sess, 1351, gridops, griddir) # 1351 is an arbitrary choice.

            return 

        print('starting training')
        for i in range(100000):

            if i < HEAD_START:
                for _ in range(D_ITER):
                    sess.run([D_solver])      
                
                ops = [step, G_solver]

            else:
                ops = [step, train_step_lr, train_step_diff]

            # get global step
            gs = sess.run(ops)[0]


            if i % 100 == 0:
                print('That took %fs' % (time()-t))
                l, d_l, g_l = sess.run(
                    [loss, D_loss, G_loss])
                print('Loss at %d = DL: %f, GL: %f' % (i, d_l, g_l))
                print('Loss at %d = %f' % (i, l))
                t = time()

            if i % 1000 == 0:
                data_save(sess, ops_to_save, logdir, None, None, None, None)
                data_save(sess, ops_to_save, gendir_shapes, shapes_in, shapes_out, None, None)
                data_save(sess, ops_to_save, gendir_mnist, mnist_in, mnist_out, None, None)
                data_save(sess, ops_to_save, gendir_refl, refl_in, refl_out, None, None)
                data_save(sess, ops_to_save, gendir_lines, lines_in, lines_out, None, None)
                save_grids(sess, i, gridops, griddir)

                summ = sess.run(summary_op)
                summ_writer.add_summary(summ, global_step=gs)

                # save the model
                saver_net.save(sess, logdir+'model', global_step=gs)

    return


if __name__ == '__main__':
    main()
