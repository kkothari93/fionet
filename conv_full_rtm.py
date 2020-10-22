import os
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
D_ITER = FLAGS.diter

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


def rtm():

    def conv_part(x1):
        from UNet import UNet

        inp_d1 = UNet(config={'nblocks': 5,
                              'start_ch': 32,
                              'out_ch': 50,
                              'kernel_size': 3}).net(x1, 'lr_d1')

        # inp_d1 = conv2d(x1, 100, 5, activation_fn=tf.nn.leaky_relu)
        # inp_d1 = conv2d(inp_d1, 100, 3, activation_fn=tf.nn.leaky_relu)
        # inp_d1 = conv2d(inp_d1, 50, 3, activation_fn=tf.identity)
        return inp_d1

    BS = 8

    mean_inp = tf.convert_to_tensor(np.load('../data/mean_full_rtm_extend.npy'), tf.float32)

    step = tf.train.get_or_create_global_step()

    inp, req = read_tfrecords(
        DATA_DIR+'explosion_full_rtm_refl_3_0p2_sens_trace/', batch_size=BS, img_size=128)

    inp_wo_direct = inp - mean_inp

    # """Set up generalization data"""
    shapes_inp_path = DATA_DIR+'explosion_full_rtm_shapes_0p2_sens_trace_cl.npy'
    shapes_out_path = DATA_DIR+'explosion_full_rtm_shapes_0p2_source_p_cl.npy'
    shapes_in, shapes_out = np.load(shapes_inp_path)[:BS], np.load(shapes_out_path)[:BS]

    waves_inp_path = DATA_DIR+'explosion_full_rtm_waves_0p2_sens_trace_cl.npy'
    waves_out_path = DATA_DIR+'explosion_full_rtm_waves_0p2_source_p_cl.npy'
    waves_in, waves_out = np.load(waves_inp_path)[:BS], np.load(waves_out_path)[:BS]

    refl_inp_path = DATA_DIR+'explosion_full_rtm_expl_0p2_sens_trace_cl.npy'
    refl_out_path = DATA_DIR+'explosion_full_rtm_expl_0p2_source_p_cl.npy'
    refl_in, refl_out = np.load(refl_inp_path)[:BS], np.load(refl_out_path)[:BS]

    kate_inp_path = DATA_DIR+'explosion_full_rtm_rot_refl_0p2_sens_trace_cl.npy'
    kate_out_path = DATA_DIR+'explosion_full_rtm_rot_refl_0p2_source_p_cl.npy'
    kate_in, kate_out = np.load(kate_inp_path)[:BS], np.load(kate_out_path)[:BS]

    geo_inp_path = DATA_DIR+'explosion_full_rtm_geo_0p2_sens_trace_cl.npy'
    geo_out_path = DATA_DIR+'explosion_full_rtm_geo_0p2_source_p_cl.npy'
    geo_in, geo_out = np.load(geo_inp_path)[:BS], np.load(geo_out_path)[:BS]

    lines_inp_path = DATA_DIR+'explosion_full_rtm_lines_0p2_sens_trace_cl.npy'
    lines_out_path = DATA_DIR+'explosion_full_rtm_lines_0p2_source_p_cl.npy'
    lines_in, lines_out = np.load(lines_inp_path)[:BS], np.load(lines_out_path)[:BS]


    """Set up directories"""
    logdir = LOG_DIR
    griddir = logdir+'grids/'
    gendir_shapes = logdir+'gen_shapes/'
    gendir_lines = logdir+'gen_lines/'
    gendir_waves = logdir+'gen_waves/'
    gendir_kate = logdir+'gen_kate/'
    gendir_refl = logdir+'gen_refl/'
    gendir_geo = logdir+'gen_geo/'

    safemkdir(logdir)
    safemkdir(griddir)
    safemkdir(gendir_shapes)
    safemkdir(gendir_waves)
    safemkdir(gendir_lines)
    safemkdir(gendir_refl)
    safemkdir(gendir_kate)
    safemkdir(gendir_geo)

    save_params()

    """Set up flows"""
    box_npy = np.load('box_vecs_unit.npy')
    diffnet = convSingleBigDiffeo((128, 128), box_npy, nhidden=32, nlayers=3)

    bg_vec = tf.constant([[ 1.92812687, -1.18299809, -0.61859749]], dtype=tf.float32)
    bg_vec_sig = tf.concat([tf.nn.sigmoid(bg_vec), 
        tf.constant([[4.0/8.0, 1.0]], dtype=tf.float32)], axis=-1)

    bcs = diffnet.bc[1:49]
    pi_t = tf.constant(np.pi, dtype=tf.float32)
    angle = tf.math.acos(bcs[:,1]) 

    thresh = tf.cond(step > HEAD_START, lambda: pi_t/2, lambda: pi_t/6)

    open_mask = tf.cast((angle<thresh), tf.float32)

    params = tf.concat((tf.tile(bg_vec_sig, [48, 1]), bcs), axis=-1)
    print(params.get_shape().as_list())

    flows = diffnet.net(params)

    inp_sum = tf.reduce_sum(inp_wo_direct, keepdims=True, axis=-1)

    inp_d1 = conv_part(inp_wo_direct)

    zero = tf.constant([0], tf.float32)

    # us = tf.keras.layers.UpSampling2D(size=(2,2), interpolation='bilinear')
    us = tf.identity

    d1_grids = us(flows[:, 64-1:192+1, 64-1:192+1, :2])

    x_s_d1 = d1_grids[:,:,:,0]
    y_s_d1 = d1_grids[:,:,:,1]

    branch = tf.cond(step > HEAD_START, lambda: inp_d1, lambda: inp_wo_direct)

    warp1 = diffnet.apply(branch[:, :, :, 1:49],
                          x_s_d1, y_s_d1, 'warp_d1')
    net_out = warp1 

    net_out_sum = tf.reduce_sum(net_out, axis=-1, keepdims=True)
    req_fio_sum = tf.reduce_sum(req[:, :, :, 1:49]*open_mask, axis=-1, keepdims=True)
    req_full_sum = tf.reduce_sum(req, axis=-1, keepdims=True)

    out_arti = net_out_sum

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

        ret['op'] = opt.apply_gradients(ret['grads'], global_step=step)
        
        return ret

    lr_opt = build_optimizer(loss, d1_vars, None, lr=1e-4, gen_summary=True)
    train_step_lr, lr_grad_summ = lr_opt['op'], lr_opt['grad_summ_op']

    diff_opt = build_optimizer(loss, diff_vars, step, lr=DIFF_LR, gen_summary=True)
    train_step_diff, diff_grad_summ = diff_opt['op'], diff_opt['grad_summ_op']

    D_solver = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(
        D_loss, var_list=disc_vars)
    G_solver = tf.train.AdamOptimizer(learning_rate=DIFF_LR).minimize(
        G_loss, var_list=diff_vars+d1_vars, global_step=step)

    saver_diff = tf.train.Saver(var_list=diff_vars)
    saver_net = tf.train.Saver()

    scalar_summ = tf.summary.merge([tf.summary.scalar('D_loss', D_loss),
        tf.summary.scalar('G_loss', G_loss),
        tf.summary.scalar('loss', loss)])

    summary_op = tf.summary.merge_all()

    do_diff_load = True

    def data_save(sess, gvars, dirname, data_in, data_out):
        """Save the data from the ops"""
        if data_in is None or data_out is None:
            xi, xi_lr1, o, w1, r = sess.run(gvars)
        else:
            xi, xi_lr1, o, w1, r = sess.run(
                gvars, feed_dict={inp: data_in, req: data_out})

        np.save(dirname+'inp.npy', xi)
        np.save(dirname+'inp_lr1.npy', xi_lr1)
        np.save(dirname+'out.npy', o)
        np.save(dirname+'req.npy', r)
        np.save(dirname+'o1.npy', w1)

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
            do_diff_load = False

        ckpt = tf.train.latest_checkpoint(DIFF_LOAD_DIR)
        print(ckpt)
        if ckpt and do_diff_load:
            print('Training variables:')
            diffnet.load(sess, ckpt, saver_diff)
            print('Preload successful')

        print('starting training')
        ops_to_save = [inp, inp_d1, net_out, warp1, req]
        gridops = [x_s_d1, y_s_d1]

        def save_grids(sess, i, gridops, griddir):
            # save the grids
            xsd1, ysd1 = sess.run(gridops)

            np.save(griddir + 'd1_%d.npy' %
                    i, np.stack((xsd1, ysd1), axis=0))

        t = time()
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
                print('Loss at %d = DL:%f, GL:%f' % (i, d_l, g_l))
                print('Loss at %d = %f' % (i, l))
                t = time()

            if i % 1000 == 0:
                data_save(sess, ops_to_save, logdir, None, None)
                data_save(sess, ops_to_save, gendir_shapes, shapes_in, shapes_out)
                data_save(sess, ops_to_save, gendir_waves, waves_in, waves_out)
                data_save(sess, ops_to_save, gendir_refl, refl_in, refl_out)
                data_save(sess, ops_to_save, gendir_lines, lines_in, lines_out)
                data_save(sess, ops_to_save, gendir_kate, kate_in, kate_out)
                data_save(sess, ops_to_save, gendir_geo, geo_in, geo_out)
                save_grids(sess, i, gridops, griddir)

                summ = sess.run(summary_op)
                summ_writer.add_summary(summ, global_step=i)

                # save the model
                saver_net.save(sess, logdir+'model', global_step=i)

    return


if __name__ == '__main__':
    rtm()
