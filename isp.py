import os
import glob
import numpy as np
from time import time
import tensorflow as tf

from tfrecords import read_tfrecords
from WPRN import convWPRN
from tensorflow.contrib.layers import fully_connected, conv2d

from ops import *
from utils import *

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

def main():

    def conv_part(x1):
        from UNet import UNet

        inp_d1 = UNet(config={'nblocks': 5,
                              'start_ch': 16,
                              'out_ch': 50,
                              'kernel_size': 3}).net(x1, 'lr_d1')

        return inp_d1

    BS = 8

    step = tf.train.get_or_create_global_step()

    inp, req = read_tfrecords(
        DATA_DIR+'explosion_bs_thick_lines_0p2_sens_trace/', batch_size=BS)

    # """Set up generalization data"""
    shapes_inp_path = DATA_DIR+'explosion_bs_rand_shapes_0p2_sens_trace_cl.npy'
    shapes_out_path = DATA_DIR+'explosion_bs_rand_shapes_0p2_source_p_cl.npy'
    shapes_in, shapes_out = np.load(shapes_inp_path)[:BS], np.load(shapes_out_path)[:BS]

    mnist_inp_path = DATA_DIR+'explosion_bs_mnist_gen_0p2_sens_trace_cl.npy'
    mnist_out_path = DATA_DIR+'explosion_bs_mnist_gen_0p2_source_p_cl.npy'
    mnist_in, mnist_out = np.load(mnist_inp_path)[:BS], np.load(mnist_out_path)[:BS]

    refl_inp_path = DATA_DIR+'explosion_bs_bg2_refl_0p2_sens_trace_cl.npy'
    refl_out_path = DATA_DIR+'explosion_bs_bg2_refl_0p2_source_p_cl.npy'
    refl_in, refl_out = np.load(refl_inp_path)[:BS], np.load(refl_out_path)[:BS]

    lines_inp_path = DATA_DIR+'explosion_bs_thick_lines_0p2_sens_trace_cl.npy'
    lines_out_path = DATA_DIR+'explosion_bs_thick_lines_0p2_source_p_cl.npy'
    lines_in, lines_out = np.load(lines_inp_path)[:BS], np.load(lines_out_path)[:BS]

    """Set up directories"""
    logdir = LOG_DIR
    griddir = logdir+'grids/'
    gendir_shapes = logdir+'gen_shapes/'
    gendir_lines = logdir+'gen_lines/'
    gendir_mnist = logdir+'gen_mnist/'
    gendir_refl = logdir+'gen_refl/'

    safemkdir(logdir)
    safemkdir(griddir)
    safemkdir(gendir_shapes)
    safemkdir(gendir_mnist)
    safemkdir(gendir_lines)
    safemkdir(gendir_refl)

    save_params()

    """Set up flows"""
    box_npy = np.load('box_vecs_unit.npy')
    diffnet = convWPRN((128, 128), box_npy, nhidden=32, nlayers=3)

    # This is a random code we begin with, for some reason setting the seed
    # would still change this vector everytime, so an explicit constant is
    # set here for reproducibility purposes. 
    bg_vec = tf.constant([[ 1.92812687, -1.18299809, -0.61859749]], dtype=tf.float32)
    bg_vec_sig = tf.concat([tf.nn.sigmoid(bg_vec), 
        tf.constant([[3.0/8.0, 1.0]], dtype=tf.float32)], axis=-1)

    params = tf.concat(
        (tf.tile(bg_vec_sig, [48, 1]), diffnet.bc[1:49]), axis=-1)

    # gives the grids
    flows = diffnet.net(params)


    """Conv portion of the newtork"""
    inp_d1 = conv_part(inp)

    x_s_d1 = flows[:, 64-1:192+1, 64-1:192+1, 0]
    y_s_d1 = flows[:, 64-1:192+1, 64-1:192+1, 1]

    b1 = tf.cond(step > HEAD_START, lambda: inp_d1, lambda: inp)

    """Geometric module"""
    warp1 = diffnet.apply(b1[:, :, :, 1:49],
                          x_s_d1, y_s_d1, 'warp_d1')

    diff_img = diffnet.apply(inp[:, :, :, 1:49],
                          x_s_d1, y_s_d1, 'diff_out')
    net_out = warp1 

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

    """Set up otimizers"""
    
    # two different learning rates are required for routing and the low rank (lr) network
    # we also set up summaries for the low rank (lr) part
    lr_opt = build_optimizer(loss, d1_vars, None, lr=1e-4, gen_summary=True)
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

        # preload the network from previous checkpoint
        ckpt = tf.train.latest_checkpoint(logdir)
        if ckpt:
            saver_net.restore(sess, ckpt)
            print('Loaded network successfully!')
            do_diff_load = False

        # preload the pre-initialized routing network
        ckpt = tf.train.latest_checkpoint(DIFF_LOAD_DIR)
        print(ckpt)
        if ckpt and do_diff_load:
            print('Training variables:')
            diffnet.load(sess, ckpt, saver_diff)
            print('Preload of routing module successful')

        # outputs that need to be saved
        ops_to_save = [inp, inp_d1, net_out, warp1, req]

        # routing network outputs
        gridops = [x_s_d1, y_s_d1]

        def save_grids(sess, i, gridops, griddir):
            # save the grids
            xsd1, ysd1 = sess.run(gridops)

            np.save(griddir + 'd1_%d.npy' %
                    i, np.stack((xsd1, ysd1), axis=0))

        t = time()
        if not TRAIN:
            diff_dir = os.path.join(DIFF_LOAD_DIR, 'grids/')
            if os.path.exists(diff_dir):
                def extract_num(fname):
                    """fname format d1_{}.npy"""
                    fname = fname.split('/')[-1]
                    return int(fname[3:].split('.')[0])
                
                d1 = max(glob.glob(diff_dir+'d1_*.npy'), key=extract_num)

            grids_d1 = np.load(d1)

            griddir = logdir+'grids_test/'
            gendir_shapes = logdir+'gen_shapes/'
            gendir_lines = logdir+'gen_lines/'
            gendir_mnist = logdir+'gen_mnist/'
            gendir_refl = logdir+'gen_refl/'
            safemkdir(griddir)
            safemkdir(gendir_shapes)
            safemkdir(gendir_mnist)
            safemkdir(gendir_lines)
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
                print('Loss at %d = DL:%f, GL:%f' % (gs, d_l, g_l))
                print('Loss at %d = %f' % (gs, l))
                t = time()

            if i % 1000 == 0:
                data_save(sess, ops_to_save, logdir, None, None)
                data_save(sess, ops_to_save, gendir_shapes, shapes_in, shapes_out)
                data_save(sess, ops_to_save, gendir_mnist, mnist_in, mnist_out)
                data_save(sess, ops_to_save, gendir_refl, refl_in, refl_out)
                data_save(sess, ops_to_save, gendir_lines, lines_in, lines_out)
                save_grids(sess, i, gridops, griddir)

                summ = sess.run(summary_op)
                summ_writer.add_summary(summ, global_step=gs)

                # save the model
                saver_net.save(sess, logdir+'model', global_step=gs)

    return


if __name__ == '__main__':
    main()
