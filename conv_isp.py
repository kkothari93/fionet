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


def adverserial_loss(yhat, y):

    from discriminator import discriminator

    def gradient_penalty(d, yhat, y):
        bs = tf.shape(yhat)[0]
        alpha = tf.random_uniform(shape=[bs, 1, 1, 1], minval=0., maxval=1.)
        differences = yhat - y
        interpolates = y + (alpha * differences)
        gradients = tf.gradients(
            d(interpolates, reuse=True), [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(
            tf.square(gradients), reduction_indices=[1, 2, 3]))
        gradient_penalty = tf.reduce_mean((slopes-1.)**2)

        return gradient_penalty

    d_hat = discriminator(yhat)
    d_real = discriminator(y, reuse=True)
    gp = gradient_penalty(discriminator, yhat, y)

    D_loss = tf.reduce_mean(d_real) - tf.reduce_mean(d_hat)  + gp
    G_loss = tf.reduce_mean(d_hat)  # + 10*gp

    disc_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')

    clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in disc_vars]

    return D_loss, G_loss, clip_D, disc_vars


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


def isp():



    def conv_part(x1, x2):
        from UNet import UNet

        inp_d1 = UNet(config={'nblocks': 3,
                              'start_ch': 16,
                              'out_ch': 50,
                              'kernel_size': 3}).net(x1, 'lr_d1')

        inp_d2 = UNet(config={'nblocks': 3,
                              'start_ch': 16,
                              'out_ch': 50,
                              'kernel_size': 3}).net(x2, 'lr_d2')
        # with tf.variable_scope('lr_d1'):
        #     inp_d1 = conv2d(x1,
        #                     kernel_size=3,
        #                     num_outputs=100,
        #                     activation_fn=tf.nn.relu)
        #     inp_d1 = conv2d(inp_d1,
        #                     kernel_size=3,
        #                     num_outputs=50,
        #                     activation_fn=tf.identity)

        # with tf.variable_scope('lr_d2'):
        #     inp_d2 = conv2d(x2,
        #                     kernel_size=3,
        #                     num_outputs=100,
        #                     activation_fn=tf.nn.relu)
        #     inp_d2 = conv2d(inp_d2,
        #                     kernel_size=3,
        #                     num_outputs=50,
        #                     activation_fn=tf.identity)

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
        DATA_DIR+'explosion_thick_lines_30k_0p2/', batch_size=BS)

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

    face_inp_path = DATA_DIR+'explosion_face_refl_0p2_final_p_cl.npy'
    face_out_path = DATA_DIR+'explosion_face_refl_0p2_source_p_cl.npy'
    face_in, face_out = np.load(face_inp_path)[
        :BS], np.load(face_out_path)[:BS]

    """Set up directories"""
    logdir = LOG_DIR
    griddir = logdir+'grids/'
    gendir_shapes = logdir+'gen_shapes/'
    gendir_lines = logdir+'gen_lines/'
    gendir_mnist = logdir+'gen_mnist/'
    gendir_refl = logdir+'gen_refl/'
    gendir_face = logdir+'gen_refl/'

    safemkdir(logdir)
    safemkdir(griddir)
    safemkdir(gendir_shapes)
    safemkdir(gendir_mnist)
    safemkdir(gendir_lines)
    safemkdir(gendir_refl)

    """Set up flows"""
    box_npy = np.load('box_vecs_unit.npy')
    diffnet = convSingleBigDiffeo((128, 128), box_npy, nhidden=32, nlayers=3)

    bg_vec = tf.Variable(tf.random.normal(
        (1, 3), dtype=tf.float32), name='bg_params')
    bg_vec_sig = tf.concat([tf.nn.sigmoid(bg_vec), 
        tf.constant([[3.0/8.0, 1.0]], dtype=tf.float32)], axis=-1)

    params = tf.concat(
        (tf.tile(bg_vec_sig, [48, 1]), diffnet.bc[1:49]), axis=-1)
    print(params.get_shape().as_list())

    flows = diffnet.net(params)

    inp_sum = tf.reduce_sum(inp, keepdims=True, axis=-1)

    inp_d1, inp_d2 = conv_part(inp, inp)
    # inp_d1, inp_d2 = inp, inp

    # d1_out, d2_out = get_diff_grids(diffnet, box_npy, size=(128, 128))

    zero = tf.constant([0], tf.float32)
    l_d1, l_d2 = zero, zero

    # x_s_d1, y_s_d1, l_d1 = d1_out
    # x_s_d2, y_s_d2, l_d2 = d2_out

    x_s_d1 = flows[:, 64-1:192+1, 64-1:192+1, 0]
    y_s_d1 = flows[:, 64-1:192+1, 64-1:192+1, 1]

    x_s_d2 = flows[:, 64-1:192+1, 64-1:192+1, 4]
    y_s_d2 = flows[:, 64-1:192+1, 64-1:192+1, 5]

    print(x_s_d1.get_shape().as_list())

    warp1 = diffnet.apply(inp_d1[:, :, :, 1:49],
                          x_s_d1, y_s_d1, 'warp_d1')

    warp2 = diffnet.apply(inp_d2[:, :, :, 1:49],
                          x_s_d2, y_s_d2, 'warp_d2')

    net_out = warp1 + warp2


    net_out_sum = tf.reduce_sum(net_out, axis=-1, keepdims=True)
    req_fio_sum = tf.reduce_sum(req[:, :, :, 1:49], axis=-1, keepdims=True)
    req_full_sum = tf.reduce_sum(req, axis=-1, keepdims=True)

    out_arti = net_out_sum

    """Setup losses"""
    bs = tf.shape(req)[0]
    norm_net_out = tf.norm(tf.reshape(net_out, (-1,)), axis=-1)
    norm_req_fio = tf.norm(tf.reshape(req_fio_sum, (-1,)), axis=-1)

    loss_diff = tf.reduce_sum(tf.square(net_out_sum-req_fio_sum)) / \
        tf.cast(bs, tf.float32)
    # loss_diff = tf.reduce_sum(tf.square(req_renorm-net_out_renorm)) / \
    #     tf.cast(bs, tf.float32)
    # loss_diff = tf.reduce_sum(tf.square(net_out-req[:, :, :, 1:49])) / \
    #     tf.cast(bs, tf.float32)

    # D_loss, G_loss, clip_D, disc_vars = adverserial_loss(
    #     tf.abs(net_out_sum), tf.abs(req_fio_sum))

    loss_sing = l_d1 + l_d2

    loss = loss_diff

    # loss_artifact = tf.reduce_sum(
    #     tf.square(out_arti-req_full_sum))/tf.cast(bs, tf.float32)

    """Collect different variables"""
    d1_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='lr_d1')
    d2_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='lr_d2')
    # print('\ntrainable_variables: lr\n')
    # print(d1_vars + d2_vars)

    diffd1_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='flow_field')
    diffd2_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='bg_params')
    # diffd2_vars = []

    print('\ntrainable_variables: diff\n')
    print(diffd1_vars + diffd2_vars)

    artifact_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='artifact')
    print('\n artifact_vars\n')
    print(artifact_vars)

    """Setup optimizers"""
    def build_optimizer(loss, gvars, step, 
        optimizer=None, 
        lr=1e-4,
        gen_summary=False):

        ret = {}
        
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

    lr_opt = build_optimizer(loss, d1_vars+d2_vars, step, lr=1e-4, gen_summary=True)
    train_step_lr, lr_grad_summ = lr_opt['op'], lr_opt['grad_summ_op']

    diff_opt = build_optimizer(loss, diffd1_vars+diffd2_vars, step, lr=5e-6, gen_summary=True)
    train_step_diff, diff_grad_summ = diff_opt['op'], diff_opt['grad_summ_op']

    # train_step_artifact = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(
    #     loss_artifact, var_list=artifact_vars, global_step=step)
    # D_solver = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(
    #     D_loss, var_list=disc_vars, global_step=step)
    # G_solver = tf.train.AdamOptimizer(learning_rate=1e-5).minimize(
    #     G_loss, var_list=diffd1_vars + d1_vars + d2_vars, global_step=step)

    saver_diff = tf.train.Saver(var_list=diffd1_vars)
    saver_net = tf.train.Saver()
    # saver_arti = tf.train.Saver(var_list=artifact_vars)

    # scalar_summ = tf.summary.merge([tf.summary.scalar('D_loss', D_loss),
    #     tf.summary.scalar('G_loss', G_loss),
    #     tf.summary.scalar('loss', loss)])
    scalar_summ = tf.summary.merge([tf.summary.scalar('loss', loss)])

    summary_op = tf.summary.merge([lr_grad_summ, diff_grad_summ, scalar_summ])

    do_diff_load = True

    def data_save(sess, gvars, dirname, data_in, data_out):
        """Save the data from the ops"""
        if data_in is None or data_out is None:
            xi, xi_lr1, xi_lr2, o, w1, w2, oa, r = sess.run(gvars)
        else:
            xi, xi_lr1, xi_lr2, o, w1, w2, oa, r = sess.run(
                gvars, feed_dict={inp: data_in, req: data_out})

        np.save(dirname+'inp.npy', xi)
        np.save(dirname+'inp_lr1.npy', xi_lr1)
        np.save(dirname+'inp_lr2.npy', xi_lr2)
        np.save(dirname+'out.npy', o)
        np.save(dirname+'req.npy', r)
        np.save(dirname+'o1.npy', w1)
        np.save(dirname+'o2.npy', w2)
        np.save(dirname+'oa.npy', oa)

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

        t = time()
        for i in range(100000):

            # for _ in range(3):
            #     sess.run([D_solver, clip_D])

            if i > HEAD_START:
                ops = [step, train_step_diff, train_step_lr]#, G_solver]
            else:
                ops = [step, train_step_diff]

            # get global step
            gs = sess.run(ops)[0]

            ops_to_save = [inp, inp_d1, inp_d2,
                           net_out, warp1, warp2, out_arti, req]
            gridops = [x_s_d1, y_s_d1, x_s_d2, y_s_d2]

            def save_grids(sess, i, gridops, griddir):
                # save the grids
                xsd1, ysd1, xsd2, ysd2 = sess.run(
                    [x_s_d1, y_s_d1, x_s_d2, y_s_d2])

                np.save(griddir + 'd1_%d.npy' %
                        i, np.stack((xsd1, ysd1), axis=0))
                np.save(griddir + 'd2_%d.npy' %
                        i, np.stack((xsd2, ysd2), axis=0))

            if i % 100 == 0:
                print('That took %fs' % (time()-t))
                # l, ls, bg_v, d_l, g_l = sess.run(
                #     [loss, loss_sing, bg_vec_sig, D_loss, G_loss])
                # print('Loss at %d = %f, %f' % (i, d_l, g_l))
                l, ls, bg_v = sess.run(
                    [loss, loss_sing, bg_vec_sig])
                print('Loss at %d = %f, %f' % (i, l, ls))
                print(bg_v[0])
                t = time()

            if i % 1000 == 0:
                data_save(sess, ops_to_save, logdir, None, None)
                data_save(sess, ops_to_save, gendir_shapes,
                          shapes_in, shapes_out)
                data_save(sess, ops_to_save, gendir_mnist, mnist_in, mnist_out)
                data_save(sess, ops_to_save, gendir_refl, refl_in, refl_out)
                data_save(sess, ops_to_save, gendir_lines, lines_in, lines_out)
                save_grids(sess, i, gridops, griddir)

                summ = sess.run(summary_op)
                summ_writer.add_summary(summ, global_step=gs)

                # save the model
                saver_net.save(sess, logdir+'model', global_step=gs)
                # saver_arti.save(sess, logdir+'arti', global_step=gs)

    return


if __name__ == '__main__':
    isp()
