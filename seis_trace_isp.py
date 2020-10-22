# seismic trace --> inverse source problem

import os
import numpy as np
import tensorflow as tf
from time import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from tensorflow.contrib.layers import fully_connected, conv2d, max_pool2d

from tensorflow.python.client import timeline

from utils import *
from SingleBigDiffeo import SingleBigDiffeo

from jspaceDataGenerator import get_iterator, dataGenerator
from tfrecords import read_tfrecords
from tensorflow.keras.layers import UpSampling2D

import complexCurveNet as ccn


def prep_test_data(inp_path, out_path, BS=8):
    test_inp_data = np.load(inp_path).astype(np.float32)[:BS]
    test_out_data = np.load(out_path).astype(np.float32)[:BS]

    # lose channel dimension as mp_decompose does not like it!
    test_inp_data = test_inp_data[:, :, :, 0]
    test_out_data = test_out_data[:, :, :, 0]
    # test_inp_data = np.tile(test_inp_data, (1, 1, 1, 50))
    # test_out_data = np.tile(out_data, (1, 1, 1, 50))

    # I use a dummy generator sa I only need access to a pre-built CL
    # class without specifying class properties.
    dummygenerator = dataGenerator(inp_path, BS)
    test_inp_data = dummygenerator.cl.mp_decompose(
        test_inp_data)  # .sum(axis=-1, keepdims=True)
    test_out_data = dummygenerator.cl.mp_decompose(
        test_out_data)

    return test_inp_data, test_out_data

def from_seis_trace():

    def conv_part(x1):
        from UNet import UNet

        inp_d1 = UNet(config={'nblocks': 5,
                           'start_ch': 32,
                           'out_ch': 50,
                           'kernel_size': 3}).net(x1, 'lr_d1')

        
        # with tf.variable_scope('lr_d1'):

        #     upsamp = UpSampling2D((2,2), interpolation='bilinear')

        #     x1 = upsamp(x1)

        #     inp_d1 = conv2d(x1,
        #                     kernel_size=3,
        #                     num_outputs=100,
        #                     activation_fn=tf.nn.relu)
        #     inp_d1 = conv2d(inp_d1,
        #                     kernel_size=5,
        #                     num_outputs=100,
        #                     activation_fn=tf.nn.relu)
            
        #     inp_d1 = max_pool2d(inp_d1, 2)

        #     inp_d1 = conv2d(inp_d1,
        #                     kernel_size=3,
        #                     num_outputs=50,
        #                     activation_fn=tf.nn.relu)

        #     inp_d1 = conv2d(inp_d1,
        #                     kernel_size=3,
        #                     num_outputs=50,
        #                     activation_fn=tf.identity)


        return inp_d1

    # def conv_part(x1):
    #     import complexCurveNet as ccn
    #     def single_layer_ccn(x, act_fn, tile, name):
    #         _, y = ccn.complexCurveNetOS(tile, name).apply(x)
    #         y = act_fn(y)
    #         return y

    #     tile = np.load('../CurveNet/tiling256.npy')
        
    #     x = single_layer_ccn(x1, tf.nn.relu, tile, 'lr_d1/1')
    #     x = single_layer_ccn(x, tf.nn.relu, tile, 'lr_d1/2')
    #     x = single_layer_ccn(x, tf.nn.relu, tile, 'lr_d1/3')
    #     inp_d1 = single_layer_ccn(x, tf.identity, tile, 'lr_d1/4')


    #     return inp_d1

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

        sx = size[0] + 2
        sy = size[1] + 2

        print(grids.get_shape().as_list())
        # # npts = size[0]*size[1]*48
        # # size_splits = [10000,]*(npts//10000) + [npts - 10000*(npts//10000)]
        # # print(size_splits)
        # # grids_broken = tf.stack(tf.split(grids, 48, axis=0), axis=0)

        # # out_broken = tf.map_fn(diffnet.net_w_inp, grids,
        # #     name='apply_diff', dtype=(tf.float32,tf.float32,tf.float32))
        # # print(out_broken)

        # xlist = []
        # ylist = []
        # for i in range(48):
        #     reuse=True if i!=0 else False
        #     x, y, _ = diffnet.net_w_inp(grids[i], scope='d1', reuse=reuse)
        #     xlist.append(x)
        #     ylist.append(y)

        # x_d1 = tf.concat(xlist, axis=0)
        # y_d1 = tf.concat(ylist, axis=0)


        # # x_d1 = tf.reshape(out_broken[0], (48, sx, sy))
        # # y_d1 = tf.reshape(out_broken[1], (48, sx, sy))

        # print(x_d1.get_shape().as_list())
        # print(y_d1.get_shape().as_list())

        x_d1, y_d1, _, _ = diffnet.net_w_inp(grids, 'd1')

        return (x_d1, y_d1)

    def artifact_removal(x):
        from UNet import UNet

        out = UNet(config={'nblocks': 5,
                           'start_ch': 16,
                           'kernel_size': 3}).net(x, 'artifact')

        out = out + x

        return out

    BS = 8

    inp, req = read_tfrecords(
        '/mnt/ext6TB/fio/data/explosion_rtm_waves_bg_0p8_sens_trace/', batch_size=BS)

    # """Set up generalization data"""
    shapes_inp_path = '/home/konik/fiodata/explosion_rtm_random_shapes_0p4_sens_trace.npy'
    shapes_out_path = '/home/konik/fiodata/explosion_rtm_random_shapes_0p4_source_p.npy'

    shapes_in, shapes_out = prep_test_data(shapes_inp_path, shapes_out_path)

    refl_inp_path = '/home/konik/fiodata/explosion_rtm_reflectors_0p4_sens_trace.npy'
    refl_out_path = '/home/konik/fiodata/explosion_rtm_reflectors_0p4_source_p.npy'

    refl_in, refl_out = prep_test_data(refl_inp_path, refl_out_path)

    mnist_inp_path = '/home/konik/fiodata/explosion_rtm_mnist_gen_0p4_sens_trace.npy'
    mnist_out_path = '/home/konik/fiodata/explosion_rtm_mnist_gen_0p4_source_p.npy'

    mnist_in, mnist_out = prep_test_data(mnist_inp_path, mnist_out_path)

    """Set up directories"""
    logdir = 'SingleBigDiffeo_rtm_unet_waves_bg_0p8/'
    griddir = logdir+'grids/'
    gendir_shapes = logdir+'gen_shapes/'
    gendir_refl = logdir+'gen_refl/'
    gendir_mnist = logdir+'gen_mnist/'

    safemkdir(logdir)
    safemkdir(griddir)
    safemkdir(gendir_shapes)
    safemkdir(gendir_refl)
    safemkdir(gendir_mnist)

    """Set up flows"""
    box_npy = np.load('box_vecs_unit.npy')
    print('Loaded box vectors')
    diffnet = SingleBigDiffeo((128, 128), box_npy, nhidden=32, nlayers=3)

    d1_out = get_diff_grids(diffnet, box_npy, size=(128, 128))

    x_s_d1, y_s_d1 = d1_out

    inp_sum = tf.reduce_sum(inp, keepdims=True, axis=-1)

    # inp_d1, inp_d2 = conv_part(inp_sum)
    inp_d1 = conv_part(inp)
    #inp_d1 = inp

    warp1 = diffnet.apply(inp_d1[:, :, :, 1:49],
                          x_s_d1, y_s_d1, 'warp_d1')

    print(warp1.get_shape().as_list())

    net_out = warp1    # loss_artifact = tf.reduce_sum(
    #     tf.square(out_arti-req_full_sum))/tf.cast(bs, tf.float32)

    net_out_sum = tf.reduce_sum(net_out, axis=-1)
    req_sum = tf.reduce_sum(req[:, :, :, 1:49], axis=-1)
    req_full_sum = tf.reduce_sum(req, axis=-1, keepdims=True)

    """remove artifacts"""
    out_arti = artifact_removal(tf.expand_dims(net_out_sum, axis=-1))

    # print(sum([np.prod(v.get_shape().as_list())
    #            for v in tf.trainable_variables()]))

    """Losses and training"""
    bs = tf.shape(req)[0]
    loss = tf.reduce_sum(tf.square(net_out_sum-req_sum)) / \
        tf.cast(bs, tf.float32)
    # loss_artifact = tf.reduce_sum(
    #     tf.square(out_arti-req_full_sum))/tf.cast(bs, tf.float32)

    d1_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='lr_d1')
    print('\ntrainable_variables (low rank)\n')
    print(d1_vars)

    diffd1_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='d1')
    print('\ntrainable_variables (diffeo)\n')
    print(diffd1_vars)

    # artifact_vars = tf.get_collection(
    #     tf.GraphKeys.TRAINABLE_VARIABLES, scope='artifact')
    # print('\ntrainable_variables (UNet)\n')
    # print(artifact_vars)

    train_step_lr = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(
            loss, var_list=d1_vars)
    train_step_diff = tf.train.AdamOptimizer(learning_rate=1e-6).minimize(
            loss, var_list=diffd1_vars)
    # train_step_artifact = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(
    #     loss_artifact, var_list=artifact_vars)

    saver_diff = tf.train.Saver(var_list=diffd1_vars)
    saver_net = tf.train.Saver(var_list=diffd1_vars+d1_vars)

    do_diff_load = True

    options = tf.RunOptions(
        trace_level=tf.RunOptions.FULL_TRACE,
        report_tensor_allocations_upon_oom=True)
    run_metadata = tf.RunMetadata()

    def test_data_save(sess, gvars, gendir, data_in, data_out):    
        xi, xi_lr1, o, w1, oa, r = sess.run(
            gvars,
            feed_dict={inp: data_in, req: data_out})
        
        np.save(gendir+'inp.npy', xi)
        np.save(gendir+'inp_lr1.npy', xi_lr1)
        np.save(gendir+'out.npy', o)
        np.save(gendir+'req.npy', r)
        np.save(gendir+'o1.npy', w1)
        np.save(gendir+'oa.npy', oa)

        # save the grids
        xsd1, ysd1 = sess.run(
            [x_s_d1, y_s_d1])
        np.save(griddir + 'd1.npy', np.stack((xsd1, ysd1), axis=0))

        return None

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())  # , options=run_options)

        # preload various parts of the network
        ckpt = tf.train.latest_checkpoint(logdir)
        
        if ckpt:
            print('Checkpoint is')
            print(ckpt)
            
            print(tf.train.list_variables(ckpt))
            saver_net.restore(sess, ckpt)
            print('Loaded network successfully!')
            do_diff_load = False

        ckpt = tf.train.latest_checkpoint('./hj_flow_xt/')

        if ckpt and do_diff_load:
            diffnet.load(sess, ckpt, saver_diff)
            print('Preload successful')

        print('starting training')

        t = time()
        for i in range(50000):

            sess.run([train_step_lr, train_step_diff])
            # sess.run([train_step_lr])
            # sess.run([train_step_diff])
            # sess.run(train_step_artifact)
            if i % 100 == 0:
                print('That took %fs' % (time()-t))
                l = sess.run(loss)
                print('Loss at %d = %f' % (i, l))
                t = time()

            if i % 1000 == 0:
                xi, xi_lr1, o, w1, oa, r = sess.run(
                    [inp, inp_d1, net_out, warp1, out_arti, req])
                np.save(logdir+'inp.npy', xi)
                np.save(logdir+'inp_lr1.npy', xi_lr1)
                np.save(logdir+'out.npy', o)
                np.save(logdir+'req.npy', r)
                np.save(logdir+'o1.npy', w1)
                np.save(logdir+'oa.npy', oa)

            if i % 1000 == 0:
                # gvars = [inp, inp_d1, net_out, warp1, out_arti, req]
                # test_data_save(sess, gvars, gendir_shapes, shapes_in, shapes_out)
                # test_data_save(sess, gvars, gendir_mnist, mnist_in, mnist_out)
                # test_data_save(sess, gvars, gendir_refl, refl_in, refl_out)

                # save the model
                saver_net.save(sess, logdir+'model', global_step=i)

    return


if __name__ == '__main__':
    from_seis_trace()
