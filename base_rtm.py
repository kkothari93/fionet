import os
import numpy as np
import tensorflow as tf
from time import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from tensorflow.contrib.layers import fully_connected, conv2d

from utils import *
from UNet import UNet

from SingleBigDiffeo import SingleBigDiffeo

from tfrecords import read_tfrecords

tf.random.set_random_seed(1)
np.random.seed(0)

def isp():

    def baseline(x):
        out = UNet(config={'nblocks': 6,
            'start_ch': 32,
            'out_ch': 1,
            'kernel_size': 3}).net(x, 'baseline')

        return out


    BS = 8
    Ndata = 3000

    DATA_DIR = '/home/konik/fiodata/full_rtm_128/'

    mean_inp = tf.convert_to_tensor(np.load('../data/mean_full_rtm_extend.npy'), tf.float32)

    inp, req = read_tfrecords(
        DATA_DIR+'explosion_full_rtm_lines_0p2_sens_trace/', batch_size=BS, img_size=128)

    inp_wo_direct = inp - mean_inp

    # """Set up generalization data"""
    shapes_inp_path = DATA_DIR+'explosion_full_rtm_shapes_0p2_sens_trace_cl.npy'
    shapes_out_path = DATA_DIR+'explosion_full_rtm_shapes_0p2_source_p_cl.npy'
    shapes_in, shapes_out = np.load(shapes_inp_path)[:BS], np.load(shapes_out_path)[:BS]

    geo_inp_path = DATA_DIR+'explosion_full_rtm_geo_0p2_sens_trace_cl.npy'
    geo_out_path = DATA_DIR+'explosion_full_rtm_geo_0p2_source_p_cl.npy'
    geo_in, geo_out = np.load(geo_inp_path)[:BS], np.load(geo_out_path)[:BS]

    waves_inp_path = DATA_DIR+'explosion_full_rtm_waves_0p2_sens_trace_cl.npy'
    waves_out_path = DATA_DIR+'explosion_full_rtm_waves_0p2_source_p_cl.npy'
    waves_in, waves_out = np.load(waves_inp_path)[:BS], np.load(waves_out_path)[:BS]

    refl_inp_path = DATA_DIR+'explosion_full_rtm_expl_0p2_sens_trace_cl.npy'
    refl_out_path = DATA_DIR+'explosion_full_rtm_expl_0p2_source_p_cl.npy'
    refl_in, refl_out = np.load(refl_inp_path)[:BS], np.load(refl_out_path)[:BS]

    kate_inp_path = DATA_DIR+'explosion_full_rtm_rot_refl_0p2_sens_trace_cl.npy'
    kate_out_path = DATA_DIR+'explosion_full_rtm_rot_refl_0p2_source_p_cl.npy'
    kate_in, kate_out = np.load(kate_inp_path)[:BS], np.load(kate_out_path)[:BS]

    lines_inp_path = DATA_DIR+'explosion_full_rtm_lines_0p2_sens_trace_cl.npy'
    lines_out_path = DATA_DIR+'explosion_full_rtm_lines_0p2_source_p_cl.npy'
    lines_in, lines_out = np.load(lines_inp_path)[:BS], np.load(lines_out_path)[:BS]

    """Set up directories"""
    logdir = 'baseline_6x32_full_rtm_extend/'
    griddir = logdir+'grids/'
    gendir_shapes = logdir+'gen_shapes/'
    gendir_refl = logdir+'gen_refl/'
    gendir_lines = logdir+'gen_lines/'
    gendir_kate = logdir+'gen_kate/'
    gendir_waves = logdir+'gen_waves/'
    gendir_geo = logdir+'gen_geo/'

    safemkdir(logdir)
    safemkdir(griddir)
    safemkdir(gendir_shapes)
    safemkdir(gendir_waves)
    safemkdir(gendir_lines)
    safemkdir(gendir_refl)
    safemkdir(gendir_kate)
    safemkdir(gendir_geo)

    inp_sum = tf.reduce_sum(inp_wo_direct, keepdims=True, axis=-1)

    net_out = baseline(inp_sum)
    # inp_d1, inp_d2 = inp, inp

    net_out_sum = tf.reduce_sum(net_out, axis=-1, keepdims=True)
    req_hp = tf.reduce_sum(req[:, :, :, 1:], axis=-1, keepdims=True)
    req_full = tf.reduce_sum(req, axis=-1, keepdims=True)

    print(sum([np.prod(v.get_shape().as_list())
               for v in tf.trainable_variables()]))

    """Losses and training"""
    bs = tf.shape(req)[0]

    loss = tf.reduce_sum(tf.square(net_out-req_hp)) / \
        tf.cast(bs, tf.float32)

    step = tf.train.get_or_create_global_step()

    train_step = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(
        loss, global_step=step)
    
    saver = tf.train.Saver()

    def test_data_save(sess, gvars, gendir, data_in, data_out):    
        xi, o, r = sess.run(
            gvars,
            feed_dict = {inp: data_in, req: data_out})
        
        np.save(gendir+'inp.npy', xi)
        np.save(gendir+'out.npy', o)
        np.save(gendir+'req.npy', r)

        return None

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # np.random.shuffle(idx)
    with tf.Session(config = config) as sess:
        sess.run(tf.global_variables_initializer())  # , options=run_options)

        ckpt = tf.train.latest_checkpoint(logdir)
        if ckpt:
            saver.restore(sess, ckpt)
            print('Loaded network successfully!')
        print('starting training')

        t = time()
        niterations = 100*(Ndata//BS)
        print("Will run for %d iterations"%niterations)
        for i in range(niterations):

            # if i%10000==0:
            #     inp_npy = inp_npy[idx]
            #     req_npy = req_npy[idx]

            # k = i%Ndata//BS
            # s = k*BS
            # e = (k+1)*BS

            ops = [step, train_step]

            # feed_dict = {inp: inp_npy[s:e], req: req_npy[s:e]}
            # vals = sess.run(ops, feed_dict=feed_dict)
            vals = sess.run(ops)
            gs = vals[0]

            if i % 100 == 0:
                print('That took %fs' % (time()-t))
                # l = sess.run(loss, feed_dict=feed_dict)
                l = sess.run(loss)
                print('Loss at %d = %f' % (i, l))
                t = time()

            if i % 1000 == 0:
                xi, o, r = sess.run(
                    [inp, net_out, req])
                # xi, o, r = sess.run(
                #     [inp, net_out, req], feed_dict=feed_dict)
                np.save(logdir+'inp.npy', xi)
                np.save(logdir+'out.npy', o)
                np.save(logdir+'req.npy', r)

            if i % 1000 == 0:
                gvars = [inp, net_out, req]

                test_data_save(sess, gvars, gendir_shapes, shapes_in, shapes_out)
                test_data_save(sess, gvars, gendir_waves, waves_in, waves_out)
                test_data_save(sess, gvars, gendir_refl, refl_in, refl_out)
                test_data_save(sess, gvars, gendir_lines, lines_in, lines_out)
                test_data_save(sess, gvars, gendir_kate, kate_in, kate_out)
                test_data_save(sess, gvars, gendir_geo, geo_in, geo_out)

                # save the model
                saver.save(sess, logdir+'model', global_step=gs)
                print('Model saved')

    return


if __name__ == '__main__':
    isp()
