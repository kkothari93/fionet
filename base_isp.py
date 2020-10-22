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

from jspaceDataGenerator import get_iterator, dataGenerator
from tfrecords import read_tfrecords

tf.random.set_random_seed(1)
np.random.seed(0)


NOTRAIN = True

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

def isp():

    def baseline(x):
        out = UNet(config={'nblocks': 5,
            'start_ch': 32,
            'out_ch': 1,
            'kernel_size': 3}).net(x, 'baseline')

        return out


    BS = 8

    inp, req = read_tfrecords(
        '/mnt/ext6TB/fio/data/explosion_thick_lines_0p2/', batch_size=BS)

    # inp_npy = np.load('/home/konik/fiodata/explosion_thick_lines_30k_0p2_final_p.npy').astype(np.float32)
    # req_npy = np.load('/home/konik/fiodata/explosion_thick_lines_30k_0p2_source_p.npy').astype(np.float32)
    Ndata = 3000

    # """Set up generalization data"""
    shapes_inp_path = '/home/konik/fiodata/explosion_random_shapes_0p2_final_p_10dB_cl.npy'
    shapes_out_path = '/home/konik/fiodata/explosion_random_shapes_0p2_source_p_cl.npy'
    shapes_in, shapes_out = np.load(shapes_inp_path)[:BS], np.load(shapes_out_path)[:BS]


    mnist_inp_path = '/home/konik/fiodata/explosion_mnist_gen_0p2_final_p_10dB_cl.npy'
    mnist_out_path = '/home/konik/fiodata/explosion_mnist_gen_0p2_source_p_cl.npy'
    mnist_in, mnist_out = np.load(mnist_inp_path)[:BS], np.load(mnist_out_path)[:BS]

    lines_inp_path = '/home/konik/fiodata/explosion_thick_lines_gen_0p2_final_p_10dB_cl.npy'
    lines_out_path = '/home/konik/fiodata/explosion_thick_lines_gen_0p2_source_p_cl.npy'
    lines_in, lines_out = np.load(lines_inp_path)[:BS], np.load(lines_out_path)[:BS]

    refl_inp_path = '/home/konik/fiodata/explosion_reflectors_2_0p2_final_p_10dB_cl.npy'
    refl_out_path = '/home/konik/fiodata/explosion_reflectors_2_0p2_source_p_cl.npy'
    refl_in, refl_out = np.load(refl_inp_path)[:BS], np.load(refl_out_path)[:BS]

    faces_inp_path = '/home/konik/fiodata/explosion_face_refl_0p2_final_p_10dB_cl.npy'
    faces_out_path = '/home/konik/fiodata/explosion_face_refl_0p2_source_p_cl.npy'
    faces_in, faces_out = np.load(faces_inp_path)[
        :BS], np.load(faces_out_path)[:BS]

    """Set up directories"""
    logdir = 'baseline_5x32_isp_3k_hp/'
    griddir = logdir+'grids/'
    gendir_shapes = logdir+'gen_shapes_noise/'
    gendir_lines = logdir+'gen_lines_noise/'
    gendir_mnist = logdir+'gen_mnist_noise/'
    gendir_refl = logdir+'gen_refl_noise/'
    gendir_faces = logdir+'gen_faces_noise/'

    safemkdir(logdir)
    safemkdir(griddir)
    safemkdir(gendir_shapes)
    safemkdir(gendir_mnist)
    safemkdir(gendir_lines)
    safemkdir(gendir_refl)
    safemkdir(gendir_faces)

    inp_sum = tf.reduce_sum(inp, keepdims=True, axis=-1)

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

        if NOTRAIN:
            gvars = [inp, net_out, req]

            test_data_save(sess, gvars, gendir_shapes, shapes_in, shapes_out)
            test_data_save(sess, gvars, gendir_mnist, mnist_in, mnist_out)
            test_data_save(sess, gvars, gendir_refl, refl_in, refl_out)
            test_data_save(sess, gvars, gendir_lines, lines_in, lines_out)
            test_data_save(sess, gvars, gendir_faces, faces_in, faces_out)

            return

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
                test_data_save(sess, gvars, gendir_mnist, mnist_in, mnist_out)
                test_data_save(sess, gvars, gendir_lines, lines_in, lines_out)

                # save the model
                saver.save(sess, logdir+'model', global_step=gs)

    return


if __name__ == '__main__':
    isp()
