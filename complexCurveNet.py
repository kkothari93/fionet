import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tcl

class complexCurveNet():
    def __init__(self,  tile_npy, trainable=True):
        _, ch, img_size, _ = tile_npy.shape
        with tf.variable_scope('curvenet', reuse=tf.AUTO_REUSE):
            self.mask = tf.convert_to_tensor(tile_npy, 
                dtype=tf.float32, name='mask')

            self.conv_weights = tf.Variable(
                tf.random.normal(shape=(1,ch,img_size,img_size))*self.mask,
                trainable=trainable,
                dtype=tf.float32,
                name='conv_weights')

            self.conv_weights_cmplx = tf.cast(self.conv_weights, tf.complex64)

    def apply(self, x_nhwc):
        with tf.variable_scope('curvenet', reuse=tf.AUTO_REUSE):
            x_nchw = tf.transpose(x_nhwc, [0,3,1,2])
            x_nchw = tf.complex(x_nchw, x_nchw*0)

            x_fft = tf.signal.fft2d(x_nchw)
            x_fft = x_fft*self.conv_weights_cmplx

            x = tf.math.real(tf.signal.ifft2d(x_fft))

            x = tf.transpose(x, [0,2,3,1])

        return x_fft, x

class complexCurveNetOS():
    def __init__(self,  tile_npy, name='', trainable=True):
        self.name = name
        with tf.variable_scope(self.name + '_curvenet' , reuse=tf.AUTO_REUSE):
            self.mask = tf.convert_to_tensor(tile_npy, 
                dtype=tf.float32, name='mask')

            self.conv_weights = tf.Variable(
                tf.random.normal(shape=(1,50,256,256))*self.mask,
                trainable=trainable,
                dtype=tf.float32,
                name='conv_weights')

            self.conv_weights_cmplx = tf.cast(self.conv_weights, tf.complex64)

    def apply(self, x_nhwc):
        with tf.variable_scope(self.name + '_curvenet', reuse=tf.AUTO_REUSE):
            x_nhwc = tf.pad(x_nhwc, [[0,0],[0,128],[0,128],[0,0]])
            x_nchw = tf.transpose(x_nhwc, [0,3,1,2])
            x_nchw = tf.complex(x_nchw, x_nchw*0)

            x_fft = tf.signal.fft2d(x_nchw)
            x_fft = x_fft*self.conv_weights_cmplx
            x = tf.real(tf.signal.ifft2d(x_fft))
            x = x[:,:,:128,:128]
            x = tf.reshape(x, (-1, 50, 128, 128))
            x = tf.transpose(x, [0,2,3,1])

            x_fft = x_fft[:,:,::2, ::2]
            x_fft = tf.reshape(x_fft, (-1, 50, 128, 128))

        return x_fft, x
