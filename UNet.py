import tensorflow as tf
from tensorflow.contrib.layers import conv2d, max_pool2d, avg_pool2d


class UNet(object):
    """ This is a UNet class that sets up a regular UNet
    with different number of channels, pooling options
    and output options

    """
    config_def = {
        'pool_type': 'max',
        'start_ch': 4,
        'activation_fn': tf.nn.leaky_relu,
        'kernel_size': 3,
        'nblocks': 3,
        'add_skip': True,
        'out_ch': None,
        'name': 'UNet',
        'multiply': False,
    }

    def __init__(self, config=config_def):
        """
        Parameters:

            config: dict with following parameters

        """
        # update dictionary with defaults if key
        # not passed
        for key in self.config_def:
            if key not in config:
                config[key] = self.config_def[key]

        self.config = config

    def __call__(self, x):
        return self.net(x)

    def midactivations_group_l1_l2_norm(self):
        """Calculates the group l2_l1 sparsity norm for
        middle layer activations"""
        if not hasattr(self, 'mid_outputs'):
            raise KeyError("mid level output")

        shape = self.mid_outputs.get_shape().as_list()
        print(shape)

        # self.mid_outputs_vect should be
        # batch_size x img_size**2 x nchannels
        self.mid_outputs_vect = tf.reshape(self.mid_outputs,
                                           (-1, shape[1] * shape[2], shape[3]))

        # self.mid_norm_l2 should be batch size x num channels
        self.mid_norm_l2 = tf.norm(self.mid_outputs_vect,
                                   ord=2, axis=1)

        # self.mid_norm_l2_l1 should be batch size
        self.mid_norm_l2_l1 = tf.norm(self.mid_norm_l2, ord=1, axis=1)

        return self.mid_norm_l2_l1

    def net(self, x, name):

        """Build a Unet network"""
        # name = self.config['name']
        nblocks = self.config['nblocks']
        multiply = self.config['multiply']
        nchannels = self.config['start_ch']
        kernel_size = self.config['kernel_size']
        activation_fn = self.config['activation_fn']
        pool_fn = max_pool2d if self.config['pool_type'] == 'max' \
            else avg_pool2d

        _, _, _, input_ch = x.get_shape().as_list()

        if self.config['add_skip']:
            skip = []

        # == downsampling layers
        
        block_in = x
        for i in range(nblocks):
            params = {'kernel_size': kernel_size,
                      'num_outputs': 2**i * nchannels,
                      'activation_fn': activation_fn}

            with tf.variable_scope(name + '/downsampling_%d' % i):
                x1 = conv2d(block_in, **params)
                x2 = conv2d(x1, **params)
                mp = pool_fn(x2, kernel_size=[2, 2], stride=2)

                if self.config['add_skip']:
                    skip.append(x2)

                block_in = mp

        self.mid_outputs = mp

        block_in = conv2d(self.mid_outputs,
                          kernel_size=3,
                          num_outputs=2**nblocks,
                          activation_fn=activation_fn)

        if multiply:
            alpha = tf.Variable(tf.random.uniform(block_in.get_shape().as_list()[1:],
                                dtype=tf.float32) + 0.5, trainable=True)
            block_in = block_in * alpha

        # == upsampling layers

        upsampler = tf.keras.layers.UpSampling2D((2, 2))

        for i in range(nblocks):
            params = {'kernel_size': kernel_size,
                      'num_outputs':
                      2**(nblocks - i - 1) * nchannels,
                      'activation_fn': activation_fn}

            with tf.variable_scope(name + '/upsampling_%d' % i):
                x1 = upsampler(block_in)

                if self.config['add_skip'] and i != 0:
                    x_concat = tf.concat([x1, skip[-i - 1]], axis=3)
                    x2 = conv2d(x_concat, **params)
                else:
                    x2 = conv2d(x1, **params)

                x3 = conv2d(x2, **params)

                block_in = x3

        # == output layer
        if self.config['out_ch'] is None:
            out_ch = input_ch
        else:
            out_ch = self.config['out_ch']

        params = {'kernel_size': kernel_size,
                  'num_outputs': out_ch,
                  'activation_fn': tf.identity}

        self.output = conv2d(block_in, **params)
        
        return self.output


if __name__ == '__main__':
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    tf.logging.set_verbosity(tf.logging.ERROR)

    data_train, data_test = tf.keras.datasets.mnist.load_data()
    model = UNet(config={'start_ch': 16, 'nblocks': 5})

    x = tf.placeholder(tf.float32, shape=[None, 128, 128, 1], name='input')

    y = model(x)
    norm = model.midactivations_group_l1_l2_norm()
    print(y.get_shape().as_list())

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        train_writer = tf.summary.FileWriter('./')
        train_writer.add_graph(sess.graph)
