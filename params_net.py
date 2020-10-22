import tensorflow as tf
import tensorflow.contrib.layers as layers


def bgnet(x, reuse=False):
    """Get background parameters from the pressure snaps
      Network to classify fake and true samples.
      params:
        x: Input images [batch size, 64, 64, 3]
      returns:
        h: Features from penultimate layer of discriminator 
          [batch size, 7]
    """
    batch_norm = layers.layer_norm

    h = x
    with tf.variable_scope("discriminator", reuse=reuse) as scope:
        h = layers.conv2d(
            inputs=h,
            num_outputs=32,
            kernel_size=4,
            stride=2,
            activation_fn=tf.nn.leaky_relu,
            normalizer_fn=batch_norm)
        # [32,32,64]

        h = layers.conv2d(
            inputs=h,
            num_outputs=64,
            kernel_size=4,
            stride=2,
            activation_fn=tf.nn.leaky_relu,
            normalizer_fn=batch_norm)
        # [16,16,128]

        h = layers.conv2d(
            inputs=h,
            num_outputs=128,
            kernel_size=4,
            stride=2,
            activation_fn=tf.nn.leaky_relu,
            normalizer_fn=batch_norm)
        # [8,8,256]

        h = layers.conv2d(
            inputs=h,
            num_outputs=256,
            kernel_size=4,
            stride=2,
            activation_fn=tf.nn.leaky_relu,
            normalizer_fn=batch_norm)
        
        h = layers.conv2d(
            inputs=h,
            num_outputs=512,
            kernel_size=4,
            stride=2,
            activation_fn=tf.nn.leaky_relu,
            normalizer_fn=batch_norm)

        h = layers.flatten(h)

        h = layers.fully_connected(h, 7, activation_fn = identity)
    
    return h

tf.random.set_seed(0)
tf.reset_default_graph()

BS = 64

gs = tf.train.get_or_create_global_step()

x = tf.placeholder(tf.float32, [None, 128, 128, 1])
y = tf.placeholder(tf.float32, [None, 7])

yhat = bgnet(x)

bs = tf.shape(x)[0]
loss = tf.reduce_sum(tf.square(y-yhat)**2)/tf.cast(bs, tf.float32)

train_step = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(
    loss, global_step = gs)


inputs = np.load('/home/konik/fiodata/explosion_thick_lines_rand_bg_0p2_final_p.npy').astype(np.float32)
outputs = np.load('/home/konik/fiodata/explosion_thick_lines_rand_bg_0p2_params.npy').astype(np.float32)

Ndata = len(inputs)-100
idx = np.arange(Ndata)

inputs_train = inputs_[:Ndata]
inputs_test = inputs_[Ndata:]

outputs_train = outputs_[:Ndata]
outputs_test = outputs_[Ndata:]

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:

    sess.run(tf.global_variables_initializer())

    for i in range(20):
        np.random.shuffle(idx)

        inputs_train = inputs_train[idx]
        outputs_train = outputs_train[idx]

        for j in range(Ndata//BS):
            s = j*BS
            e = (j+1)*BS
            in_ = inputs_train[s:e]
            out_ = outputs_train[s:e]

            feed_dict = {x: in_, y: out_}

            l_train, _ = sess.run([loss, train_step], feed_dict=feed_dict)

        feed_dict = {x: inputs_test[:BS], y: outputs_test[:BS]}
        l_test = sess.run(loss, feed_dict=feed_dict)

        print('Epoch #%d: training loss %0.6f, test loss %0.6f'%(i+1, l_train, l_test))
