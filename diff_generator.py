import tensorflow as tf
import numpy as np

from tensorflow.contrib.layers import conv2d, fully_connected, conv2d_transpose, layer_norm


def motion_field(bv):
	with tf.variable_scope('flow_field', reuse=tf.AUTO_REUSE):
		upsampler = tf.keras.layers.UpSampling2D(size=(2,2), interpolation='bilinear')
		c = fully_connected(bv, 16, activation_fn = tf.nn.relu)
		c = fully_connected(c, 16*16, activation_fn = tf.nn.relu)

		c = tf.reshape(c, (bs, 4, 4, 16))

		h = upsampler(h)
		h = conv2d(h, 256, kernel_size=3)
		h = conv2d(h, 256, kernel_size=3)
		# [8x8]

		h = upsampler(h)
		h = conv2d(h, 128, kernel_size=3)
		h = conv2d(h, 128, kernel_size=3)
		# [16x16]

		h = upsampler(h)
		h = conv2d(h, 64, kernel_size=3)
		h = conv2d(h, 64, kernel_size=3)
		# [32x32]
		
		h = upsampler(h)
		h = conv2d(h, 32, kernel_size=3)
		h = conv2d(h, 32, kernel_size=3)
		h64 = h
		h64 = h64[:,16:48,16:48,:]
		# [64x64]

		h = upsampler(h)
		h = conv2d(h, 16, kernel_size=3, activation_fn = tf.leaky_relu)
		h = conv2d(h, 16, kernel_size=3, activation_fn = tf.leaky_relu)
		# [128x128]

		h = upsampler(h)
		h = conv2d(h, 8, kernel_size=3, activation_fn = tf.leaky_relu)
		h = conv2d(h, 8, kernel_size=3, activation_fn = tf.leaky_relu)
		# [256x256]

		h = h[:,64:340, 64:340, :]

		flows = conv2d(h64, 8, kernel_size=1, activation_fn = tf.identity)

	return flows

def network(x, bv):

	bs = tf.shape(x)[0]

	normalizer_fn = layer_norm
	with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
		h = conv2d(x, 16, 3, normalizer_fn=layer_norm)
		h = tf.nn.max_pool2d(h, 2, 2, padding='SAME')
		# [64x64]

		h = conv2d(h, 32, 3, normalizer_fn=layer_norm)
		h = tf.nn.max_pool2d(h, 2, 2, padding='SAME')
		# [32x32]

		h = conv2d(h, 64, 3, normalizer_fn=layer_norm)
		h = tf.nn.max_pool2d(h, 2, 2, padding='SAME')
		# [16x16]

		h = conv2d(h, 128, 3, normalizer_fn=layer_norm)
		h = tf.nn.max_pool2d(h, 2, 2, padding='SAME')
		# [8x8]

		h = conv2d(h, 256, 3, normalizer_fn=layer_norm)
		h = tf.nn.max_pool2d(h, 2, 2, padding='SAME')
		# [4x4]

	with tf.variable_scope('flow_field', reuse=tf.AUTO_REUSE):
		upsampler = tf.keras.layers.UpSampling2D(size=(2,2), 
			interpolation='bilinear')
		c = fully_connected(bv, 16, activation_fn = tf.nn.relu)
		c = fully_connected(c, 16*16, activation_fn = tf.nn.relu)

		c = tf.reshape(c, (bs, 4, 4, 16))

		h = tf.concat((h,c), axis=-1)

		h = upsampler(h)
		h = conv2d(h, 128, kernel_size=3,activation_fn = tf.nn.leaky_relu)
		h = conv2d(h, 128, kernel_size=3,activation_fn = tf.nn.leaky_relu)
		# [8x8]

		h = upsampler(h)
		h = conv2d(h, 64, kernel_size=3,activation_fn = tf.nn.leaky_relu)
		h = conv2d(h, 64, kernel_size=3,activation_fn = tf.nn.leaky_relu)
		# [16x16]

		h = upsampler(h)
		h = conv2d(h, 32, kernel_size=3,activation_fn = tf.nn.leaky_relu)
		h = conv2d(h, 32, kernel_size=3,activation_fn = tf.nn.leaky_relu)
		# [32x32]

		h = upsampler(h)
		h = conv2d(h, 16, kernel_size=3,activation_fn = tf.nn.leaky_relu)
		h = conv2d(h, 16, kernel_size=3,activation_fn = tf.nn.leaky_relu)
		# [64x64]

		h = h[:,16:48,16:48,:]

		flows = conv2d(h, 8, kernel_size=1, activation_fn = tf.identity)

	return flows

c_vals = tf.placeholder(tf.float32, [None, 128,128,1])
flow_fields = tf.placeholder(tf.float32, [None, 32,32,8])
directions = tf.placeholder(tf.float32, [None, 2])


cdata = tf.data.Dataset.from_tensor_slices(c_vals)
flows = tf.data.Dataset.from_tensor_slices(flow_fields)
dirs = tf.data.Dataset.from_tensor_slices(directions)

zip_data = tf.data.Dataset.zip((cdata, flows, dirs))
zip_data = zip_data.prefetch(100)
zip_data = zip_data.repeat()
zip_data = zip_data.shuffle(64*5)
zip_data = zip_data.batch(64)

iterator = zip_data.make_initializable_iterator()
c, ff, xi = iterator.get_next()

yhat = network(c, xi)
d1_norm = tf.reduce_sum(tf.image.total_variation(yhat[:,:,:,:2]))
d2_norm = tf.reduce_sum(tf.image.total_variation(yhat[:,:,:,4:6]))
loss = (tf.reduce_sum(tf.abs(yhat - ff))+0.1*(d1_norm + d2_norm))/64.0

step = tf.train.get_or_create_global_step()
lr = tf.cast(step<10000, tf.float32)*1e-3 +\
 tf.cast(step>=10000, tf.float32)*1e-4
train_step = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	sess.run(iterator.initializer, 
		feed_dict={
		c_vals: np.load('bg_gl_0p2_div.npy').astype(np.float32),
		directions: np.load('dir_vecs_0p2_div.npy').astype(np.float32),
		flow_fields: np.load('flow_fields_0p2_div.npy').astype(np.float32),
		})

	for i in range(40000):
		_, l = sess.run([train_step, loss])

		if i%100 == 0:
			print('[%d] Loss is : %f'%(i, l))

		if i%1000 == 0:
			grids, req, c_v = sess.run([yhat, ff, c])
			np.save('yhat_d1.npy', grids[:,:,:,:4])
			np.save('req_d1.npy', req[:,:,:,:4])
			np.save('yhat_d2.npy', grids[:,:,:,4:])
			np.save('req_d2.npy', req[:,:,:,4:])
			np.save('bg.npy', c_v[:,:,:,0])