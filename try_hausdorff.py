import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import apply_affine_transform
from utils import *
from hausdorff import Weighted_Hausdorff_loss

from SingleBigDiffeo import SingleBigDiffeo

from tfrecords import read_tfrecords
import matplotlib.pyplot as plt


BS = 8

box_npy = np.load('box_vecs_unit.npy')
diffnet = SingleBigDiffeo((128, 128), box_npy, nhidden=32, nlayers=3)
grid = diffnet.create_cartesian_grid((128,128))
grids = []
npts = tf.shape(grid)[0]
bv = tf.convert_to_tensor(
    np.array([[0.6, 0.8]]), dtype=tf.float32)
bgrid = tf.concat((grid, tf.tile(bv, (npts, 1))), axis=-1)
x_d1, y_d1, _, _ = diffnet.net_w_inp(bgrid, scope='out', return_sing_loss=False)


inp, req = read_tfrecords(
    '/mnt/ext6TB/fio/data/explosion_thick_lines_0p2/', batch_size=BS)

inp = tf.reduce_sum(req, axis=-1, keepdims=True)
inp = tf.cast(inp > 0.01, tf.float32)
print(inp.get_shape().as_list())

req = inp[:,:,:,:]

out = diffnet.apply(inp, x_d1, y_d1, 'try_hd')

loss = Weighted_Hausdorff_loss(req, out, batch_size=BS)

train = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)

print(tf.trainable_variables())


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config = config) as sess:
	sess.run(tf.global_variables_initializer())

	for i in range(1000):
		_, l = sess.run([train, loss])
		print(l)

		if i%100==0:
			g_e, g_r = sess.run([out, req])
			print(g_e.shape)

			# fig, ax = plt.subplots(1,2)
			# ax[0].imshow(g_r[0,:,:,0])
			# ax[1].imshow(g_e[0,:,:,0])
			# plt.show()

			np.save('g_e.npy', g_e)
			np.save('g_r.npy', g_r)

