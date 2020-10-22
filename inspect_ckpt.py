import numpy as np
import tensorflow as tf

sess = tf.Session()

logdir = 'SingleBigDiffeo_fourier_thicklines_0p2_bs8/'
saver = tf.train.import_meta_graph(logdir+'model-94000.meta')
ckpt = tf.train.latest_checkpoint(logdir)
print(ckpt)
print(tf.train.list_variables(ckpt))
saver.restore(sess, ckpt)

var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='lr')

for v in var:
	print(v.name)

w1, w2 = sess.run(var)
print(w2.dtype)
np.save(logdir + 'lr_d1_w1.npy', w1)
np.save(logdir + 'lr_d2_w1.npy', w2)

# w1, _, w2, _ = sess.run(var[:4])
# np.save(logdir+'lr_d1_w1.npy',w1)
# np.save(logdir+'lr_d1_w2.npy',w2)

# w1, _, w2, _ = sess.run(var[4:])
# np.save(logdir+'lr_d2_w1.npy',w1)
# np.save(logdir+'lr_d2_w2.npy',w2)	