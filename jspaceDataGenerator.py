import os
import sys
import signal
import pyct as ct
import numpy as np
from time import time
import tensorflow as tf
import multiprocessing as mp

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

tf.logging.set_verbosity(tf.logging.ERROR)

NSCALE = 4
NANGLES = 16
NDATA = 2000
NBOXES = 50

tf.reset_default_graph()


class CL():
    def __init__(self, size, nscale, nangles):
        self.A = ct.fdct2(size, nscale, nangles, False, norm=False, cpx=False)
        # need this signal handlers to properly handle Keyboard Interrupts
        # original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
        # self.pool = mp.Pool(8)
        # signal.signal(signal.SIGINT, original_sigint_handler)

        self.shapes = self.op_shapes()
        self.split_lengths = self.__get_split_lengths()
        self.indices = np.concatenate((np.array([0]),
                                       np.cumsum(self.split_lengths)))

    @staticmethod
    def decompose(A, img, idx):
        coeffs = A.fwd(img)
        N = 50
        out = np.zeros(img.shape + (N,))
        for i in range(N):
            g = np.zeros_like(coeffs)
            g[idx[i]:idx[i+1]] = coeffs[idx[i]:idx[i+1]]
            out[:, :, i] = A.inv(g)

        return out

    @staticmethod
    # need staticmethod because passing self requires apply_async
    # to pickle everything in self which includes the Pool object
    # We don't need it anyway. So I pass both A and the img
    def analyze(A, img):
        return A.fwd(img)

    def ns_analyze(self, img):
        return self.A.fwd(img)

    def op_shapes(self):
        shapes = []
        for size in self.A.sizes:
            shapes.extend(size)
        return shapes

    def __get_split_lengths(self):
        """gets the sizes of the splits"""

        # first get the shapes
        shapes = self.shapes

        # then get the size of splits
        split_lengths = [np.prod(s) for s in shapes]

        return split_lengths

    @staticmethod
    # need staticmethod because passing self requires apply_async
    # to pickle everything in self which includes the Pool object
    # We don't need it anyway. So I pass both A and the img
    def synthesize(A, coeff):
        return A.inv(coeff)

    def ns_synthesize(self, coeff):
        return self.A.inv(coeff)

    def get_struct(self, coeff):
        return self.A.struct(coeff)

    def mp_analyze(self, imgs):
        list_calls = [self.pool.apply_async(self.analyze,
                                            args=(self.A, img)) for img in imgs]
        d = [call.get() for call in list_calls]
        d = np.stack(d, axis=0)

        return d

    def mp_decompose(self, imgs):
        # original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
        pool = mp.Pool(8)
        # signal.signal(signal.SIGINT, original_sigint_handler)
        list_calls = [pool.apply_async(self.decompose,
                                            args=(self.A,
                                                img,
                                                self.indices)) for img in imgs]
        d = [call.get() for call in list_calls]
        d = np.stack(d, axis=0)
        pool.close()
        pool.join()
        return d

    def mp_synthesize(self, coeffs):
        list_calls = [self.pool.apply_async(self.synthesize,
                                            args=(self.A, c)) for c in coeffs]
        d = [call.get() for call in list_calls]
        d = np.stack(d, axis=0)

        return d


class dataGenerator():
    def __init__(self, path, batch_size=4):
        self.input_data = np.load(path).astype(np.float32)
        self.img_size = self.input_data.shape[1:3]
        self.i = 0
        self.ntotal = len(self.input_data)
        self.cl = CL(self.img_size, 4, 16)
        self.batch_size = batch_size

        self.__shapes = self.cl.shapes
        self.__split_lengths = self.cl.split_lengths

    def split(self, coeffs):
        """gets the sizes of the splits"""

        shapes = self.__shapes

        split_coeffs = tf.split(coeffs, self.__split_lengths, axis=1)

        reshaped_j = [tf.reshape(split_coeffs[i], (-1,)+shapes[i]+(1,))
                      for i in range(len(self.__shapes))]

        return reshaped_j

    def reset(self):
        self.i = 0

    def join(self, t_list):
        """Does the exact opposite of split"""
        shapes = self.__shapes
        flat = tf.keras.layers.Flatten()
        reshaped_j = [flat(t) for t in t_list]
        coeffs = tf.concat(reshaped_j, axis=-1)

        return coeffs

    def generator_analyze_batch(self):
        # generator batch provides a 4x speedup at batch_size of 32
        nt = self.ntotal
        bs = self.batch_size
        i = self.i % (nt//bs)
        batch_inp = self.input_data[i*bs:(i+1)*bs]

        batch_inp = np.squeeze(batch_inp, axis=-1)
        batch_out = self.cl.mp_analyze(batch_inp)

        self.i += 1

        yield batch_out

    def generator_decompose_batch(self):
        # generator batch provides a 4x speedup at batch_size of 32
        nt = self.ntotal
        bs = self.batch_size
        i = self.i % (nt//bs)
        batch_inp = self.input_data[i*bs:(i+1)*bs]

        batch_inp = np.squeeze(batch_inp, axis=-1)
        batch_out = self.cl.mp_decompose(batch_inp)

        self.i += 1

        yield batch_out

    def generator(self):
        nt = self.ntotal
        i = self.i % nt
        inp = self.input_data[i, :, :, 0]
        out = self.cl.ns_analyze(inp)
        self.i += 1

        yield out

    def decompose_batch(self):
        return tf.data.Dataset.from_generator(self.generator_decompose_batch,
                                              output_types=tf.float32)

    def analyze_batch(self):
        return tf.data.Dataset.from_generator(self.generator_analyze_batch,
                                              output_types=tf.float32)


    @staticmethod
    def get_dataset_iterator(dataset, bs):
        dataset = dataset.prefetch(5*bs)
        dataset = dataset.shuffle(5*bs)
        dataset = dataset.repeat()

        return dataset.make_one_shot_iterator()

    @staticmethod
    def combine(d1, d2):
        return tf.data.Dataset.zip((d1,d2))
        
def get_iterator(inp_path, out_path, batch_size):
    d = dataGenerator(inp_path, batch_size=batch_size).decompose_batch()
    d_out = dataGenerator(out_path, batch_size=batch_size).decompose_batch()
    combined_data = dataGenerator.combine(d, d_out)

    iterator = dataGenerator.get_dataset_iterator(combined_data, batch_size)
    
    return iterator, combined_data


def test():

    # inp_path = '/mnt/ext6TB/fio/data/scale_4_angles_16/explosion_mix_data_final_p.npy'
    # out_path = '/mnt/ext6TB/fio/data/scale_4_angles_16/explosion_mix_data_source_p.npy'
    inp_path = '/home/konik/Documents/FIO/data/wp_in_0p2_128.npy'
    out_path = '/home/konik/Documents/FIO/data/wp_out_0p2_128.npy'
    iterator, _ = get_iterator(inp_path, out_path, 8)

    inp_dec, out_dec = iterator.get_next()
    print(inp_dec.get_shape().as_list())

    # inp_img, inp_coeffs = inp_iterator.get_next()
    # out_img, out_coeffs = out_iterator.get_next()

    # inp_splits = d.split(inp_coeffs)
    # out_splits = d.split(out_coeffs)

    with tf.Session() as sess:
        for i in range(1000):
            t = time()
            g_r, g = sess.run([inp_dec, out_dec])
            print(g.dtype)
            print(g.shape)
            print(g_r.dtype)
            print(g_r.shape)
            print('%d- Took %fs' % (i, time()-t))
            t = time()


if __name__ == '__main__':
    test()
