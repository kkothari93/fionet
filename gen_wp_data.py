import pyct as ct
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat

from jspaceDataGenerator import CL

class wpdata(CL):
	def __init__(self, size, nscale=4, nangles=16):
		CL.__init__(self, size, nscale, nangles)

		self.img_size = size
		

		self.sizes = []
		for s in self.A.sizes:
			self.sizes.extend(s)

		self.__lengths = [np.prod(s) for s in self.sizes]
		self.coeff_size = np.sum(self.__lengths)

		self.__allw_indices = self.__get_idx_for_gen_wp()



	def __get_idx_for_gen_wp(self):
		"""Get the middle indices of each box"""
		
		offset = 0
		allowed_indices = []
		coeffs = np.zeros_like(self.A.fwd(np.random.rand(*self.img_size)))

		for i, s in enumerate(self.sizes[:-1]):
			if i>0:
				offset += self.__lengths[i-1]
			idx = np.arange(self.__lengths[i]).reshape(s) + offset
			m, n = idx.shape
			idx = idx[m//4:-m//4, n//4: -n//4]
			allowed_indices.extend(list(idx.ravel()))

		return allowed_indices


	@staticmethod
	def analyze_norms(A, img):
		coeffs = A.fwd(img)
		chat = A.struct(coeffs)

		norms = np.zeros(50)
		c = 0

		for arr_list in chat:
			for arr in arr_list:
				sqrt_shape = np.sqrt(np.prod(arr.shape))
				norms[c] = np.linalg.norm(arr)/sqrt_shape
				c += 1
		return norms

	def gen_wp(self, n, plot_samples=True, save_data=True, matname='trial'):
		sizes = []
		coeffs = np.zeros((n,) + (np.sum(self.__lengths), ))

		idx = np.random.choice(self.__allw_indices, n, replace=False)
		idx = (np.arange(n), idx)

		coeffs[idx] = 100.0

		imgs = CL.mp_synthesize(self, coeffs)
		# c_r = CL.mp_analyze(self, imgs)


		## plot a gsxgs grid of random 16 images chosen from the n images
		# gs = grid_size
		if plot_samples:
			gs = 2 
			idx = np.random.choice(np.arange(n), gs**2)
			chosen_imgs = imgs[idx]

			fig, ax = plt.subplots()

			im = ax.imshow(chosen_imgs.reshape(
				(gs,gs)+self.img_size).swapaxes(1,2).reshape(gs*self.img_size[0], -1))
			plt.show()

		if save_data:
			savemat(matname, {'data': imgs})

		return imgs, coeffs



def analyze_mat(path, cl):
	d = loadmat(path)
	p0 = d['img']
	pf = d['p_f']

	norms_0 = cl.analyze_norms(cl.A, p0)
	norms_f = cl.analyze_norms(cl.A, pf)

	fig, ax = plt.subplots(1,2)
	ax[0].bar(np.arange(50), norms_0)
	ax[1].bar(np.arange(50), norms_f)

	plt.show()

def check_consistency(cl):

	orig = np.random.rand(*cl.img_size)
	trial_c = cl.A.fwd(orig)
	recon = cl.A.inv(cl.A.fwd(orig))

	print('consistency in image')
	print(np.linalg.norm(orig-recon))

	# orig = np.random.rand(cl.coeff_size)
	orig = trial_c
	recon = cl.A.fwd(cl.A.inv(orig))
	print('consistency in coeffs')
	print(np.linalg.norm(orig-recon))

	return

def check_consistency_on_data(imgs, coeffs, cl):
	img = imgs[0]
	c = coeffs[0]
	print(np.where(c!=0))

	print('consistency error in coeffs:')
	print(np.linalg.norm(cl.A.fwd(img) - c))

	print('consistency error in img:')
	print(np.linalg.norm(cl.A.inv(c) - img))

	print('self-consistency:')
	print(np.linalg.norm(cl.A.inv(cl.A.fwd(img)) - img))
	print(np.linalg.norm(cl.A.fwd(cl.A.inv(c)) - c))

	return

def check(cl):

	c = np.zeros(cl.coeff_size)
	c[40228] = 1.0
	img = cl.A.inv(c)
	c_r = cl.A.fwd(img)

	c_r[:np.prod(cl.sizes[0])] = 0.0
	img_r = cl.A.inv(c_r)
	# c_r = cl.A.inv(cl.A.fwd(cl.A.inv(c_r)))

	norms = cl.analyze_norms(cl.A, img)
	norms_r = cl.analyze_norms(cl.A, img_r)
	fig, ax = plt.subplots(1,2)
	ax[0].bar(np.arange(50), norms)
	ax[1].bar(np.arange(50), norms_r)

	plt.show()

def analyze_data(data, cl):

	norms_0 = cl.analyze_norms(cl.A, data[0])
	norms_f = cl.analyze_norms(cl.A, data[1])

	fig, ax = plt.subplots(1,2)
	ax[0].bar(np.arange(50), norms_0)
	ax[1].bar(np.arange(50), norms_f)

	plt.show()


import sys

n, matname = sys.argv[1:]
n = int(n)
data_gen = wpdata((512, 512))
data, coeffs = data_gen.gen_wp(n, plot_samples=False, matname=matname)
# print(data.shape)
# print(coeffs.shape)
# check_consistency_on_data(data, coeffs, data_gen)
# check_consistency(data_gen)

# analyze_data(data, data_gen)
# analyze_mat(
# 	'/home/konik/Documents/FIO/kwave/final_p_single_wp.mat', data_gen)
