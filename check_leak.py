import numpy as np
import pyct as ct

img = np.random.rand(128,128)
A = ct.fdct2((128,128), 4, 16, False, norm=False, cpx=False)
c = np.random.rand(49577) #length of coeffs for 128x128 image

def fwd(A, img):
	t = A.fwd(img)
	return t

def inv(A, c):
	return A.inv(c)

def check_leak_fwd(A, img):
	coeffs = np.zeros(49577)
	for i in range(5000):
		coeffs = fwd(A, img)
		# coeffs = A.fwd(img)

	return

def check_leak_inv(A, c):
	img = np.zeros((128, 128))
	for i in range(5000):
		# img = inv(A, c)
		img = A.inv(c)

	return 

def segfault():
	shape = (128,128,128)
	a = np.random.randn(*shape)
	F = ct.fdct3(shape, 4, 8, True)
	F.fwd(a)

if __name__=='__main__':
	check_leak_fwd(A, img)
