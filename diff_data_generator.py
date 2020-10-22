#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 14:02:31 2020

@author: konik
"""

import numpy as np
from time import time
from scipy.io import loadmat
import matplotlib.pyplot as plt
import multiprocessing as mp
from matplotlib import collections as mc

import pyct as ct

def op_shapes(A):
    shapes = []
    for size in A.sizes:
        shapes.extend(size)
    return shapes

def op_lengths(A):
    shapes = op_shapes(A)
    return [np.prod(s) for s in shapes]

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


def get_box_decomposition(A, img, boxid):
    coeffs = A.fwd(img)
    chat = A.struct(coeffs)
    n = sum([len(s) for s in chat])
    
    coeffs_recon = np.zeros((n,)+img.shape)
    c = 0
    
    for i in range(len(chat)):
        for j in range(len(chat[i])):
            if c==boxid:
                zero_array = np.zeros_like(coeffs)
                zero_array = A.struct(zero_array)

                zero_array[i][j] = chat[i][j].copy()                        
                coeffs_recon[c] = A.inv(A.vect(zero_array))
            
            c+=1
    
    return coeffs_recon

def binarize(img, thresh):
    img = img.copy()
    img[np.abs(img)>thresh]=1
    img[np.abs(img)!=1] = 0
    
    return img

def get_avg_direction(A, img, box_vecs):
    n = analyze_norms(A, img).reshape(-1,1)
    bv_avg = np.sum(box_vecs*n, axis=0)/(np.sum(n) + 1e-12)
    bv_avg_norm = np.linalg.norm(bv_avg)
    return bv_avg/(bv_avg_norm + 1e-12)
    

def get_center(img_, cx_ar = None, return_img=False):
    n, m = img_.shape
    
    img = np.abs(img_.copy())
    
#     img = binarize(img_, 1e-2)
    
    nn = np.arange(n, dtype=np.int32)
    mm = np.arange(m, dtype=np.int32)
    
    N, M = np.meshgrid(nn, mm)
    grid = np.stack((N,M), axis=-1)
    
    if return_img:
        ret_img = np.zeros((n,m,2))

    if cx_ar is not None:
        cx, cy = cx_ar
        
        # check for vertical or horizontal line
        isvertical = True
        bimg = binarize(y, 1e-2)
        if np.sum(bimg[:,cy]) >= np.sum(bimg[cx,:]):
            isvertical = False


        if isvertical:
            cond = grid[:,:,0]<cx
        else:
            cond = grid[:,:,1]<cy

        ncond = np.logical_not(cond)            
        c = np.ones((2,2))*-np.array(img.shape)

        if np.sum(img[cond]) > 1:
            c[0] = np.sum(img[cond].reshape(-1,1)*grid[cond], axis=0)/np.sum(img[cond])
        
        if np.sum(img[ncond]) > 1:
            c[1] = np.sum(img[ncond].reshape(-1,1)*grid[ncond], axis=0)/np.sum(img[ncond])
        
        if return_img:
            ret1 = ret_img[:,:,0]
            ret2 = ret_img[:,:,1]
            ret1[cond] = img_[cond]
            ret2[ncond] = img_[ncond]
            ret_img[:,:,0] = ret1
            ret_img[:,:,1] = ret2
            return c.astype(np.int32), ret_img
        else:
            return c.astype(np.int32)
    
    else:

        c = np.sum(img[:,:,np.newaxis]*grid, axis=(0,1))/np.sum(img)
        return c.reshape(1,2).astype(np.int32)

def T(x):
    return np.array([(x[0]-256)/512.0, (x[1]-256)/512.0])
    
def analyze_example_fast(x, y, bv, A, plot=False):
    
    cx = get_center(x)
    sx = analyze_norms(A, x).reshape(-1,1)
    b_avg = np.mean(bv*sx, axis=0)/np.sum(sx)
    b_avg /= np.linalg.norm(b_avg)
    cy, img = get_center(y, cx[0], return_img=True)
    
    t_bvec1 = get_avg_direction(A, img[:,:,0], bv)
    t_bvec2 = get_avg_direction(A, img[:,:,1], bv)
    
    result = np.concatenate((T(cx[0]), b_avg,
                             T(cy[0]), t_bvec1,
                             T(cy[1]), t_bvec2), axis=0)

    if plot:
        fig, ax = plt.subplots(1,2, figsize=(20,10))
        ax[0].imshow(x)
        ax[0].scatter(cx[:,0],cx[:,1])
        ax[1].imshow(y)
        ax[1].scatter(cy[:,0],cy[:,1])
        plt.show()
    return result

TIME = 'hr_0p2_1'
IS = 1024
inp = np.load('../data/wp_in_%s.npy'%TIME)
out = np.load('../data/wp_out_%s.npy'%TIME)

A = ct.fdct2((IS,IS), 4, 16, False, norm=False, cpx=False)

#shapes = op_shapes(A)
#lengths = op_lengths(A)
#ids = np.cumsum([0]+lengths)

bv = np.load('../jspace/box_vecs_unit.npy')

t = time()
#final = np.zeros((len(inp), 12))
final = np.load('final.npy')

for i in range(750, 1000):
    x = inp[i,:,:,0]
    y = out[i,:,:,0]
    final[i] = analyze_example_fast(x,y,bv,A,plot=False)
    if i%50==0:
        print('Finished %d. in %fs'%(i+1, time()-t))
        np.save('final.npy', final)
        t = time()