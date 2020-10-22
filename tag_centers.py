#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 13:31:54 2020

@author: konik
"""
import matplotlib.pyplot as plt
import numpy as np

x = np.load('../data/wp_out_0p2_1.npy')
y = np.load('../data/wp_in_0p2_1.npy')
N = len(x)

def get_center(*args):
    img_ = args[0]
    n, m = img_.shape
    
    img = np.abs(img_.copy())
    
#     img = binarize(img_, 1e-2)
    
    nn = np.arange(n, dtype=np.int32)
    mm = np.arange(m, dtype=np.int32)
    
    N, M = np.meshgrid(nn, mm)
    grid = np.stack((N,M), axis=-1)


    if len(args)>1:
        cx, cy = args[1]
        
        # check for vertical or horizontal line
        isvertical = False
        if np.sum(np.abs(img[:,cy])) <= np.sum(np.abs(img[cx,:])):
            isvertical = True

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
        
        return c.astype(np.int32)
    
    else:

        c = np.sum(img[:,:,np.newaxis]*grid, axis=(0,1))/np.sum(img)
        return c.reshape(1,2).astype(np.int32)


def log_img(x):
    return np.log10(np.abs(x)+1e-10)

fig, ax = plt.subplots(figsize=(16,16))
ax.set_xlim([-128, 640])
ax.set_ylim([-128, 640])
i = 0
centers = np.ones((2*N, 2))*-512
a = plt.imshow(log_img(x[0,:,:,0]+y[0,:,:,0]))
b = plt.scatter([],[],color='b',s=50,picker=5)


null = b.get_offsets()

def next_image(event):
    global a, b, null
    
    global i, centers, N
    centers[2*i:2*(i+1)] = b.get_offsets()
    i += 1
    if i==N:
        plt.close('all')
    a.set_data(log_img(x[i,:,:,0]+y[i,:,:,0]))
    b.set_offsets(null)
    plt.draw()
    
    
    


def add_or_remove_point(event):
    global a, b
    
    if event.key == 'n':
        print('here')
        a.remove()
        b.remove()
        plt.draw()


    xydata_b = b.get_offsets()
    
    #click x-value
    xdata_click = event.xdata
    ydata_click = event.ydata
    newdata_pt = np.array([xdata_click, ydata_click])
    
    def find_closest(pt, arr):
        d = np.argmin(np.linalg.norm(arr - pt.reshape(1,2), axis=-1))
        return np.concatenate((arr[:d], arr[d+1:]), axis=0)


    if event.button == 1:
        #insert new scatter point into b    
        new_xydata_b = np.insert(xydata_b,0,newdata_pt,axis=0)
        #update b
        b.set_offsets(new_xydata_b)
        plt.draw()
    
    elif event.button == 3:
        new_xydata_b = find_closest(newdata_pt, xydata_b)
        b.set_offsets(new_xydata_b)
        plt.draw()

fig.canvas.mpl_connect('button_press_event',add_or_remove_point)
fig.canvas.mpl_connect('key_press_event',next_image)
fig.suptitle('%d'%i)