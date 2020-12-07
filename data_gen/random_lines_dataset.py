import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imsave

def random_lines_dataset(Ndata):

    n = 512
    n_seg_min = 1
    n_seg_max = 5#30
    min_length = 20
    max_length = 100
    max_width = 12
    min_width = 6
    min_amp = .2

    for k in range(Ndata):
        n_range = np.arange(n)
        xv, yv = np.meshgrid(n_range, n_range)
        gp = np.vstack((xv.reshape(1, n**2), yv.reshape(1, n**2)))
        n_seg = int(np.random.rand()*(n_seg_max-n_seg_min)) + n_seg_min

        f = np.zeros((n, n))

        for i in range(n_seg):
            center = np.floor(np.random.rand(2, 1) * n)
            angle = np.random.rand(1) * np.pi
            unit_dir = np.array([np.cos(angle), np.sin(angle)])
            unit_ort = np.array([-np.sin(angle), np.cos(angle)])
            length = np.random.rand(1) * (max_length - min_length) + min_length
            width = np.random.rand(1) * (max_width - min_width) + min_width
                
            xi1 = np.dot( unit_dir.T, gp - center)
            xi2 = np.dot( unit_ort.T, gp - center)
            idx = (np.abs(xi1) <= length / 2) & (np.abs(xi2) <= width / 2)
            amp = np.random.rand(1) * (1 - min_amp) + min_amp
                
            f += amp * idx.astype(float).reshape((n, n))

        if not os.path.exists('thick_lines/'):
            os.makedirs('thick_lines/')

        imsave(os.path.join('thick_lines/','%d.png'%k), f)

    return None

if __name__ == '__main__':
    random_lines_dataset(3000)
