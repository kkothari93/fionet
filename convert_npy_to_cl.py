import numpy as np
import pyct as ct
from tqdm import tqdm

NSCALE = 4
NANGLES = 16

def curvelet_decomposition(A, img):
    coeffs = A.fwd(img)
    lengths = [np.prod(g) for d in A.sizes for g in d]
    N = len(lengths)

    lengths = np.cumsum([0]+lengths)

    out = np.zeros(img.shape + (N,))
    for i in range(N):
        z = np.zeros_like(coeffs)

        s = lengths[i]
        e = lengths[i+1]
        z[s:e] = coeffs[s:e] 

        out[:,:,i] = A.inv(z)

    return out


def convert_npy_to_cl(A, arr):
    lengths = [np.prod(g) for d in A.sizes for g in d]
    nboxes = len(lengths)
    N = len(arr)

    out = np.zeros((N,) + arr.shape[1:-1] + (nboxes,))
    
    for i in tqdm(range(N)):
        out[i] = curvelet_decomposition(A, arr[i,:,:,0])

    return out

def main():
    import sys

    filepath = sys.argv[1]
    x = np.load(filepath)

    try:
        N = int(sys.argv[2])
        x = x[:N]
        print('%d images found!'%N)
    except:
        N = len(x)

    try:
        noise_level = int(sys.argv[3])
        add_noise = True
        print('Adding noise at %ddB level'%noise_level)
    except:
        add_noise = False


    noise = np.random.randn(*x.shape)
    noise_str = '_%ddB'%noise_level if add_noise else ''
    if add_noise:
        noise_var = noise*np.linalg.norm(x.reshape(N,-1), axis=-1).reshape(N,1,1,1)*10**(-noise_level/20.0)/np.sqrt(np.prod(x.shape[1:3]))
        x = x + noise_var

    print('loaded file')
    print('convert %d images'%N)

    A = ct.fdct2(x.shape[1:3], NSCALE, NANGLES, False, norm=False, cpx=False)
    x_cl = convert_npy_to_cl(A, x)

    np.save(filepath.split('.')[0] + noise_str + '_cl.npy', x_cl)

    return x_cl



if __name__ == '__main__':
    main()
