import os
import argparse
import numpy as np
import tensorflow as tf


def get_pixel_value(img, x, y):
    """
    Utility function to get pixel value for coordinate
    vectors x and y from a  4D tensor image.
    Input
    -----
    - img: tensor of shape (B, H, W, C)
    - x: flattened tensor of shape (B*H*W,)
    - y: flattened tensor of shape (B*H*W,)
    Returns
    -------
    - output: tensor of shape (B, H, W, C)
    """
    shape = tf.shape(x)
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]

    batch_idx = tf.range(0, batch_size)
    batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))
    b = tf.tile(batch_idx, (1, height, width))

    indices = tf.stack([b, y, x], 3)

    return tf.gather_nd(img, indices)

def bilinear_sampler(img, x, y):
    """
    Performs bilinear sampling of the input images according to the
    normalized coordinates provided by the sampling grid. Note that
    the sampling is done identically for each channel of the input.
    To test if the function works properly, output image should be
    identical to input image when theta is initialized to identity
    transform.
    Input
    -----
    - img: batch of images in (B, H, W, C) layout.
    - grid: x, y which is the output of affine_grid_generator.
    Returns
    -------
    - out: interpolated images according to grids. Same size as grid.
    """
    H = tf.shape(img)[1]
    W = tf.shape(img)[2]
    max_y = tf.cast(H - 1, 'int32')
    max_x = tf.cast(W - 1, 'int32')
    zero = tf.zeros([], dtype='int32')

    # rescale x and y to [0, W-1/H-1]
    x = tf.cast(x, 'float32')
    y = tf.cast(y, 'float32')
    x = x * tf.cast(max_x-1, 'float32')
    y = y * tf.cast(max_y-1, 'float32')

    # grab 4 nearest corner points for each (x_i, y_i)
    x0 = tf.cast(tf.floor(x), 'int32')
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), 'int32')
    y1 = y0 + 1

    # clip to range [0, H-1/W-1] to not violate img boundaries
    x0 = tf.clip_by_value(x0, zero, max_x)
    x1 = tf.clip_by_value(x1, zero, max_x)
    y0 = tf.clip_by_value(y0, zero, max_y)
    y1 = tf.clip_by_value(y1, zero, max_y)

    # get pixel value at corner coords
    Ia = get_pixel_value(img, x0, y0)
    Ib = get_pixel_value(img, x0, y1)
    Ic = get_pixel_value(img, x1, y0)
    Id = get_pixel_value(img, x1, y1)

    # recast as float for delta calculation
    x0 = tf.cast(x0, 'float32')
    x1 = tf.cast(x1, 'float32')
    y0 = tf.cast(y0, 'float32')
    y1 = tf.cast(y1, 'float32')

    # calculate deltas
    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    # add dimension for addition
    wa = tf.expand_dims(wa, axis=3)
    wb = tf.expand_dims(wb, axis=3)
    wc = tf.expand_dims(wc, axis=3)
    wd = tf.expand_dims(wd, axis=3)

    # compute output
    out = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])

    return out

def move_ch_to_batch(x):
    """Moves the channels to batch dimension
    Input x of shape bsxHxWxC --> bs*CxHxWx1
    """
    bs = tf.shape(x)[0]
    H = tf.shape(x)[1]
    W = tf.shape(x)[2]
    C = tf.shape(x)[3]

    x = tf.reshape(tf.transpose(x, [0,3,1,2]), (bs*C,H,W,1))
    return x

def move_ch_back(x, C):
    """Moves the channels back to last axis from batch dimension
    Input x of shape bs*CxHxWxC --> bsxHxWxC
    """
    bs = tf.shape(x)[0]/C
    H = tf.shape(x)[1]
    W = tf.shape(x)[2]

    x = tf.transpose(tf.reshape(x, (bs,C,H,W)), [0,2,3,1])

    return x


def safemkdir(logdir):
    try:
        os.mkdir(logdir)
    except:
        print('%s exists!'%logdir)
        pass
    return

def flags():
    parser = argparse.ArgumentParser()
    parser.add_argument(
      '--head_start', type=int,
      default=5000, 
      help='head start for diffeos.')

    parser.add_argument(
      '--diff_load_dir', type=str, 
      default='./hj_flow_yxi2xeta_ccbg2/',
      help='Diffeo directory')

    parser.add_argument(
      '--log_dir', type=str, 
      default='SingleBigDiffeo_isp_unet_thick_lines_0p2/', 
      help='log directory')

    parser.add_argument(
      '--data_dir',
      type=str,
      default='../data/',
      help='data directory')

    parser.add_argument(
      '--diff_lr',
      type=float,
      default=1e-6,
      help='data directory')

    parser.add_argument(
      '--w2_alpha',
      type=float,
      default=1.0,
      help='data directory')

    parser.add_argument(
      '--ground_metric',
      type=str,
      default='l2',
      help='data directory')

    parser.add_argument(
      '--lam',
      type=float,
      default=0.01,
      help='data directory')

    parser.add_argument(
      '--diter',
      type=int,
      default=4,
      help='number of d iterations')

    parser.add_argument('--train', dest='train', action='store_true')
    parser.add_argument('--notrain', dest='train', action='store_false')
    parser.set_defaults(train=True)

    FLAGS, unparsed = parser.parse_known_args()

    return FLAGS, unparsed
