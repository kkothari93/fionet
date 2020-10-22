import os, glob
import numpy as np
import tensorflow as tf

# adapted from https://gist.github.com/swyoon/8185b3dcf08ec728fb22b99016dd533f"
__author__ = "Sangwoong Yoon"
IMG_SIZE = 128  

def np_to_tfrecords(X, Y, writer, verbose=True):
    """
    Converts a Numpy array (or two Numpy arrays) into a tfrecord file.
    For supervised learning, feed training inputs to X and training labels to Y.
    For unsupervised learning, only feed training inputs to X, and feed None to Y.
    The length of the first dimensions of X and Y should be the number of samples.
    
    Parameters
    ----------

    X : numpy.ndarray of rank 2
        Numpy array for training inputs. Its dtype should be float32, float64, or int64.
        If X has a higher rank, it should be rshape before fed to this function.
    Y : numpy.ndarray of rank 2 or None
        Numpy array for training labels. Its dtype should be float32, float64, or int64.
        None if there is no label array.
    file_path_prefix : str
        The path and name of the resulting tfrecord file to be generated, without '.tfrecords'
    verbose : bool
        If true, progress is reported.
    
    Raises
    ------
    ValueError
        If input type is not float (64 or 32) or int.
    
    """
    def _dtype_feature(ndarray):
        """match appropriate tf.train.Feature class with dtype of ndarray. """
        assert isinstance(ndarray, np.ndarray)
        dtype_ = ndarray.dtype
        if dtype_ == np.float64 or dtype_ == np.float32:
            return lambda array: tf.train.Feature(float_list=tf.train.FloatList(value=array))
        elif dtype_ == np.int64:
            return lambda array: tf.train.Feature(int64_list=tf.train.Int64List(value=array))
        else:  
            raise ValueError("The input should be numpy ndarray. \
                               Instaed got {}".format(ndarray.dtype))
            
    assert isinstance(X, np.ndarray)
    assert len(X.shape) == 2  # If X has a higher rank, 
                               # it should be rshape before fed to this function.
    assert isinstance(Y, np.ndarray) or Y is None
    
    # load appropriate tf.train.Feature class depending on dtype
    dtype_feature_x = _dtype_feature(X)
    if Y is not None:
        assert X.shape[0] == Y.shape[0]
        assert len(Y.shape) == 2
        dtype_feature_y = _dtype_feature(Y)            
    
    if verbose:
        print("Serializing {:d} examples into {}".format(X.shape[0], result_tf_file))
        
    # iterate over each sample,
    # and serialize it as ProtoBuf.
    for idx in range(X.shape[0]):
        x = X[idx]
        if Y is not None:
            y = Y[idx]
        
        d_feature = {}
        d_feature['X'] = dtype_feature_x(x)
        if Y is not None:
            d_feature['Y'] = dtype_feature_y(y)
            
        features = tf.train.Features(feature=d_feature)
        example = tf.train.Example(features=features)
        serialized = example.SerializeToString()
        writer.write(serialized)
    
    if verbose:
        print("Writing {} done!".format(result_tf_file))

def convert_to_cl(cl, imgs):
    if len(imgs.shape) == 4:
        ## lose channel dimension
        imgs = imgs[:,:,:,0]
    out = cl.mp_decompose(imgs)
    return out


def check_files(dirpath):
    files =  glob.glob(dirpath+'*.tfrecords') 
    filesSize = len(files)
    cnt = 0

    for filename in files:
        cnt = cnt + 1
        print('checking %d/%d %s' % (cnt, filesSize, filename))
        try:
            for i, example in enumerate(tf.python_io.tf_record_iterator(filename)): 
            
                tf_example = tf.train.Example.FromString(example) 
        except:
            print("Record %d in %s" % (i, filename))    
            # os.remove(filename)

    return 


def main(inp_path, out_path, dirpath):
    import os
    from tqdm import tqdm
    from jspaceDataGenerator import CL, NSCALE, NANGLES, NBOXES
    inp = np.load(inp_path).astype(np.float32)
    out = np.load(out_path).astype(np.float32)

    cld = CL(inp.shape[1:3], NSCALE, NANGLES)

    samples_per_part = 100
    print('Found %d samples'%len(inp))

    parts = inp.shape[0]//samples_per_part


    try:
        os.makedirs(dirpath) 
    except:
        print('%s exists'%dirpath)

    for part_num in tqdm(range(parts)):
        s = part_num*samples_per_part
        e = (part_num+1)*samples_per_part
        
        inp_cl = convert_to_cl(cld, inp[s:e])
        out_cl = convert_to_cl(cld, out[s:e])

        inp_cl = inp_cl.reshape(samples_per_part, -1)
        out_cl = out_cl.reshape(samples_per_part, -1)

        # inp_cl/ out_cl have shape (samples_per_part,)+inp.shape[1:3]+(NBOXES,)

        # Generate tfrecord writer
        result_tf_file = os.path.join(dirpath + '%d.tfrecords'%part_num)
        writer = tf.python_io.TFRecordWriter(result_tf_file)

        np_to_tfrecords(inp_cl, out_cl, writer, verbose=False)

    return






#################################    
##      Test and Use Cases     ##
#################################

def original_tests():


    # 1-1. Saving a dataset with input and label (supervised learning)
    xx = np.random.randn(10,5)
    yy = np.random.randn(10,1)
    np_to_tfrecords(xx, yy, 'test1', verbose=True)

    # 1-2. Check if the data is stored correctly
    # open the saved file and check the first entries
    for serialized_example in tf.python_io.tf_record_iterator('test1.tfrecords'):
        example = tf.train.Example()
        example.ParseFromString(serialized_example)
        x_1 = np.array(example.features.feature['X'].float_list.value)
        y_1 = np.array(example.features.feature['Y'].float_list.value)
        break
        
    # the numbers may be slightly different because of the floating point error.
    print(xx[0])
    print(x_1)
    print(yy[0])
    print(y_1)


    # 2. Saving a dataset with only inputs (unsupervised learning)
    xx = np.random.randn(100,100)
    np_to_tfrecords(xx, None, 'test2', verbose=True)

    return None

def read_tfrecords(data_path, repeat=True, batch_size=8, img_size=128):
    import os, glob
    from time import time
    from tqdm import tqdm

    IMG_SIZE = img_size

    # Create a description of the features.
    feature_description = {
        'X': tf.io.FixedLenFeature([IMG_SIZE*IMG_SIZE*50], tf.float32),
        'Y': tf.io.FixedLenFeature([IMG_SIZE*IMG_SIZE*50], tf.float32)
    }

    def _parse_function(example_proto):
      # Parse the input `tf.Example` proto using the dictionary above.
      parsed_example = tf.parse_single_example(example_proto, feature_description)
      parsed_example['X'] = tf.reshape(parsed_example['X'], (IMG_SIZE,IMG_SIZE, 50))
      parsed_example['Y'] = tf.reshape(parsed_example['Y'], (IMG_SIZE,IMG_SIZE, 50))
      return parsed_example

    # data_path = '/home/konik/Documents/FIO/data/' + 'explosion_thick_lines_0p2/'
    filenames = glob.glob(data_path+'*.tfrecords')
    data = tf.data.TFRecordDataset(filenames, num_parallel_reads = 4)
    data = data.map(_parse_function)
    if repeat:
        data = data.repeat()
    data = data.batch(batch_size).prefetch(10*batch_size).shuffle(10*batch_size)

    iterator = data.make_one_shot_iterator()

    features = iterator.get_next()
    print(features)

    x = features['X']
    y = features['Y']

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        xv, yv = sess.run([x,y])
        np.save('sens_trace.npy', xv)
        np.save('source_p.npy', yv)

    return x, y


if __name__ == '__main__':
    root = '/home/konik/fiodata/full_rtm_128/'
    inp_path = root+'explosion_full_rtm_refl_3_0p2_sens_trace.npy'
    out_path = root+'explosion_full_rtm_refl_3_0p2_source_p.npy'
    dirpath = root+'explosion_full_rtm_refl_3_0p2_sens_trace/'
    main(inp_path, out_path, dirpath)
    check_files(dirpath)
    # read_tfrecords(dirpath)

