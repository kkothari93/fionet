import numpy as np
import tensorflow as tf

def adverserial_loss(yhat, y, ret_summ_slopes=True, ground_metric='l2'):

    from discriminator import discriminator

    def gp(grads, interpolates):
        slopes = tf.sqrt(tf.reduce_sum(
            tf.square(grads), reduction_indices=[1, 2, 3]))
        if ret_summ_slopes:
            summ_slopes = tf.summary.histogram('slopes_summary', slopes)
        else:
            summ_slopes = None

        gradient_penalty = tf.reduce_mean((slopes-1.)**2)
        return gradient_penalty, summ_slopes 

    def gp_w2(grads, interpolates, nbr_rad = 7):
        """
        calculate the gradient in W2 metric in pixel space
        Algorithm 2 of http://proceedings.mlr.press/v97/dukler19a/dukler19a.pdf
        """
        if nbr_rad%2==0:
            nbr_rad += 1

        def compute_d(nbr_rad, IS):
            """Computes degree of each node"""
            from scipy.ndimage.filters import convolve

            N = nbr_rad
            
            a = np.arange(N)
            xx, yy = np.meshgrid(a,a)
            c = np.array([[xx[N//2,N//2], yy[N//2,N//2]]])
            
            kernel = np.linalg.norm(np.stack((xx.ravel(), yy.ravel()), axis=-1) - c, axis=-1)
            kernel /= kernel.sum()
            kernel = kernel.reshape(N,N)

            ones = np.ones((IS,IS))
            out = convolve(ones, kernel, mode='constant', cval=0)
            out /= out.sum()

            return out.reshape(1,IS,IS,1)

        def compute_kernels(nbr_rad):
            N = nbr_rad
            Nsqm1 = N**2 - 1

            a = np.arange(N**2)

            K = np.zeros((N, N, N**2))
            K[a%N,a//N, a] = -1 
            K[N//2,N//2, :] = 1
            # delete ii term
            K = np.delete(K, N**2//2, -1)
            
            M = np.zeros_like(K)
            M[a%N,a//N,a] = 0.5
            M[N//2,N//2,:] = 0.5
            # delete ii term
            M = np.delete(M, N**2//2, -1)
            
            # conv2d requires shape [filter_height, filter_width, in_channels, out_channels]
            return K.reshape(N,N,1,Nsqm1), M.reshape(N,N,1,Nsqm1)


        IS = grads.get_shape().as_list()[1] # get image size
        d = compute_d(nbr_rad, IS)
        K, M = compute_kernels(nbr_rad)

        K = tf.convert_to_tensor(K, dtype=tf.float32)
        M = tf.convert_to_tensor(M, dtype=tf.float32)
        d = tf.convert_to_tensor(d, dtype=tf.float32)

        # alpha D^T 11^T D
        DT_one = tf.reduce_sum(grads, axis=1, keepdims=True)
        oneT_D = tf.transpose(DT_one, [0,2,1,3])
        add_term = W2_ALPHA * DT_one * oneT_D
 
        H = tf.nn.conv2d(grads, K, [1,]*4, "SAME")
        V = tf.nn.conv2d(interpolates/d, M, [1,]*4, "SAME")
        
        H = H*H
        W = V*H

        W = tf.reduce_sum(W, reduction_indices=[1,2,3])
        slopes = tf.math.sqrt(W + add_term)
        if ret_summ_slopes:
            summ_slopes = tf.summary.merge([
                tf.summary.histogram('slopes_summary', slopes),
                tf.summary.scalar()])
        else:
            summ_slopes = None

        return tf.reduce_mean(tf.square(slopes-1)**2), summ_slopes

    def gradient_penalty(d, yhat, y):
        """WGAN-GP loss of I. Gulrajani et al (2017)"""
        bs = tf.shape(yhat)[0]
        alpha = tf.random_uniform(shape=[bs, 1, 1, 1], minval=0., maxval=1.)
        differences = yhat - y
        interpolates = y + (alpha * differences)
        gradients = tf.gradients(
            d(interpolates, reuse=True), [interpolates])[0]
        if ground_metric=='l2':
            gp_func = gp
        else:
            gp_func = gp_w2


        return gp_func(gradients, interpolates)

    d_hat = discriminator(yhat)
    d_real = discriminator(y, reuse=True)

    gploss, summ_slopes = gradient_penalty(discriminator, yhat, y)

    D_loss = tf.reduce_mean(d_hat) - tf.reduce_mean(d_real)  + 10*gploss
    G_loss = -tf.reduce_mean(d_hat)

    summ_slopes = tf.summary.merge([summ_slopes,
        tf.summary.scalar(
            'd_score', tf.reduce_mean(d_real)),
        tf.summary.scalar(
            'g_score', tf.reduce_mean(d_hat)),
        tf.summary.scalar(
            'WofW', tf.reduce_mean(d_real) - tf.reduce_mean(d_hat))])

    disc_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')

    return D_loss, G_loss, disc_vars, summ_slopes