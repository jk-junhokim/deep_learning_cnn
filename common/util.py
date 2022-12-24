import numpy as np

def smooth_curve(x):
    """
    http://glowingpython.blogspot.jp/2012/02/convolution-with-numpy.html
    """
    window_len = 11
    s = np.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]]
    w = np.kaiser(window_len, 2)
    y = np.convolve(w/w.sum(), s, mode='valid')
    return y[5:len(y)-5]


def shuffle_dataset(x, t):
    """
    Input
    x : train data
    t : answer label
    
    Returns
    x, t : shuffled data
    """
    permutation = np.random.permutation(x.shape[0])
    x = x[permutation,:] if x.ndim == 2 else x[permutation,:,:,:]
    t = t[permutation]

    return x, t


def conv_output_size(input_size, filter_size, stride=1, pad=0):
    return (input_size + 2*pad - filter_size) / stride + 1


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """
    Reduces Dimension of Input Data
    
    Parameters
    input_data : (number of batch, channel, height, width)
    filter_h : height
    filter_w : width
    pad : padding
    stride : stride
    
    Returns:
    col : 2 dimensional array
    """
    N, C, H, W = input_data.shape
    output_h = (H + 2*pad - filter_h)//stride + 1
    output_w = (W + 2*pad - filter_w)//stride + 1

    # padding before weight multiplication
    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    # initialize output data
    col = np.zeros((N, C, filter_h, filter_w, output_h, output_w))

    # only apply stride. not multiplication step.
    # we apply stride before the multiplication in order to expedite the multiplication step.
    # since stride typically overlaps the same input data,
    # 'col' lacks space-wise efficiency.
    # however the time complexity improves due to prearranged data.
    for y in range(filter_h):
        y_max = y + stride * output_h
        for x in range(filter_w):
            x_max = x + stride * output_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*output_h*output_w, -1)
    
    return col

def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """
    Backward for im2col

    Parameters
    col : 2 dimensional matrix
    input_shape : original data shape（ex. (10, 1, 28, 28)）
    filter_h : filter height
    filter_w : filter width
    stride : stide
    pad : padding
    
    Returns
    img : transformed images
    """
    N, C, H, W = input_shape
    output_h = (H + 2*pad - filter_h)//stride + 1
    output_w = (W + 2*pad - filter_w)//stride + 1
    col = col.reshape(N, output_h, output_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride*output_h
        for x in range(filter_w):
            x_max = x + stride*output_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]