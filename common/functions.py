import numpy as np

"""
ACTIVATION FUNCTIONS
"""
### IDENTITY FUNCTION ###
def identity_function(x):
    return x


### STEP FUNCTION ###
def step_function(x):
    return np.array(x > 0, dtype = np.int)

    
### SIGMOID FUNCTION ###
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)
    

### RELU FUNCTION ###
def relu(x):
    return np.maximum(0, x)


def relu_grad(x):
    grad = np.zeros(x)
    grad[x>=0] = 1
    return grad

"""
CATEGORIZATION ALGORITHM
"""    
### SOFTMAX FUNCTION ###
def softmax(x):
    if x.ndim == 2:
        x = x.T # why tranpose for batch?
        x = x - np.max(x, axis=0) # prevent overflow
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T # transpose again?

    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))


def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)

"""
LOSS FUNCTIONS
"""
### CROSS ENTROPY ERROR FUNCTION ###
def cross_entropy_error(y, t):
    if y.ndim == 1: # single image
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # if answer label is one-hot encoding change to numerical label
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    delta = 1e-7

    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


def softmax_loss(X, t):
    y = softmax(X)
    return cross_entropy_error(y, t)
