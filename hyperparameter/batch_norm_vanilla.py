import numpy as np

def batchnorm_forward(x, gamma, beta, eps):
    """
    input
    x: initial data. size = (batch_size, 784)
    gamma: scale factor
    beta: shift factor
    eps: elipson (10e-7)
    """
    N, D = x.shape

    # step1: calculate mean
    mu = (1.0/N) * np.sum(x, axis=0)
    """
    N -> size of batch
    axis=0 -> average among all pictures (column addition)
    """

    # step2: subtract mean for every training data
    xmu = x - mu

    # step3: lower branch of computation graph
    sq = xmu ** 2

    # step4: variance
    var = (1.0/N) * np.sum(sq, axis=0)

    # step5: eps for numerical stability
    sqrtvar = np.sqrt(var + eps)

    # step 6: invert
    ivar = (1.0) / sqrtvar

    # step7: normalization
    xhat = xmu * ivar

    # step8: scale
    gammax = gamma * xhat

    # step9: shift
    out = gammax + beta

    # store values for back propagation
    cache = (xhat, gamma, xmu, ivar, sqrtvar, var, eps)

    return out, cache

def batchnorm_backward(dout, cache):
    # unfold variables in cache
    xhat, gamma, xmu, sqrtvar, var, eps = cache

    # get dimensions
    N, D = dout.shape

    # step9: start back prop (going backwards)
    dbeta = np.sum(dout, axis=0)
    dgammax = dout

    # step8
    dgamma = np.sum(dgammax*xhat, axis=0)
    dxhat = dgammax * gamma

    # step7
    divar = np.sum(dxhat*xmu, axis=0)
    dxmu1 = dxhat * ivar

    # step6
    dsqrtvar = -1.0 / (sqrt**r) * divar

    # step5
    dvar = 0.5 * 1.0 / np.sqrt(var+eps) * dsqrtvar

    # step4
    dsq = 1.0 / N * np.ones((N, D)) * dvar

    # step3
    dxmu2 = 2 * xmu * dsq

    # step2
    dx1 = (dxmu1 + dxmu2)
    dmu = -1 * np.sum(dxmu1+dxmu2, axis=0)

    # step1
    dx2 = 1.0 / N * np.ones((N, D)) * dmu

    # step0
    dx = dx1 + dx2

    return dx, dgamma, dbeta