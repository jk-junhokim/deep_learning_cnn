import sys, os
sys.path.append(os.pardir)
import numpy as np
from activation_functions.activation_functions import softmax
from loss_functions.cross_entropy_error import cross_entropy_error
from gradient_functions.numerical_gradient import numerical_gradient

"""
Create an array of the given shape and
populate it with random samples
from a uniform distribution over [0, 1)
"""

class simpleNet:
    def __init__(self):
        self.W = np.random.rand(2, 3)

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss

net = simpleNet()
print(net.W) # initialized weights
x = np.array([0.6, 0.9])
p = net.predict(x)
print(p)
print(np.argmax(p))
t = np.array([0, 0, 1]) # answer label
net.loss(x, t)

def f(W):
    return net.loss(x, t)

dW = numerical_gradient(f, net.W)
print(dW)

