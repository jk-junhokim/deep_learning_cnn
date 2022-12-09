import sys, os
import numpy as np
from collections import OrderedDict
# sys.path.append(os.pardir)
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from common.functions import *
from common.gradients import numerical_gradient
from common.layers import *

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):

        # weight initialization
        # generates random normal (gaussian) distribution
        # mean = 0, std = 1
        # but why use gaussian? not uniform?
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        # create layers
        self.layers = OrderedDict() # order of dictionary does not change (imported)
        self.layers["Affine1"] = Affine(self.params['W1'], self.params['b1']) # layer function
        self.layers['Relu1'] = Relu() # activation function (only one hidden layer)
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])

        # separate layer (not in self.layers). need to call indepently
        self.lastLayer = SoftmaxWithLoss() # categorization function
        
    def predict(self, x): # returns a numerical value

        # The backward propagation version
        for layer in self.layers.values():
            x = layer.forward(x)
        
        return x


        """
        # The multivariable gradient version
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2) # softmax is probability for each outcome

        return y
        """

    # x : input data, t : answer label
    def loss(self, x, t):
        y = self.predict(x)
        # uses the numerical prediction value of x
        # which went through the Affine layer

        # use the outcome (y) to compare with the actual answer label (t)
        # used for backpropagation method
        return self.lastLayer.forward(y, t)

        # this return statement is for multivariable gradient descent method
        # return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis = 1)
        if t.ndim != 1:
            t = np.argmax(t, axis = 1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        
        return accuracy

    # the numerical(calculus) method
    def multivariable_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        # loss_w is a lambda function object 
        # which is the CEE function at inputs (x, t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1']) # numerical_gradient function is imported
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads

    # the backpropagation method
    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # save gradient values
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db

        return grads
