import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.function.functions import cross_entropy_error, sigmoid, relu, sum_squares_error, softmax
from common.gradient.gradients import numerical_gradient

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):

        # weight initialization
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        
        return y

    # x : input data, t : answer label
    def loss(self, x, t):
        y = self.predict(x)
        
        return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis = 1)
        t = np.argmax(t, axis = 1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        
        return accuracy

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W1'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b2'])

        return grads


##### TwoLayerNet TEST #####
"""
net = TwoLayerNet(input_size = 784, hidden_size = 100, output_size = 10) # network initialization

W1_shape = net.params['W1'].shape # (784, 100)
b1_shape = net.params['b1'].shape # (100, )
W2_shape = net.params['W2'].shape # (100, 10)
b2_shape = net.params['b2'].shape # (10, )

print("W1 shape :")
print(W1_shape)
print("b1 shape :")
print(b1_shape)
print("W2 shape :")
print(W2_shape)
print("b2 shape :")
print(b2_shape)

##### PREDICTION TEST #####
x = np.random.rand(100, 784) # input data size = 100
y = net.predict(x)
# print("Prediction Test Results :")
# print(y)

##### GRADIENT TEST #####
x = np.random.rand(100, 784) # input data size = 100
t = np.random.rand(100, 10) # answer label size = 100

# grads = net.numerical_gradient(x, t)

grads['W1'].shape # (784, 100)
grads['b1'].shape # (100, )
grads['W2'].shape # (100, 10)
grads['b2'].shape # (10, )
"""

