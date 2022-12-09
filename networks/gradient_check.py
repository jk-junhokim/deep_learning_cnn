import sys, os
sys.path.append(os.pardir)
import numpy as np
from mnist_dataset.mnist import load_mnist
from networks.two_layer_net import *

# read data
(x_train, t_train), (x_test, t_test) = load_mnist(normalize = True, one_hot_label = True)

network = TwoLayerNet(input_size = 784, hidden_size = 50, output_size = 10)

x_batch = x_train[:3]
t_batch = t_train[:3]

grad_multivariable = network.multivariable_gradient(x_batch, t_batch)
grad_backpropagation = network.gradient(x_batch, t_batch)

# difference between grad_numerical method & grad_backpropagation method
for key in grad_multivariable.keys():
    diff = np.average(np.abs(grad_backpropagation[key] - grad_multivariable[key]))
    print(key + ":" + str(diff))
    