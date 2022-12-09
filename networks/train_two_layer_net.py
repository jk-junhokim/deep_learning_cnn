import sys, os
import numpy as np
# sys.path.append(os.pardir)
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from mnist_dataset.mnist import load_mnist
from networks.two_layer_net import TwoLayerNet
import matplotlib.pyplot as plt

# Read Data
(x_train, t_train), (x_test, t_test) = load_mnist(normalize = True, one_hot_label = True)
"""
train = 60000
test = 10000
"""

# Initialization (reference to imports above)
"""
Input setting is for only image.
Hidden Layer w/ 50 perceptrons
Output is fixed to 10. One hot encoded.
"""
network = TwoLayerNet(input_size = 784, hidden_size = 50, output_size = 10)
# 784 -> 50 -> 10

# Hyperparameters
iters_num = 10000 # iterations
train_size = x_train.shape[0] # number of training images  = 60000.(image = (1x784))
batch_size = 100
learning_rate = 0.1

# Tracking Data
train_loss_list = [] # should decrease
train_acc_list = [] # should increase
test_acc_list = [] # should increase

# Repeats per Epoch
iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    # Batch Settings
    batch_mask = np.random.choice(train_size, batch_size) # choose 100 random int from 60000
    x_batch = x_train[batch_mask] # select the row number(=image number) from train data
    t_batch = t_train[batch_mask]

    # Gradient Calculations
    # grad = network.multivariable_gradient(x_batch, t_batch) # Numerical Method
    grad = network.gradient(x_batch, t_batch) # Backpropogation Method (higher efficiency)

    # Gradient Descent
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    # Track Training Curve
    loss = network.loss(x_batch, t_batch) # outpu should decrease after every gradient descent
    train_loss_list.append(loss)

    # Accuracy per Epoch (uses updated weight, bias variables)
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_train)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))
        
    
# Draw Graph
# markers = {'train': 'o', 'test': 's'}
# x = np.arange(len(train_acc_list))
# plt.plot(x, train_acc_list, label='train acc')
# plt.plot(x, test_acc_list, label='test acc', linestyle='--')
# plt.xlabel("epochs")
# plt.ylabel("accuracy")
# plt.ylim(0, 1.0)
# plt.legend(loc='lower right')
# plt.show()
