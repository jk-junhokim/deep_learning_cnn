import numpy as np


### STEP FUNCTION ###
def step_function(x):
    return np.array(x > 0, dtype = np.int)

    
### SIGMOID FUNCTION ###
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

    
### RELU FUNCTION ###
def relu(x):
    return np.maximum(0, x)

    
### SOFTMAX FUNCTION ###
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c) # prevent overflow
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y


# softmax example
"""
a = np.array([0.3, 2.9, 4.0])
y = softmax(a)
print(y)
print(np.sum(y))
"""

