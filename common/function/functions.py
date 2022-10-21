import numpy as np

"""
ACTIVATION FUNCTIONS
"""

### STEP FUNCTION ###
def step_function(x):
    return np.array(x > 0, dtype = np.int)

    
### SIGMOID FUNCTION ###
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

    
### RELU FUNCTION ###
def relu(x):
    return np.maximum(0, x)

"""
CATEGORIZATION ALGORITHM
"""    
### SOFTMAX FUNCTION ###
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c) # prevent overflow
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y

"""
LOSS FUNCTIONS
"""
### CROSS ENTROPY ERROR FUNCTION ###
def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))


### SUM OF SQUARES FOR ERROR FUNCTION ###
def sum_squares_error(y, t):
    return 0.5 * np.sum((y-t)**2)


"""
TESTING
"""
    
# softmax test
"""
a = np.array([0.3, 2.9, 4.0])
y = softmax(a)
print(y)
print(np.sum(y))
"""

# cee test
"""
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
test1 = cross_entropy_error(np.array(y), np.array(t))
print(test1)

y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
test2 = cross_entropy_error(np.array(y), np.array(t))
print(test2)
"""

# sse test
"""
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
test1 = sum_squares_error(np.array(y), np.array(t))
print(test1)

y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
test2 = sum_squares_error(np.array(y), np.array(t))
print(test2)
"""

