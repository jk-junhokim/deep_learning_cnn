import numpy as np
from gradient_functions.numerical_gradient import function_2, numerical_gradient

### FIND THE MIN VAL POINT ###
def gradient_descent(f, init_x, lr, step_num):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
        
    return x # this is a shallow copy of init_x

"""
The gradient_descent function returns a updated init_x variable.
You need to initialze the init_x variable for each test.
"""

# learning rate = 0.1
init_x = np.array([-3.0, 4.0]) # equal starting point
test1 = gradient_descent(function_2, init_x, 0.1, 100)
print(test1) # closest to (0, 0)

# learning rate = 10.0
init_x = np.array([-3.0, 4.0]) # equal starting point
test2 = gradient_descent(function_2, init_x, 10.0, 100)
print(test2) # diverges

# learning rate = 1e-10
init_x = np.array([-3.0, 4.0]) # equal starting point
test3 = gradient_descent(function_2, init_x, 1e-10, 100)
print(test3) # stops learning before reaching minimum value


