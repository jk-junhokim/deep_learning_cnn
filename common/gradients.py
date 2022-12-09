import numpy as np

def function_2(x):
    return x[0]**2 + x[1]**2    

### NUMERICAL GRADIENT FOR ONE DIMENSIONAL MATRIX ###
def numerical_gradient_1d(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x) # intialize gradient values

    for idx in range(x.size):
        tmp_val = x[idx]
        # f(x+h)
        x[idx] = tmp_val + h
        fxh1 = f(x)

        # f(x-h)
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val

    return grad

### NUMERICAL GRADIENT FOR MULTI DIMENSIONAL MATRICES ###
def numerical_gradient(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val # original value
        it.iternext()   
        
    return grad

"""
test1 = numerical_gradient(function_2, np.array([3.0, 4.0]))
test2 = numerical_gradient(function_2, np.array([0.0, 2.0]))
test3 = numerical_gradient(function_2, np.array([3.0, 0.0]))
print(test1)
print(test2)
print(test3)
"""

    
### GRADIENT DESCENT TO FIND THE MIN VAL POINT ###
def gradient_descent(f, init_x, lr, step_num):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
        
    return x # this is a shallow copy of init_x

"""
##### The gradient_descent function returns a updated init_x variable.
You need to initialze the init_x variable for each test. #####

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
"""
