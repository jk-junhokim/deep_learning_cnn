import numpy as np

def function_2(x):
    return x[0]**2 + x[1]**2    

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

# test1 = numerical_gradient(function_2, np.array([3.0, 4.0]))
# test2 = numerical_gradient(function_2, np.array([0.0, 2.0]))
# test3 = numerical_gradient(function_2, np.array([3.0, 0.0]))
# print(test1)
# print(test2)
# print(test3)

