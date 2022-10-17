import numpy as np
from gradient_functions.numerical_gradient import function_2, numerical_gradient

def gradient_descent(f, init_x, lr, step_num):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
        
    return x

init_x = np.array([-3.0, 4.0])
test1 = gradient_descent(function_2, init_x = init_x, lr = 0.1, step_num = 100)
test2 = gradient_descent(function_2, init_x = init_x, lr = 10.0, step_num = 100)
test3 = gradient_descent(function_2, init_x = init_x, lr = 1e-10, step_num = 100)
print(test1)
print(test2)
print(test3)

