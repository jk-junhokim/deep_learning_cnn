import numpy as np

def function_2(x):
    return x[0]**2 + x[1]**2    

def numerical_gradient(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)

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

test1 = numerical_gradient(function_2, np.array([3.0, 4.0]))
test2 = numerical_gradient(function_2, np.array([0.0, 2.0]))
test3 = numerical_gradient(function_2, np.array([3.0, 0.0]))
# print(test1)
# print(test2)
# print(test3)

