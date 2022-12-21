import numpy as np
import matplotlib.pyplot as plt

### ACTIVATION FUNCTIONS ###
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def ReLU(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)
    
### ACTIVATION LAYER DISTRIBUTION PER WEIGHT INITIALIZATION ###
node_num = 100  # node per hidden layer
weight_initialization_var = [1, 0.01, np.sqrt(1.0 / node_num), np.sqrt(2.0 / node_num)]
activation_func = [sigmoid, ReLU, tanh]
input_data = np.random.randn(1000, 100)
hidden_layer_size = 5

activations = {}
x = input_data

def draw_histogram(activations):
    # draw histogram
    for i, a in activations.items():
        plt.subplot(1, len(activations), i+1)
        plt.title(str(i+1) + "-layer")
        if i != 0: plt.yticks([], [])
        # plt.xlim(0.1, 1)
        # plt.ylim(0, 7000)
        plt.hist(a.flatten(), 30, range=(0,1))
    plt.show()

for i in range(len(weight_initialization_var)):
    for j in range(len(activation_func)):
        activations = {}
        for k in range(hidden_layer_size):
            if k != 0:
                x = activations[k-1]

            w = np.random.randn(node_num, node_num) * weight_initialization_var[i]
            a = np.dot(x, w)
            z = activation_func[j](a)

            activations[k] = z
        
        draw_histogram(activations)
        