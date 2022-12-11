import sys, os
import numpy as np
from collections import OrderedDict
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from common.functions import *
from common.layers import *

class MultiLayerNetExtend:
    """
    input_size : 784
    hidden_size_list : e.g. [100, 100, 100]
    output_size : 10
    activation : 'relu' / 'sigmoid'
    weight_init_std : e.g. 0.01, He, Xavier
    weight_decay_lambda : control weight (prevent overfitting)
    use_dropout : true / false
    dropout_ration : dropout ration
    use_batchNorm : true / false
    """
    def __init__(self, input_size, hidden_size_list, output_size,
                 activation='relu', weight_init_std='relu', weight_decay_lambda=0, 
                 use_dropout = False, dropout_ration = 0.5, use_batchnorm=False):

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_list = hidden_size_list
        self.hidden_layer_num = len(hidden_size_list)
        self.use_dropout = use_dropout
        self.weight_decay_lambda = weight_decay_lambda
        self.use_batchnorm = use_batchnorm
        self.params = {}

        # Weight Initialization (Efficient Method. Not Random)
        self.__init_weight(weight_init_std)

        # Create Multi-Layers
        activation_layer = {'sigmoid': Sigmoid, 'relu': Relu}
        self.layers = OrderedDict()
        

    def __init_weight(self, weight_init_std):
        all_size_list = [self.input_size] + self.hidden_size_list + [self.output_size]
        for idx in range(1, len(all_size_list)):
            scale = weight_init_std
            if str(weight_init_std).lower() in ('relu', 'he'):
                scale = np.sqrt(2.0 / all_size_list[idx-1])
            elif str(weight_init_std).lower() in ('sigmoid', 'xavier'):
                scale = np.sqrt(1.0 / all_size_list[idx-1])
            self.params['W' + str(idx)] = scale * np.random.randn(all_size_list[idx-1], all_size_list[idx])
            self.params['b' + str(idx)] = np.zeros(all_size_list[idx])

