import numpy as np

class Layer:
    def __init__(self, weights, biases, act_func, is_input = False, is_output = False):
        self.is_input = is_input
        self.is_output = is_output
        self.w = weights
        self.b = biases
        self.a = act_func
        self.num_neurons = self.w.shape[0]
        
    def __call__(self, x):
        zs = np.dot(self.w, x) + self.b
        return zs, self.a(zs)