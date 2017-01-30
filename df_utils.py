import numpy as np
import math
from train_utils import diag

class Differentiable:
    def __init__(self, func, derivative):
        self.func = func
        self.derivative = derivative
    
    def __call__(self, x, y= None):
        if y != None: 
            return self.func(x,y)
        else:
            return self.func(x)

# activation functions
def relu_function(x):
    return np.array(map(lambda xi: max(0, xi), x))

def relu_derivative(x):
    return diag(np.array(map(lambda xi: 1 if xi > 0 else 0, x)))

def identity_function(x):
    return x

def identity_derivative(x):
    return diag(np.array([1] * len(x)))

def sigmoid_function(x):
    return np.array(map(lambda xi: 1 / (1 + math.exp(-xi)), x))

def sigmoid_derivative(x):
    s = sigmoid_function(x)
    return diag(np.array(map(lambda sx: sx * (1 - sx), s)))

def softmax_function(x):
    ret = np.exp(x)
    normalize = 1. / sum(ret)
    ret = ret * normalize
    return ret

def softmax_derivative(x):
    p = softmax_function(x)
    deriv = np.empty((len(x), len(x)))
    for i in range(len(x)):
        for j in range(len(x)):
            if i == j:
                deriv[i][j] = p[i] * (1. - p[i])
            else:
                deriv[i][j] = -1. * p[i] * p[j]
    return np.array(deriv)

relu = Differentiable(relu_function, relu_derivative)
identity = Differentiable(identity_function, identity_derivative)
sigmoid = Differentiable(sigmoid_function, sigmoid_derivative)
softmax = Differentiable(softmax_function, softmax_derivative)

# loss functions
def cross_entropy_loss_function(g_x, y):
    return -1. * np.dot(y, map(lambda x: math.log(x), g_x))

def cross_entropy_loss_derivative(g_x, y):
    return -1 * np.multiply(y, map(lambda x: 1. / x, g_x))

def squared_loss_function(g_x, y):
    return 0.5 * (y - g_x)**2

def squared_loss_derivative(g_x, y):
    return g_x - y

cross_entropy_loss = Differentiable(cross_entropy_loss_function, cross_entropy_loss_derivative)
squared_loss = Differentiable(squared_loss_function, squared_loss_derivative)