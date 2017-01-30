import numpy as np
import math

# turn a vector into a diagonal matrix
def diag(v):
    ret = np.zeros((len(v), len(v)))
    for i in range(len(v)):
        ret[i][i] = v[i]
    return ret

# learning schedule
def eta(t, tau_0 = 100, kappa = 0.75):
    return (tau_0 + t)**(-1 * kappa)

# for weight initialization
def random_weights_gaussian(shape):
    return np.random.normal(0, 1. / math.sqrt(shape[0]), shape)