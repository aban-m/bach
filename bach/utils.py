import numpy.random as npr
import numpy as np

def init_zeros(*shape):
    return np.zeros(shape)
def init_unif(*shape):
    return npr.random(shape) - 0.05
def init_normal(*shape):
    return npr.randn(*shape)
