import numpy as np

try:
    from .utils import *
except ImportError:
    from utils import *

class Layer(object):
    def __init__(self, m, n):
        assert m and n
        self.m = m
        self.n = n

    def forward(self, inp): raise NotImplementedError
    def backward(self, out): raise NotImplementedError

class DenseLayer(Layer):
    def __init__(self, m, n, w_init = init_normal, b_init = init_zeros):
        super().__init__(m, n)
        self.weights = w_init(n, m)
        self.biases = b_init(n, 1)
    def forward(self, inp):
        return self.weights @ inp + self.biases
