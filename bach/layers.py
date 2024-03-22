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

    def forward(self, *inp): pass
    def backward(self, *out): pass

    def __call__(self, *inp): return self.forward(*inp)

class DenseLayer(Layer):
    def __init__(self, m, n, w_init = init_normal, b_init = init_zeros):
        super().__init__(m, n)
        self.W = w_init(m, n)
        self.B = b_init(1, n)

        self.last_X = None
        self.last_Y = None
        
    def forward(self, X):
        super().forward(X)
        Y = X @ self.W + self.B
        
        self.last_X = X
        self.last_Y = Y

        return Y
    
    def backward(self, dJ_dY, X=None):
        if X is None:
            X = self.last_X
        super().backward(dJ_dY, X)
        dJ_dX, dJ_dW, dJ_dB = \
               dJ_dY @ self.W.T,\
               X.T @ dJ_dY,\
               dJ_dY
        return dJ_dX, dJ_dW, dJ_dB
