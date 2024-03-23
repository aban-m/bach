import numpy as np

try:
    from .layers import *
    from .activations import *
    from .optimize import *
except ImportError:
    from layers import *
    from activations import *
    from optimize import *


class Network(object):
    def __init__(self, input_shape, topology=[], loss=None, optimizer=None):
        self.layers = []
        self.input_shape = input_shape
        self.L = len(self.layers)
        self.loss = None
        self.loss_d = None

        if optimizer is not None and not isinstance(optimizer, Callback):
            raise TypeError('optimizer must inherit from Callback class')
        self.optimizer = optimizer
        if loss is not None: self.set_loss(loss)

        m = self.input_shape
        for elem in topology:
            n = elem; act = None
            try:
                n, act = elem
            except:
                pass
            self.add(DenseLayer(m, n))
            if act:
                obj = None; prime = None
                if type(act) == str:
                    try: obj = act_lookup[act]
                    except KeyError: raise ValueError(f'Unrecognized activation -- {act}.')
                elif not hasattr(act, 'd'):
                    obj, prime = act
                self.add(ActivationLayer(n, obj, prime))
            m = n
            
    def set_loss(self, loss):
        d = None
        if type(loss) == str:
            try: loss = loss_lookup[loss]
            except KeyError: raise ValueError(f'Unrecognized loss -- {loss}')
        if not hasattr(loss, 'd'):
            loss, d = loss
        else:
            d = loss.d
        self.loss, self.loss_d = loss, d
        
    def add(self, layer):
        self.layers.append(layer)
        self.L += 1

    def forprop(self, X):
        last = X
        for i, layer in enumerate(self.layers):
            last = layer(last)
        return last

    def backprop(self, X, y, callbacks=[]):
        assert all(isinstance(callback, Callback) for callback in callbacks)
        ypred = self.forprop(X)
        J = self.loss(y, ypred)

        # keeping the last error
        dJ_dY = self.loss_d(y, ypred)
        
        for i, layer in enumerate(reversed(self.layers)):
            for callback in callbacks:
                callback.on_iter_start(self.L - i - 1, layer, dJ_dY)
            output = layer.backward(dJ_dY)
            dJ_dW = None; dJ_dB = None
            if isinstance(layer, TrainableLayer):
                dJ_dY, dJ_dW, dJ_dB = output
            else:
                dJ_dY = output
            for callback in callbacks:
                callback.on_back_step(self.L - i - 1, layer, dJ_dY, dJ_dW, dJ_dB)

        return dJ_dY

    def fit(self, X, y):
        if self.optimizer is None:
            raise Exception('Optimizer is not set.')
        if self.loss is None:
            raise Exception('Loss is not set.')
        self.backprop(X, y, callbacks=[self.optimizer])
