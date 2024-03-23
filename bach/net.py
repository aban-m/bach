import numpy as np

try:
    from .layers import *
    from .activations import *
    from .losses import *
except ImportError:
    from layers import *
    from activations import *
    from losses import *

class Network(object):
    def __init__(self, input_shape, topology=[], loss=None):
        self.layers = []
        self.input_shape = input_shape
        self.L = len(self.layers)
        self.loss = None
        self.loss_d = None
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

    def predict(self, X):
        last = X
        for i, layer in enumerate(self.layers):
            last = layer(last)
        return last

    
