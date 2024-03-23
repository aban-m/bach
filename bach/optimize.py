import numpy as np
import numpy.linalg as la


try:
    from .layers import TrainableLayer
except:
    from layers import TrainableLayer

mse = lambda x, y: np.mean((x - y)**2)
mse.d = lambda x, y: (2*(x - y))/len(x)
mse.name = 'mse'

_losses = [mse]

loss_lookup = {}

for loss in _losses:
    loss_lookup[loss.name] = loss

class Callback(object):
    def __init__(self): pass
    def on_back_step(self, i, layer, dJ_dX, dJ_dW, dJ_dB): pass
    def on_iter_start(self, i, layer, dJ_dY): pass

class GradientDescent(Callback):
    def __init__(self, learning_rate = 0.01, decay = 0):
        super().__init__()
        self.a = learning_rate
        self.d = decay
    def on_back_step(self, i, layer, dJ_dX, dJ_dW, dJ_dB):
        if not isinstance(layer, TrainableLayer): return
        m = len(dJ_dB) # questionable
        layer.B = (1 - self.d*self.a)*layer.B - self.a * dJ_dB
        layer.W = (1 - self.d*self.a)*layer.W - self.a * dJ_dW
        
