import numpy as np
import numpy.random as npr

try:
    from .optimize import mse
except ImportError:
    from optimize import mse
    
EPSILON = 0.01
def atomic_grad(net, X, y, layer, i, j=None):
    layer = net.layers[layer] if type(layer) == int else layer
    what = layer.W if j is not None else layer.B
    i, j = (0, i) if j is None else (i, j)
    
    what[i, j] += EPSILON
    yplus = mse(net.forprop(X), y)
    what[i, j] -= 2*EPSILON
    yminus = mse(net.forprop(X), y)
    what[i, j] += EPSILON
    
    return (yplus - yminus)/(2*EPSILON)

def layer_grad(net, X, y, layer):
    layer = net.layers[layer] if type(layer) == int else layer
    
    dJ_dW = np.zeros_like(layer.W)
    dJ_dB = np.zeros_like(layer.B)

    for j in range(layer.n):
        dJ_dB[0, j] = atomic_grad(net, X, y, layer, j).mean()
        for i in range(layer.m):
            dJ_dW[i, j] = atomic_grad(net, X, y, layer, i, j)

    return dJ_dW, dJ_dB
