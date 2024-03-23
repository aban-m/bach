import numpy as np
import numpy.linalg as la

mse = lambda x, y: np.mean((x - y)**2)
mse.d = lambda x, y: (2*(x - y))/x.size
mse.name = 'mse'

_losses = [mse]

loss_lookup = {}

for loss in _losses:
    loss_lookup[loss.name] = loss
