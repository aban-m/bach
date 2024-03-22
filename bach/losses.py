import numpy as np
import numpy.linalg as la

mse = lambda x, y: la.norm(x - y)
mse.d = lambda x, y: (2*(x - y)).T

loss_lookup = {
    'mse': mse
}