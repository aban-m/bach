import numpy as np

try:
    from .layers import ActivationLayer
except ImportError:
    from layers import ActivationLayer

_relu = np.vectorize(lambda x: max(x, 0))
_sigmoid = np.vectorize(lambda x: 1/(1 + np.exp(-x)))
_linear = np.vectorize(lambda x: x)

_linear.d = np.vectorize(lambda x, y=None: 1)
_relu.d = np.vectorize(lambda x, y=None: int(x >= 0))
@np.vectorize
def _dsigmoid(x, y=None):
    if y is not None:
        return y*(1 - y)
    y = _sigmoid(x)
    return y*(1 - y)
_sigmoid.d = _dsigmoid

RELU = lambda m: ActivationLayer(m, _relu)
SIGMOID = lambda m: ActivationLayer(m, _sigmoid)
LINEAR = lambda m: ActivationLayer(m, _linear)

act_lookup = {
    'relu': _relu,
    'sigmoid': _sigmoid,
    'linear': _linear
}
