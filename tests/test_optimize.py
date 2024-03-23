import pytest
import numpy.random as npr
import numpy as np

from bach.optimize import *
from bach.layers import DenseLayer

c, m, n = 20, 3, 4

@pytest.fixture(params=[(0.01, 0), (0.01, 0.1)])
def optimizer(request):
    a, d = request.param
    return GradientDescent(learning_rate = a, decay = d)

@pytest.fixture
def callback_params():
    i = 3 # arbitrary, does not matter
    layer = DenseLayer(m, n)
    X = npr.randn(c, m)
    dJ_dY = npr.randn(c, n)
    dJ_dX = npr.randn(c, m)
    dJ_dW = npr.randn(m, n)
    dJ_dB = npr.randn(1, n)
    return (i, layer, dJ_dX, dJ_dW, dJ_dB)

@pytest.mark.parametrize('loss', loss_lookup.values())
class TestLoss:
    def test_pre(_, loss):
        X = npr.randn(m, n)
        Y = X
        result = loss(X, Y)
        assert not result, 'Distance must be suicidal!'
        assert isinstance(result, np.number), 'Loss must be a scalar.'
        result = loss(X, X**2)
        assert result, 'Distance must not murder!'

    def test_deriv(_, loss):
        X = npr.randn(m, n)
        result = loss.d(X, X**2)
        assert result.shape == X.shape, 'Shape must be preserved under the derivative.'


def test_optimizer_pre(optimizer, callback_params):
    try:
        layer = callback_params[1]
        W, B = layer.W.copy(), layer.B.copy()
        optimizer.on_back_step(*callback_params)
        assert not (W == layer.W).all() and not (B == layer.B).all()
    except Exception as e:
        pytest.fail(f'Callback failed: {e}')
