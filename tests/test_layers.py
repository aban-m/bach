import pytest

import numpy as np
import numpy.random as npr
from random import choice

from bach import layers
from bach.net import Network
from bach.activations import *
from bach.losses import *

pytestmark = pytest.mark.parametrize('m', [1, 2, 5])
c = 4
L = [4, 2, 5, 8, 2, 1]

@pytest.mark.parametrize('layer', [RELU, SIGMOID, LINEAR])
class TestActForward:
    def test_creation(_, layer, m):
        try: layer = layer(m)
        except Exception as e: pytest.fail(str(e))
        
    def test_forward(_, layer, m):
        layer = layer(m)
        x = np.zeros((c, m))
        y = np.ones((c, m))
        assert layer(x).shape[1] == m
        assert layer(y).shape[1] == m
         
    #@pytest.mark.xfail
    @pytest.mark.parametrize('layer2', [RELU, SIGMOID, LINEAR])
    def test_stack(_, layer, layer2, m):
        x1 = np.zeros((c, m))
        x2 = np.ones((c, m))
        layer = layer(m)
        layer2 = layer2(m)
        assert layer2(layer(x1)).shape[1] == m
        assert layer2(layer(x2)).shape[1] == m
        
class TestActBackward:
    def test_back_relu(_, m):
        n=m
        relu = RELU(m)
        X = -npr.random((c, m))
        dJ_dY = np.ones((c, n))
        relu(X)
        dJ_dX = relu.backward(dJ_dY)
        assert dJ_dX.shape == X.shape, 'Incorrect shape.'
        assert not np.any(dJ_dX), 'Derivatives must all vanish'

@pytest.mark.parametrize('n', [3, 4, 6])
class TestDenseForward:
    def test_creation(_, m, n):
        try:
            layer = layers.DenseLayer(m, n)
            assert layer.W.shape == (m, n), 'Weight shape incorrect.'
            assert layer.B.shape == (1, n), 'Bias shape incorrect.'
        except AssertionError as e: raise e
        except Exception as e:
            pytest.fail(str(e))
    def test_single(_, m, n):
        layer = layers.DenseLayer(m, n)
        X = np.zeros((c, m))
        Y = layer(X)
        assert Y.shape == (c, n), 'Output shape incorrect'
        
    @pytest.mark.parametrize('k', [1, 2, 3])
    def test_stack(_, m, k, n):
        layer1 = layers.DenseLayer(m, k)
        layer2 = layers.DenseLayer(k, n)
        X = np.zeros((c, m))
        Y = layer2(layer1(X))
        assert Y.shape == (c, n)

@pytest.mark.parametrize('n', [3, 4, 6])
class TestDenseBackward:
    def test_dense_back_1(_, m, n):
        layer = layers.DenseLayer(m, n)
        dJ_dY = npr.randn(c, n)
        X = npr.randn(c, m)

        Y = layer(X)
        dJ_dX, dJ_dW, dJ_dB = layer.backward(dJ_dY)
        
        assert dJ_dW.shape == layer.W.shape, 'dJ_dW failed.'
        assert dJ_dB.shape == layer.B.shape, 'dJ_dB failed.'
        assert dJ_dX.shape == X.shape, 'dJ_dX failed.'

    def test_dense_back_2(_, m, n):
        layer = layers.DenseLayer(m, n)
        dJ_dY = np.zeros((c, n))
        X = npr.randn(c, m)

        Y = layer(X)
        dJ_dX, dJ_dW, dJ_dB = layer.backward(dJ_dY)
        
        assert dJ_dW.shape == layer.W.shape, 'dJ_dW failed.'
        assert dJ_dB.shape == layer.B.shape, 'dJ_dB failed.'
        assert dJ_dX.shape == X.shape, 'dJ_dX failed.'
        assert not np.any(dJ_dW) and not np.any(dJ_dB), 'Derivative must vanish.'

