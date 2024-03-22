import pytest

import numpy as np
import numpy.random as npr

from bach import base
from bach import utils

@pytest.mark.parametrize("m, n", [
    (3, 2),
    (1, 1),
    (4, 4)
])
def test_create(m, n):
    try:
        base.DenseLayer(m, n)
    except Exception as e:
        pytest.fail(f'Exception: {e}')

@pytest.mark.parametrize("m, n", [
    (1, 0),
    (0, 1),
    (0, 0)
])
def test_create_empty(m, n):
    with pytest.raises(Exception):
        base.DenseLayer(m, n)
        base.Layer(m, n)

def test_pre_forward():
    L = base.DenseLayer(2, 4)
    X = np.zeros((1, 2))
    try:
        L.forward(X)
    except:
        pytest.fail('Invalid dimensions.')

def test_forward():
    L1 = base.DenseLayer(2, 4)
    L2 = base.DenseLayer(4, 3)
    L3 = base.DenseLayer(3, 1)
    X = npr.randn(100, 2)
    try:
        L3.forward(L2.forward(L1.forward(X)))
    except Exception as e:
        pytest.fail(f'Exception --', e)

@pytest.mark.parametrize("c, m, n", [
    (1, 1, 1),
    (1, 2, 4),
    (1, 4, 2),
    (1, 10, 10)
])
def test_pre_backward(c, m, n):
    L = base.DenseLayer(m, n)

    X = npr.randn(c, m)
    Y = L(X)

    try:
        L.backward(np.zeros((c, n)))
        L.backward(np.ones((c, n)))
    except Exception as e:
        pytest.fail(f'Exception --', e)
