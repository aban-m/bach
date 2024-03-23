import pytest
import numpy.random as npr
import numpy as np

from bach.losses import *

c1, c2 = 3, 4



pytestmark = pytest.mark.parametrize('loss', loss_lookup.values())

def test_pre(loss):
    X = npr.randn(c1, c2)
    Y = X
    result = loss(X, Y)
    assert not result, 'Distance must be suicidal!'
    assert isinstance(result, np.number), 'Loss must be a scalar.'
    result = loss(X, X**2)
    assert result, 'Distance must not murder!'

def test_deriv(loss):
    X = npr.randn(c1, c2)
    result = loss.d(X, X**2)
    assert result.shape == X.shape, 'Shape must be preserved under the derivative.'
