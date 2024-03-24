import numpy as np
import numpy.random as npr

import pytest
from conftest import PATIENCE
from logging import getLogger as logger
PATIENCE = min(PATIENCE, 100)/100

from bach.net import Network
from bach.optimize import GradientDescent

@pytest.fixture
def adder_network():
    net_add = Network(2, [(1, 'linear')], loss='mse')
    net_add.optimizer = GradientDescent(0.01)
    return net_add

@pytest.fixture
def sin_network():
    net_sin = Network(1, [(10, 'relu'), (10, 'relu'), (1, 'linear')], loss='mse')
    net_sin.optimizer = GradientDescent(0.05)
    return net_sin

@pytest.fixture
def xor_network():
    net_xor = Network(2, [(3, 'sigmoid'), (1, 'relu')], loss='mse')
    net_xor.optimizer = GradientDescent(0.15)
    return net_xor

@pytest.mark.slow
def test_add(adder_network, caplog):
    c = int(PATIENCE*500)
    X = 10*(npr.random((c, 2)) - 0.5)
    y = X.sum(axis=1, keepdims=True)
    for i in range(int(c*100)):
        adder_network.fit(X, y)
    err = adder_network.loss(adder_network.forprop(X), y)
    logger().info(f'[ADD] error: {err}')
    assert err < 0.1, f'Bad network! Got error {err}'

@pytest.mark.slow
def test_sin(sin_network):
    c = int(PATIENCE*100)
    X = np.linspace(-2, 2, c).reshape(-1, 1)
    y = np.sin(X)
    for i in range(int(c*100)):
        sin_network.fit(X, y)
    err = sin_network.loss(sin_network.forprop(X), y)
    logger().info(f'[SIN] error: {err}')
    assert err < 0.1, f'Bad network! Got error {err}'

@pytest.mark.slow
def test_xor(xor_network):
    f = int(PATIENCE*10000)
    X = np.array([
        [0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0]).reshape(-1, 1)
    for i in range(f):
        xor_network.fit(X, y)
    err = xor_network.loss(xor_network.forprop(X), y)
    logger().info(f'[XOR] error: {err}')
    assert err < 0.1, f'Bad network! Got error {err}'
