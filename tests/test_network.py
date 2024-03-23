import pytest

import numpy as np
import numpy.random as npr
from random import sample

from bach.net import Network
from bach.activations import *
from bach.optimize import *

from itertools import product, islice, cycle

from conftest import PICK

c = 10

L = list(product(
            [1, 4],
            [
                [1, 1, 1, 1],
                [4, 3, 2, 1],
                [1, 4, 1, 4]
            ],
            product(act_lookup.keys(), repeat=3)
))
L = sample(L, min(PICK, len(L)))

# preliminary test
def test_creation():
    try:
        Network(1, [(2, 'relu'), (1, 'relu')])
    except Exception as e:
        pytest.fail(str(e))


idfn = lambda param: str(f'input {param[0]} - arch {"/".join(map(str, param[1]))}: {", ".join(param[2])}')
@pytest.fixture(params=L, ids=idfn)
def network(request):
    input_shape, layers, acts = request.param
    return Network(input_shape, zip(layers, acts))

@pytest.fixture
def simple_network():
    return Network(1, [(1, 'relu')])

@pytest.mark.parametrize('loss_name', loss_lookup)
def test_set_loss(loss_name, simple_network):
    try:
        simple_network.set_loss(loss_name)
        assert simple_network.loss, 'Loss not set.'
        assert simple_network.loss_d, 'dLoss not set.'
    except AssertionError as e: raise e
    except Exception as e: pytest.fail(str(e))


def test_chain(network):
    last = network.input_shape
    for layer in network.layers:
        assert layer.m == last, 'Chain broken!'
        last = layer.n
    
def test_forward(network):
    X = npr.random((c, network.input_shape))
    try:
        Y = network.forprop(X)
        assert Y.shape == (c, network.layers[-1].n), 'Incorrect dimension of X'
    except AssertionError as e: raise e
    except Exception as e: pytest.fail('Could not go forward.')
