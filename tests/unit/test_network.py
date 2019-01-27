from ldl.network import Network
import sure
import pytest


def test_network_requires_shape():
    expected_message = "__init__() missing 1 required positional argument: 'shape'"
    Network.when.called_with().should.throw(TypeError, expected_message)


def test_network_sets_shape():
    shape = [10, 5, 5]
    n = Network(shape)
    n.shape.should.equal(shape)


def test_network_initializes_layers_with_bias():
    shape = [10, 5, 5]
    n = Network(shape)
    n.layers.should.equal([11, 6, 5])


