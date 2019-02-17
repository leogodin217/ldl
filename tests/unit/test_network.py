from ldl.network import Network
import numpy as np
import sure
import pytest


def test_network_sets_shape():
    # 10 x 5 x 15 x 10 netowrk
    weights = [
        np.ones([5, 10]),
        np.ones([15, 5]),
        np.ones([10, 15])
    ]

    biases = [
        np.ones(5),
        np.ones(15),
        np.ones(10)
    ]
    network = Network(weights, biases)

    network.shape.should.be.a(list)
    network.shape.should.have.length_of(4)
    network.shape[0].should.equal(10)
    network.shape[1].should.equal(5)
    network.shape[2].should.equal(15)
    network.shape[3].should.equal(10)

