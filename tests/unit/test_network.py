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


def test_get_error_rate_calculates_according_to_predict_function():
    # Four observations with predictions of 2, 2, 1, 1
    predictions = np.array([[0, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0]])
    # Four targets with 50% match to predicted
    targets = np.array([1, 1, 1, 1])
    # 10 x 3 x 5 network (Shape is not important for this test. Just has to be
    # valid)
    weights = [
        np.ones([3, 10]),
        np.ones([5, 3])
    ]
    biases = [
        np.ones([3]),
        np.ones([5])
    ]
    network = Network(weights, biases)
    error_rate = network.get_error_rate(predictions, targets, np.argmax)
    error_rate.should.equal(.50)
