import sure
import pytest
import numpy as np
from ldl.algorithms import relu_vec
from ldl.algorithms import feed_forward_vec
from ldl.algorithms import relu_vec_differential


def test_relu_vec_returns_correct_calculations():

    # Activation of three observations feeding into a two-neuron layer
    weighted_input = np.array([[-1, 1 ], [2, 0], [5, 5]])

    activations = relu_vec(weighted_input)

    activations.should.have.length_of(3)
    activations[0][0].should.equal(0)
    activations[0][1].should.equal(1)
    activations[1][0].should.equal(2)
    activations[1][1].should.equal(0)
    activations[2][0].should.equal(5)
    activations[2][1].should.equal(5)


def test_relu_vec_handles_various_sizes():
    # 100 x 4 input
    weighted_input = np.ones([100, 4])
    relu_vec.when.called_with(
        weighted_input).should_not.throw(Exception)

    result = relu_vec(weighted_input)
    result.should.have.length_of(100)
    result[0].should.have.length_of(4)

def relu_vec_differential_calculates_correct_values():
    activations = numpy.array([[-1, 0, 1, 2], [0.01, 2.5, 3, 4]])

    diff = relu_vec_differential(activations)

    diff.should.have.length_of(2)
    diff[0][0].should.equal(0)
    diff[0][1].should.equal(0)
    diff[0][2].should.equal(1)
    diff[0][3].should.equal(1)
    diff[1][0].should.equal(1)
    diff[1][1].should.equal(1)
    diff[1][2].should.equal(1)
    diff[1][3].should.equal(1)

def test_feed_forward_vec_does_not_fail_with_valid_parameters():
    # Five layers, with two neurons as output
    layers = [10, 15, 10, 5, 2]

    # Example weights for a five-layer network
    # Each set of weights has dimension L X L+1
    weights = [
        np.ones([10, 15]),
        np.ones([15, 10]),
        np.ones([10, 5]),
        np.ones([5, 2])
    ]

    # One hundred observations, with five variables
    data = np.ones([100, 10])

    result = feed_forward_vec(data=data, weights=weights, neurons=layers,
                          activation_function=relu_vec)

    result.should.be.a(np.ndarray)
    result.should.have.length_of(100)
    result[0].should.have.length_of(2)
