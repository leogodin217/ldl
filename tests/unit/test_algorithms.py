import sure
import pytest
import numpy as np
from ldl.algorithms import relu_vec
from ldl.algorithms import feed_forward_vec
from ldl.algorithms import relu_differential_vec
from ldl.algorithms import quadradic_cost_vec
from ldl.algorithms import quadradic_cost_derivative_vec
from ldl.algorithms import output_error_vec
from ldl.algorithms import layer_error_vec


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

def relu_differential_vec_calculates_correct_values():
    activations = numpy.array([[-1, 0, 1, 2], [0.01, 2.5, 3, 4]])

    diff = relu_differential_vec(activations)

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
        np.ones([11, 15]),
        np.ones([16, 10]),
        np.ones([11, 5]),
        np.ones([6, 2])
    ]

    # One hundred observations, with five variables
    data = np.ones([100, 10])

    result = feed_forward_vec(data=data, weights=weights, neurons=layers,
                          activation_function=relu_vec)

    result.should.be.a(np.ndarray)
    result.should.have.length_of(100)
    result[0].should.have.length_of(2)


def test_quadradic_cost_vec_returns_correct_value():
    y = np.array([[0, 0, 0, 0], [0, 2, 0, 0]])
    # Should result in 2 and 0, repsectively for cost, which would result
    # in .25 for average cost.
    y_predicted = np.array([[1, 1, -1, -1], [0, 2, 0, 0]])

    result = quadradic_cost_vec(y, y_predicted)
    result.should.equal(1.0)


def test_quadradic_cost_derivative_returns_correct_values():
    y = np.array([[0, 0, 0, 0], [1, 2, 3, 4]])
    y_predicted = np.array([[1, 0, .5, 0], [0, 0, 0, 0]])

    derivative = quadradic_cost_derivative_vec(y, y_predicted)

    derivative.should.have.length_of(2)
    derivative[0][0].should.equal(-1)
    derivative[0][1].should.equal(0)
    derivative[0][2].should.equal(-.5)
    derivative[0][3].should.equal(0)
    derivative[1][0].should.equal(1)
    derivative[1][1].should.equal(2)
    derivative[1][2].should.equal(3)
    derivative[1][3].should.equal(4)


def test_output_activation_vec_works_with_valid_parameters():
    # two observations for four output neurons
    y = np.array([[1, 1, 1, 1], [2, 2, 2, 2]])
    y_predicted = np.array([[0, 1, 1, 1], [0, 2, 2, 2]])
    weighted_input = np.array([[1, 1, 1, 1], [1, 1, 1, 1]])
    cost_derivative_function = quadradic_cost_derivative_vec
    activation_derivative_function = relu_differential_vec

    errors = output_error_vec(y, y_predicted, cost_derivative_function,
                              activation_derivative_function, weighted_input)

    errors.should.be.a(np.ndarray)
    errors.should.have.length_of(2)
    errors[0].should.have.length_of(4)
    errors[0][0].should.equal(1)
    errors[1][0].should.equal(2)
    errors[0][1].should.equal(0)
    errors[0][2].should.equal(0)
    errors[0][3].should.equal(0)
    errors[1][1].should.equal(0)
    errors[1][2].should.equal(0)
    errors[1][3].should.equal(0)


def test_layer_error_vec_works_with_valid_parameters():
    # 3 neurons in l going into 2 neurons in l+1
    weights = np.array([[1, 1, 1], [1, 1, 1]])
    # 5 observations of 3 neurons in l+1
    errors = np.array([[1, 1], [0, 0], [1, 1], [1, 1], [1, 1]])
    # 5 observations of 3 neurons in l
    derivatives = np.array(
        [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [0, 0, 0]])

    errors = layer_error_vec(weights, errors, derivatives)
    errors.should.be.a(np.ndarray)
    errors.should.have.length_of(5)
    errors[0].should.have.length_of(3)
    errors[0][0].should.equal(2)
    errors[0][1].should.equal(2)
    errors[0][2].should.equal(2)
    errors[1][0].should.equal(0)
    errors[1][1].should.equal(0)
    errors[1][2].should.equal(0)
    errors[4][0].should.equal(0)
    errors[4][1].should.equal(0)
    errors[4][2].should.equal(0)
