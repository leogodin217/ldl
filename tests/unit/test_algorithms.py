import sure
import pytest
import numpy as np
from ldl.algorithms import relu_vec
from ldl.algorithms import feed_forward_vec
from ldl.algorithms import relu_derivative_vec
from ldl.algorithms import quadradic_cost_vec
from ldl.algorithms import quadradic_cost_derivative_vec
from ldl.algorithms import output_error_vec
from ldl.algorithms import layer_error_vec
from ldl.algorithms import backpropagate_errors
from ldl.algorithms import get_bias_partial_derivatives_vec
from ldl.algorithms import get_weight_partial_derivatives_vec
from ldl.algorithms import get_updated_biases_vec
from ldl.algorithms import get_updated_weights_vec


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

def relu_derivative_vec_calculates_correct_values():
    activations = numpy.array([[-1.5, 0, 1.2, 2], [0.01, 2.5, 3, 4]])

    diff = relu_derivative_vec(activations)

    diff.should.have.length_of(2)
    diff[0][0].should.be.a(double)
    diff[0][0].should.equal(0)
    diff[0][1].should.equal(0)
    diff[0][2].should.equal(1)
    diff[0][3].should.equal(1)
    diff[1][0].should.equal(1)
    diff[1][1].should.equal(1)
    diff[1][2].should.equal(1)
    diff[1][3].should.equal(1)


def test_feed_forward_vec_does_not_fail_with_valid_parameters():
    # 10 x 15 x 10 x 5 x 2 network
    # Example weights for a five-layer network
    # Each set of weights has dimension L X L+1
    weights = [
        np.ones([15, 10]),
        np.ones([10, 15]),
        np.ones([5, 10]),
        np.ones([2, 5])
    ]

    biases = [
        np.ones([15]),
        np.ones([10]),
        np.ones([5]),
        np.ones([2])
    ]

    # One hundred observations, with five variables
    data = np.ones([100, 10])

    result = feed_forward_vec(data, weights, biases,
                              activation_function=relu_vec,
                              derivative_function=relu_derivative_vec)

    result.should.be.a(dict)
    result['activations'].should.be.a(list)
    result['activations'].should.have.length_of(5)
    result['activations'][0].should.have.length_of(100)
    result['activations'][0][0].should.have.length_of(10)
    result['activations'][1].should.have.length_of(100)
    result['activations'][1][0].should.have.length_of(15)
    result['activations'][2].should.have.length_of(100)
    result['activations'][2][0].should.have.length_of(10)
    result['activations'][3].should.have.length_of(100)
    result['activations'][3][0].should.have.length_of(5)
    result['activations'][4].should.have.length_of(100)
    result['activations'][4][0].should.have.length_of(2)
    result['derivatives'].should.be.a(list)
    # No derivative for the first layer
    result['derivatives'].should.have.length_of(4)
    result['derivatives'][0].should.have.length_of(100)
    result['derivatives'][0][0].should.have.length_of(15)
    result['derivatives'][1].should.have.length_of(100)
    result['derivatives'][1][0].should.have.length_of(10)
    result['derivatives'][2].should.have.length_of(100)
    result['derivatives'][2][0].should.have.length_of(5)
    result['derivatives'][3].should.have.length_of(100)
    result['derivatives'][3][0].should.have.length_of(2)


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
    activation_derivative_function = relu_derivative_vec

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


def test_backpropagate_errors_works_with_valid_parameters():
    # 10 x 5 x 3 x 2 network. We already have error for output and we don't
    # need error for input layer
    weights = [
        np.ones([3, 5]),
        np.ones([2, 3]),
    ]

    derivatives = [
        np.ones([20, 5]),
        np.ones([20, 3]),
    ]

    output_errors = np.ones([20, 2])

    errors = backpropagate_errors(weights, derivatives, output_errors)

    errors.should.be.a(list)
    errors.should.have.length_of(3)
    errors[0].should.be.a(np.ndarray)
    errors[0].shape.should.equal((20, 5))
    errors[1].shape.should.equal((20, 3))
    errors[2].shape.should.equal((20, 2))


def test_get_bias_partial_derivatives_returns_mean_of_the_error():
    # 10 x 5 x 3 x 2 network. The bias is calculated on all but the
    # input layer.
    errors = [
        np.ones([20, 5]),
        np.ones([20, 3]),
        np.ones([20, 2])
    ]

    bias_partial_derivatives = get_bias_partial_derivatives_vec(errors)

    bias_partial_derivatives.should.have.length_of(3)
    bias_partial_derivatives[0].should.be.a(np.ndarray)
    bias_partial_derivatives[0].shape.should.equal((5, ))
    bias_partial_derivatives[0][0].should.equal(1)
    bias_partial_derivatives[0][1].should.equal(1)
    bias_partial_derivatives[0][2].should.equal(1)
    bias_partial_derivatives[0][3].should.equal(1)
    bias_partial_derivatives[0][4].should.equal(1)
    bias_partial_derivatives[1].shape.should.equal((3, ))
    bias_partial_derivatives[2].shape.should.equal((2, ))


def test_get_weight_partial_derivatives_works_with_valid_input():
    # 10 x 5 x 3 x 2 network.

    # Errors for l2 - L
    errors = [
        np.ones([20, 5]),
        np.ones([20, 3]),
        np.ones([20, 2])
    ]
    # activations for l1 - L-1
    activations = [
        np.ones([20, 10]),
        np.ones([20, 5]),
        np.ones([20, 3]),
    ]

    delta_weights = get_weight_partial_derivatives_vec(activations, errors)

    delta_weights.should.be.a(list)
    delta_weights[0].should.be.a(np.ndarray)
    delta_weights[0].shape.should.equal((5, 10))
    delta_weights[0][0][0].should.equal(1)
    delta_weights[1].shape.should.equal((3, 5))
    delta_weights[1][0][0].should.equal(1)
    delta_weights[2].shape.should.equal((2, 3))
    delta_weights[2][0][0].should.equal(1)


def test_get_updated_biases_vec_works_with_valid_input():
    # 10 x 5 x 3 x 2 network
    learning_rate = 0.1
    # Biases for l+1 - L
    biases = [
        np.ones(5),
        np.ones(3),
        np.ones(2)
    ]
    delta_biases = [
        np.ones(5),
        np.ones(3),
        np.ones(2)
    ]

    updated_biases = get_updated_biases_vec(biases, delta_biases,
                                            learning_rate)

    updated_biases.should.be.a(list)
    updated_biases[0].should.be.a(np.ndarray)
    updated_biases[0].should.have.length_of(5)
    updated_biases[0][0].should.equal(0.9)
    updated_biases[1].should.have.length_of(3)
    updated_biases[1][0].should.equal(0.9)
    updated_biases[2].should.have.length_of(2)
    updated_biases[2][0].should.equal(0.9)


def test_get_updated_weights_vec_works_with_valid_input():
    # 10 x 5 x 3 x 2 network
    learning_rate = 0.1
    # Weights for l - L - 1
    weights = [
        np.ones([5, 10]),
        np.ones([3, 5]),
        np.ones([2, 3]),
    ]
    delta_weights = [
        np.ones([5, 10]),
        np.ones([3, 5]),
        np.ones([2, 3]),
    ]

    updated_weights = get_updated_weights_vec(weights, delta_weights,
                                              learning_rate)

    updated_weights.should.be.a(list)
    updated_weights.should.have.length_of(3)
    updated_weights[0].should.be.a(np.ndarray)
    updated_weights[0].shape.should.equal((5, 10))
    updated_weights[0][0][0].should.equal(0.9)
    updated_weights[1].should.be.a(np.ndarray)
    updated_weights[1].shape.should.equal((3, 5))
    updated_weights[1][0][0].should.equal(0.9)
    updated_weights[2].should.be.a(np.ndarray)
    updated_weights[2].shape.should.equal((2, 3))
    updated_weights[2][0][0].should.equal(0.9)
