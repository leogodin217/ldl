import sure
import pytest
import numpy as np
from ldl.algorithms import relu_vec


def test_relu_vec_returns_correct_calculations():
    # four observations with three variables
    layer_output = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]])
    # Two neurons with three weights
    weights = np.array([[1, 1, 1], [2, 2, 2]])

    result = relu_vec(layer_output, weights)

    result.should.have.length_of(4)
    result[0].should.have.length_of(2)
    result[0].tolist().should.equal([3, 6])
    result[1].tolist().should.equal([6, 12])
    result[2].tolist().should.equal([9, 18])
    result[3].tolist().should.equal([12, 24])


def test_relu_vec_handles_negative_results():
    # Two observations with three variables
    layer_output = np.array([[1, 1, 1], [2, 2, 2]])
    # Two neurons with three weights
    weights = np.array([[-1, -1, -1], [1, 1, 1]])

    result = relu_vec(layer_output, weights)

    result[0].tolist().should.equal([0, 3])


def test_relu_vec_handles_various_sizes():
    # 100 x 4 input
    layer_output = np.ones([100, 4])
    # 15 neurons with four weights
    weights = np.ones([14, 4])
    relu_vec.when.called_with(
        layer_output, weights).should_not.throw(Exception)
