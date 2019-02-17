'''
Neural network structure that encapsulates the needed funality to build,
train and predict.
'''


from ldl.algorithms import feed_forward_vec
from ldl.algorithms import relu_vec
from ldl.algorithms import relu_derivative_vec
from ldl.algorithms import quadradic_cost_vec
from ldl.algorithms import quadradic_cost_derivative_vec
from ldl.algorithms import backpropagate_errors
from ldl.algorithms import get_updated_biases_vec
from ldl.algorithms import get_updated_weights_vec
from ldl.algorithms import get_bias_partial_derivatives_vec
from ldl.algorithms import get_weight_partial_derivatives_vec
from ldl.algorithms import output_error_vec
import numpy as np


class Network:
    '''
    Defines the network topology and manages operations
    '''
    def __init__(self, weights, biases):
        self.weights = weights
        self.biases = biases
        self.shape = self.get_shape(self.weights)
        self.activation_fun = relu_vec
        self.activation_derivative_fun = relu_derivative_vec
        self.cost_fun = quadradic_cost_vec
        self.cost_derivative_fun = quadradic_cost_derivative_vec
        self.output_error_fun = output_error_vec
        self.backpropagate_errors_fun = backpropagate_errors
        self.updated_biases_fun = get_updated_biases_vec
        self.updated_weights_fun = get_updated_weights_vec
        self.bias_partial_derivatives_fun = get_bias_partial_derivatives_vec
        self.weight_partial_derivatives_fun = get_weight_partial_derivatives_vec

    def get_shape(self, weights):
        '''
        Gets the shape of the network from the weights

        :param weights: A list of 2D numpy arrays representing the weights of
                        the network

        :returns A list of ints, representing the shape of the network.
        '''
        shape = []
        # Weights always are of dimension l+ 1 x l. We can iterate the weights
        # and use the number of columns as the number of neurons in the
        # prefious layer.
        for layer in weights:
            shape.append(layer.shape[1])
        # Add the last layer, which is the number of rows in the last weights.
        shape.append(weights[-1].shape[0])
        return shape

    def train_network(self, epochs, data, weights, biases, targets,
                       learning_rate):
        updated_weights = weights
        updated_biases = biases
        for epoch in range(epochs):
            results = self.run_one_epoch(data, updated_weights, updated_biases,
                                         targets, learning_rate)
            updated_weights = results['updated_weights']
            updated_biases = results['updated_biases']
        return {'weights': updated_weights, 'biases': updated_biases}

    def run_one_epoch(self, data, weights, biases, targets, learning_rate):
        # Feed forward steps
        ff = feed_forward_vec(data, weights, biases,
                              self.activation_fun,
                              self.activation_derivative_fun)
        activations = ff['activations']
        derivatives = ff['derivatives']
        predictions = activations[-1]
        cost = self.cost_fun(targets, predictions)
        print(f'Cost: {cost}')

        # Calculate errors
        # delta_cost = cost_derivative_fun(targets, ff['activations'][-1])
        # Output of L-1 * weights for L
        weighted_input = np.matmul(activations[-2], weights[-1].transpose())
        output_error = self.output_error_fun(
            y=targets, y_predicted=predictions,
            cost_derivative_fun=self.cost_derivative_fun,
            activation_derivative_fun=self.activation_derivative_fun,
            weighted_input=weighted_input)
        # Back propagate errors for l+1 - L-1
        errors = self.backpropagate_errors_fun(weights=weights[1:],
                                               derivatives=derivatives[0:-1],
                                               output_errors=output_error)
        # update weights and biases
        delta_biases = self.bias_partial_derivatives_fun(errors)
        updated_biases = self.updated_biases_fun(biases, delta_biases,
                                                 learning_rate)
        delta_weights = self.weight_partial_derivatives_fun(activations,
                                                            errors)
        updated_weights = self.updated_weights_fun(weights, delta_weights,
                                                   learning_rate)
        return {'updated_weights': updated_weights,
                'updated_biases': updated_biases}
