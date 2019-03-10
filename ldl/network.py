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
from ldl.predictions import predict_digit
import numpy as np
import pandas as pd


class Network:
    '''
    Defines the network topology and manages operations. Many functions used
    in the network can be overridden.
    '''

    def __init__(self, weights, biases, name='default'):
        '''
        Initializes the network and functions. Many function defaults are set
        on initialization. All can be overriden with dot notation. The
        following functions can be overriden:
        - activation_fun
        - activation_derivative_fun
        - cost_fun
        - cost_derivative_fun
        - output_error_fun
        - backpropagate_errors_fun
        - updated_biases_fun
        - updated_weights_fun
        - bias_partial_derivatives_fun
        - weight_partial_derivatives_fun
        - predict_fun

        Args:
            :param weights: A list of 2D numpy.arrays representing the weights
                            for each layer. Each entry should have dimension
                            l x l-1
            :param biases: A list of 1D numpy.arrays representing the biases of
                           each layer.
            :param name: A string representing the name of the network. Useful
                         when comparing multiple network architectures.
        '''
        self.weights = weights
        self.biases = biases
        self.name = name
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
        self.activations = []
        self.predict_fun = predict_digit

    def get_shape(self, weights):
        '''
        Gets the shape of the network from the weights

        Args:
            :param weights: A list of 2D numpy arrays representing the weights
                            of the network

        Returns:
            A list of ints, representing the shape of the network.
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

    def train(self, epochs, data, targets, learning_rate):
        '''
        Trains a neural network without checking for cost and accuracy. This is
        the fastest method of training, but requires the user to perform all
        calculations for errors and accuracy.

        Args:
            :param epochs: Integer representing the number of epochs to run
            :param data: A 2D numpy.array with the data used to train the
                         network
            :param targets: A 2D numpy.array with the targets of the data.
                            Should have one row per observation and one column
                            per output neuron in the network.
            :param learning_rate: A float representing the learning rate of the
                                  network.
        Returns:
            A dict containing the updated weigths, biases, final cost and final
            activations.
        '''

        results = {}
        for epoch in range(epochs):
            results = self.run_one_epoch(data, targets, learning_rate)
            self.weights = results['updated_weights']
            self.biases = results['updated_biases']
            self.activations = results['activations']
            print(f'Cost: {results["cost"]}')
        return results

    def train_and_validate(self, epochs, train_data, train_targets, val_data,
                           val_targets, test_data, test_targets, test_labels,
                           learning_rate, print_status=True):
        '''
        Trains and validates a network. This is useful to see the performance
        of a network over epochs. Prints statuses and plots results at the end.

        Args:
            :param epochs: Integer representing the number of epochs.
            :param train_data: 2D numpy.array with the data used to train the
                               network. Should ahve one row per observatio and
                               one column per input neuron.
            :param train_targets: 2D numpy.array with the targets of the
                                  training data. Should have one row per
                                  observation and one column per output neuron
                                  in the network.
            :param val_data: 2D numpy.array representing used to validate the
                             network. Should have one row per observation and
                             one column per input neuron.
            :param val_targets: 2D numpy.array representing the validation
                                targets. Should have one row per observation
                                and one column per output neuron.
            :param test_data: 2D numpy.arraay with the data used to test the
                              network. Should have one row per observation and
                              one column per input neuron.
            :param test_targets: 2D numpy.array with the targets used to test
                                 the network. Should have one row per
                                 observation and one column per output neuron.
            :param test_labels: List of labels of the test data. Labels
                                represent the human-readable expected value.
                                Should contain one row per observation.
            :param learning_rate: Float representing the learning rate of the
                                  network.
            :param print_status: Boolean specififying if the status should be
                                 printed as the network learns. If true, status
                                 will be printed in epochs 1 - 10,
                                 then 20 - 500 by tens, then all by one
                                 hundreds.
        Returns:
            A Pandas table with the results of each epoch, including train cost,
            validation cost, test cost and test error.
        '''
        train_cost = ''
        val_cost = ''
        test_cost = ''
        test_error = ''
        all_results = []
        for epoch in range(epochs):
            # Train one epoch
            results = self.run_one_epoch(train_data, train_targets,
                                         learning_rate)
            # train_cost = results['cost']
            self.weights = results['updated_weights']
            self.biases = results['updated_biases']
            self.activations = results['activations']
            train_cost = results['cost']
            # Check validation
            val_predictions = self.predict(val_data)
            val_cost = self.cost_fun(val_targets, val_predictions)
            # check test
            test_predictions = self.predict(test_data)
            test_cost = self.cost_fun(test_targets, test_predictions)
            test_error = self.get_error_rate(test_predictions, test_labels,
                                             self.predict_fun)
            if print_status:
                # Generate messages
                train_cost_string = '{:.5f}'.format(train_cost)
                test_cost_string = '{:.5f}'.format(test_cost)
                val_cost_string = '{:.5f}'.format(val_cost)
                test_error_string = '{:.3f}'.format(test_error * 100) + '%'
                message = f'Epoch: {epoch + 1}, '
                message += f'\tTrain cost: {train_cost_string}, '
                message += f'Val cost: {val_cost_string}, '
                message += f'Test cost: {test_cost_string}, '
                message += f'Test error: {test_error_string}'
                # Print first 10
                if epoch + 1 < 10:
                    print(message)
                # Print every 10 up to 500
                elif epoch + 1 < 500 and (epoch + 1) % 10 == 0:
                    print(message)
                # Print every 100 after 500
                elif epoch + 1 >= 500 and (epoch + 1) % 100 == 0:
                    print(message)
                elif epoch == epochs:
                    print('Final results:')
                    print(message)
            all_results.append({'train_cost': train_cost,
                                'val_cost': val_cost,
                                'test_cost': test_cost,
                                'test_error': test_error,
                                'model': self.name})
        results_table = pd.DataFrame(all_results)
        results_table.plot(title=self.name)
        return results_table

    def run_one_epoch(self, data, targets, learning_rate):
        '''
        Runs a single epoch of the network.

        Args:
            :param data: 2D numpy.array with data to train the newtwork with.
                         Should have one row per observation and one column per
                         input neuron.
            :param targets: 2D numpy.array with the expected targets of the
                            data. Should have one row per observation and one
                            column per output neuron.
            :param learning_rate: Flot representig the learning rate of the
                                  network.

        Returns:
            A dict with the updated weights, biases, activations and cost.
        '''
        # Feed forward steps
        ff = feed_forward_vec(data, self.weights, self.biases,
                              self.activation_fun,
                              self.activation_derivative_fun)
        activations = ff['activations']
        derivatives = ff['derivatives']
        predictions = activations[-1]
        cost = self.cost_fun(targets, predictions)
        # print(f'Cost: {cost}')

        # Calculate errors
        # delta_cost = cost_derivative_fun(targets, ff['activations'][-1])
        # Output of L-1 * weights for L
        weighted_input = np.matmul(activations[-2],
                                   self.weights[-1].transpose())
        output_error = self.output_error_fun(
            y=targets, y_predicted=predictions,
            cost_derivative_fun=self.cost_derivative_fun,
            activation_derivative_fun=self.activation_derivative_fun,
            weighted_input=weighted_input)
        # Back propagate errors for l+1 - L-1
        errors = self.backpropagate_errors_fun(weights=self.weights[1:],
                                               derivatives=derivatives[0:-1],
                                               output_errors=output_error)
        # update weights and biases
        delta_biases = self.bias_partial_derivatives_fun(errors)
        updated_biases = self.updated_biases_fun(self.biases, delta_biases,
                                                 learning_rate)
        delta_weights = self.weight_partial_derivatives_fun(activations,
                                                            errors)
        updated_weights = self.updated_weights_fun(self.weights, delta_weights,
                                                   learning_rate)
        return {'updated_weights': updated_weights,
                'updated_biases': updated_biases,
                'activations': activations,
                'cost': cost}

    def predict(self, data):
        '''
        Predicts a set of observations on a trained network.

        Args:
            :param data: 2D numpy.array with the data to predict. Should have
                         one row per observation and one column per input
                         neuron.

        Returns:
            A 2D numpy.array with the predictions. Output includes one row
            per observation and one column per output neuron.
        '''
        ff = feed_forward_vec(data, self.weights, self.biases,
                              self.activation_fun,
                              self.activation_derivative_fun)
        return(ff['activations'][-1])

    def get_error_rate(self, predictions, targets, predict_function):
        '''
        Calculates the error rate of predictions compared to targets.

        Args:
            :param predictions: Usually a numpy.array of output from a neural
                                network.
            :param targets: A numpy.array of targets from a neural network. In some
                            cases, this can be an array. In others it may be a
                            class.
            :param predict_function: A function that returns the prediction from
                                     the output of a neural network. This should
                                     return a value that can be compared for
                                     equality with the targets.
        Returns:
            A float representing the percent error rate for the predictions.
        '''
        num_observations = len(predictions)
        correct = 0
        # See which predictions are correct
        for index, prediction in enumerate(predictions):
            if predict_function(prediction) == targets[index]:
                correct += 1

        # The percent that were incorrectly classified
        return 1 - (correct / num_observations)
