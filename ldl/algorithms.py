import numpy as np

'''
Algorithms requried to train a neural network.
'''


def feed_forward_vec(data, weights, biases, activation_fun,
                     derivative_fun):
    '''
    Performs the feed forward portion of the neural network in a vectorized
    manner.
    ***** This function could use more tests of the algorithm*****

    Args:
        :param: data: 2D numpy.array with one row per observation and one
                      column per feature
        :param weights: List of 2D numpy.arrays with the weights for each
                        neuron.
                        Each item should have dimension l+1 X l.
        :param biases: List of 2D numpy.arrays with the biases for each neuron.
        :param activation_fun: A function reference for the vectorized
                                    activation function. It must handle all
                                    obserations in a single call.
        :param derivative_fun: A function reference for the vectorized
                                    derivative function. This is needed so we
                                    can calculate the derivative for later use
                                    in back propagation.
    Returns:
        A dict with activations and derivatives of each neuron.

    '''

    # Cacculate the weighted input from layer to layer 2
    weighted_input = np.matmul(data, weights[0].transpose()) + biases[0]
    activation = None

    # Store activations for each layer, so they can be used in backpropagation
    activations = []
    # Input layer is added as-is
    activations.append(data)
    # Store the derivatives for each layer except the input layer so they
    # can be used in backpropagation
    derivatives = []
    # Loop over the weights to find activations and then calculate the next
    # weighted input
    for index, weight in enumerate(weights[1:]):
        # Since we already have weighted input, we just need to calculate
        # activation.
        activation = activation_fun(weighted_input)
        activations.append(activation)
        derivative = derivative_fun(weighted_input)
        derivatives.append(derivative)

        weighted_input = np.matmul(activation,
                                   weight.transpose()) + biases[index + 1]

    # Calculate the final activation
    output_activation = activation_fun(weighted_input)
    activations.append(output_activation)
    output_derivative = derivative_fun(weighted_input)
    derivatives.append(output_derivative)

    return {'activations': activations, 'derivatives': derivatives}


def relu_vec(weighted_input):
    '''
    Vectorized rectified linear unit acativation function that calculated
    the activation based on weighted input.

    Args:
        :param weighted_input: 2D array representing the activation of the
               previous layer multipled by the weights to the current layer.
    Returns:
        2D array with all activations according to RelU.
    '''

    # Change all negative values to 0. Here we use a trick where logical are
    # coerced to 0 or 1
    return weighted_input * (weighted_input > 0)


def relu_derivative_vec(weighted_input):
    '''
    Vectorized differential of the rectified linear unit activation function.
    Differentiates the entire layer for all observations.
    Simply returns 1 for values > 0 and 0 for values <= 0

    Args:
        :param activations: 2D numpy.array of layer activations for the layer.

    Returns:
        2D numpy.array with derivative of RelU for all values.
    '''

    # Set default to ones, then set everything <= 0 to 0
    return np.ones(weighted_input.shape) * (weighted_input > 0)


def quadradic_cost_vec(y, y_predicted):
    '''
    Calculates the quadradic cost function in a vectorized manner. The
    Quadradic Cost function is the same as mean-squared error. In particular,
    it sums the squared errors for each output class, then averages them
    over the entire output. Both formulas below are expressing the same thing.
    y = expected output and y_predicted = actual output.
    y(x) = y, AL = y_predicted = output of last activation layer

    cost = 1/2M SUM(LENGTH(y - y_predicted)^2)
    cost = 1/2M SUM(LENGTH(y(x) - AL)^2)

    This formula is great, because it is easy to calculate and its derivative
    is simply

    Args:
        :param y: 2D numpy.array with the expected values for all observations.
        :param y_hat: 2D numpy.array of same shape as y, with predicted values for
                      all observations.

    Returns:
        A decimal representing the average error for all observations.
    '''
    # Subtract predicted from expected
    diff = y - y_predicted
    # Square the differences
    squares = np.power(diff, 2)
    # Get the sum of squared errors for each observation
    sum_of_squares = np.sum(squares, axis=1)
    # To carry out the formula, LENGTH(y - y_predicted) includes finding the
    # square root of the squared differences. Then, the cost function takes
    # the square. This means we can skip both steps since squaring a square
    # root returns the original value.
    # Calculate final average cost, considering we should divde each square
    cost = np.mean(sum_of_squares / 2)
    return cost


def quadradic_cost_derivative_vec(y, y_predicted):
    '''
    Calculates to partial derivative of the cost function with respect to
    each output neuron.

    Args:
        :param y: 2D numpy.array with the expected values of all observations.
        :param y_predicted: 2D numpy.array of same shape as y, with predicted
                            values for all observations.
    Returns:
        2D numpy array with the partial derivatives of the cust function.
        Should have one row per observation and one column per output neuron.
    '''

    # Notice that it is y_predicted - y, not y - y_predicted as it is in the
    # cost function. In the reverse order, we would be adding to our overall
    # cost instead of decreasing it.
    return y_predicted - y


def output_error_vec(y, y_predicted, cost_derivative_fun,
                     activation_derivative_fun, weighted_input):
    '''
    Vectorized calculation of the error of the output layer depending on the
    derivatives of the cost function and the activation functions.

    Args:
        :param y: Expected values of the output layer
        :param y_predicted: Actual values of the output layer
        :param cost_derivative_fun: Function that calculates the derivative
                                         of the cost function depending on the
                                         activations of the output layer.
        :param activation_derivative_fun: Function that calculates the
                                               derivative of the activation
                                               of the output layer.
        :param weigted_input: A 2D numpy.array with the weighted input to the
                              output layer. Holds one row per observation and one
                              column per output neuron.
    Returns:
        2D numpy.array holding the error of each neuron in the output layer.
        One row for each observation, one column per neuron in the output
        layer.
    '''

    cost_derivative = cost_derivative_fun(y, y_predicted)
    activation_derivative = activation_derivative_fun(weighted_input)
    # Component-wise product
    error = cost_derivative * activation_derivative
    return error


def layer_error_vec(weights, errors, derivatives):
    '''
    Vectorized calculation of error for a single layer.

    Params:
        :param weights: 2D numpy.array representing the input weights for l+1.
        :param errors: 2D numpy.array representing the errors for l+1. One row
                       for each observation.
        :param derivatives: numpy.array representing the derivatives of the
                            current layer. One row for each observation.
    Returns:
        A 2D numpy.array representing the errors for each neuron in a single
        layer. One row per observation and one column for each neuron.
    '''

    # This determines te amount that the derivative impacts future layers
    weighted_error = np.matmul(errors, weights)
    # Component-wise multiplication simply sets the error with respect to
    # future layers
    layer_error = weighted_error * derivatives
    return layer_error


def backpropagate_errors(weights, derivatives, output_errors):
    '''
    Backpropagates errors through all the layers, except the input layer.

    Args:
        :param weights: A list of 2D numpy arrays representing the weights for
                        each layer.
        :param derivatives: A list of 2D numpy.arrays representing the derivatives
                            fore each layer except the input layer. Each array
                            contains one row per observation and one column per
                            neuron in the layer.
        :param output_errors: A 2D numpy.array representing the errors fo the
                              output error. One row per observation and one column
                              for each output neuron.
    Returns:
        A list of 2D numpy.arrays representing the errors for each layer
        except the input layer. Each array contains one row per observation and
        one column for each neuron in the layer.
    '''

    # Since backpropagation start from the end, we will reverse the lists
    weights_back = weights[::-1]  # Simple form of reversing a list
    derivatives_back = derivatives[::-1]

    errors = []
    # We already have the errors for the output layer
    errors.insert(0, output_errors)

    # Loop through layers L-1 through layer 2 and get the errors
    for index, weight in enumerate(weights_back):
        layer_error = layer_error_vec(weight, errors[0],
                                      derivatives=derivatives_back[index])
        errors.insert(0, layer_error)
    return errors


def get_bias_partial_derivatives_vec(errors):
    '''
    Calculates the partial derivatives for each neuron bias. This is equal to
    the mean error of each neuron.

    Args:
        :param errors: A list of 2D numpy.arrays with the errors for each
                       layer.

    Returns:
        A list of 2D numpy.arrays with the partial derivatives of all biases.
    '''
    delta_bias = []
    for layer in errors:
        delta_bias.append(np.mean(layer, axis=0))
    return delta_bias


def get_weight_partial_derivatives_vec(activations, errors):
    '''
    Calculates the partial derivatives for each weight. This is equal to the
    activation of input times the error of the output.

    Args:
        :param activations: A list of 2D numpy.arrays representing the
                            activations for all observations, including the
                            input layer.
        :param errors: A list of 2D numpy.arrays with errors for each layer.

    Returns:
    A list of 2D numpy.arrays with the partial derivatives of each weight.
    '''

    delta_weights = []

    for index, error in enumerate(errors):
        # Calculate the sum of partial derivatives for all observations
        layer_delta_weights = np.matmul(error.transpose(), activations[index])
        # Calculate the mean partial derivative
        observations = error.shape[0]
        delta_weights.append(layer_delta_weights / observations)
    return delta_weights


def get_updated_biases_vec(biases, delta_biases, learning_rate):
    '''
    Updates the biases for a network after gradient descent has calculated the
    partial derivatives of each bias.
    new_bias = bias - learning rate * delta_bias

    Args:
        :param biases: A list of 1D numpy.arrays representing the current
                       biases
        :param delta_biases: A list of 1D numpy.arrays representing the partial
                             derivatives of the biases after gradient descent.
        :param learning_rate: A numeric value to multiply the change by.

    Returns:
        A list of 1D numpy.arrays representing the new biases
    '''

    new_bias = []
    # Loop through the layers
    for index, bias in enumerate(biases):
        # Update the bias
        new_bias.append(bias - (delta_biases[index] * learning_rate))
    return new_bias


def get_updated_weights_vec(weights, delta_weights, learning_rate):
    '''
    Updates the weights for a network after gradient descent has calculated the
    partial derivatives of each weight.
    new_weight = weight - learning rate * delta_weight

    Args:
        :param weights: A list of 2D numpy.arrays representing the current weights
        :param delta_biases: A list of 2D numpy.arrays representing the partial
                             derivatives of the weights after gradient descent.
        :param learning_rate: A numeric value to multiply the change by.

    Returns:
        A list of 2D numpy.arrays representing the new weights
    '''

    new_weights = []
    # Loop through the layers
    for index, weight in enumerate(weights):
        # Update the weight
        new_weights.append(weight - (delta_weights[index] * learning_rate))
    return new_weights


def get_relu_biases(layers):
    '''
    Calculates optimum starting bias for relu-activated networks. This is
    simply a bunch of zeros.

    Args:
        :param shape: A list of ints representign the size of each layer.

    Returns:
        A list of 1D numpy.arrays representing the biases
    '''
    biases = []
    for layer in layers[1:]:
        # Layer is an int describing the number of neurons in the layer
        biases.append(np.zeros(layer))
    return biases


def get_relu_weights(layers):
    '''
    Calculates optimized random weights for a relu-activated network using the
    formula devised by He et al (2016) https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf
    initializes weights for each layer to a normal distribution with mean=0
    and standard deviation = sqrt(2/nl) where nl=number of input neurons to the
    layer

    Args:
        :param layers: A list of ints representing the number of neurons in
                       each layer.

    Returns:
        A list of 2D numpy.arrays
    '''
    weights = []
    for i in range(len(layers) - 1):
        mean = 0.0
        standard_deviation = np.sqrt(2 / layers[i])
        # Size is l out x l in
        size = (layers[i + 1], layers[i])
        weight = np.random.normal(loc=0.0, scale=standard_deviation,
                                  size=(layers[i + 1], layers[i]))
        weights.append(weight)
    return weights
