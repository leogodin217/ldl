import numpy as np


def feed_forward_vec(data, weights, neurons, activation_function):
    '''
    Performs the feed forward portion of the neural network in a vectorized
    manner.
    ***** This function could use more tests of the algorithm*****

    param: data: 2D numpy.array with one row per observation and one column per
                 feature
    :param weights: List of 2D numpy.arrays with the weights for each neuron.
                    Each item should have dimension L+1 X L+2. This is so we
                    can add the bias neurons.
    :param neurons: List of 1D numpy.arrays with the number of neurons for each
                    layer.
    :param activation_function: A function reference for the vectorized
                                activation function. It must handle all
                                obserations in a single call.

    '''

    # Add the bias neuron to the input data
    bias_data = np.append(data, np.ones([len(data), 1]), axis=1)
    # Cacculate the weighted input from layer to layer 2
    weighted_input = np.matmul(bias_data, weights[0])
    activation = None

    # Loop over the weights to find activations and then calculate the next
    # weighted input
    for weight in weights[1:]:
        # Since we already have weighted input, we just need to calculate
        # activation.
        activation = activation_function(weighted_input)
        # Add the bias to the activation (Add a bias neuron)
        activation_bias = np.append(activation, np.ones([len(activation), 1]),
                                    axis=1)
        weighted_input = np.matmul(activation_bias, weight)

    # Calculate the final activation
    return activation_function(weighted_input)


def relu_vec(weighted_input):
    '''
    Vectorized rectified linear unit acativation function that calculated
    the activation based on weighted input.

    :param weighted_input: 2D array representing the activation of the previous
                           layer multipled by the weights to the current layer.

    :returns 2D array with all activations according to RelU.
    '''

    # Change all negative values to 0. Here we use a trick where logical are
    # coerced to 0 or 1
    return weighted_input * (weighted_input > 0)


def relu_vec_differential(activations):
    '''
    Vectorized differential of the rectified linear unit activation function.
    Differentiates the entire layer for all observations.
    Simply returns 1 for values > 0 and 0 for values <= 0

    :param activations: 2D numpy.array of layer activations for the layer

    :returns a 2D numpy.array with derivative of RelU for all values.
    '''

    # Set default to ones, then set everything <= 0 to 0
    return np.ones(activations.shape) * activations > 0


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

    :param y: 2D numpy.array with the expected values for all observations.
    :param y_hat: 2D numpy.array of same shape as y, with predicted values for
                  all observations.

    :returns A decimal representing the average error for all observations.
    '''
    # Subtract predicted from expected
    diff = y_predicted - y
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
    ??? Need to clarify

    :param y: 2D numpy.array with the expected values of all observations.
    :param y_predicted: 2D numpy.array of same shape as y, with predicted
                        values for all observations.
    '''

    # This is simple, since the derivative is simply the difference between
    # predicted and actual
    return y_predicted - y
