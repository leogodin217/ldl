import numpy as np


def feed_forward_vec(data, weights, neurons, activation_function):
    '''
    Performs the feed forward portion of the neural network in a vectorized
    manner.

    param: data: 2D numpy.array with one row per observation and one column per
                 feature
    :param weights: List of 2D numpy.arrays with the weights for each neuron.
                    Each item should have dimension L X L+1.
    :param neurons: List of 1D numpy.arrays with the number of neurons for each
                    layer.
    :param activation_function: A function reference for the vectorized
                                activation function. It must handle all
                                obserations in a single call.

    '''

    # Cacculate the weighted input from layer to layer 2
    weighted_input = np.matmul(data, weights[0])
    print(f'L1 weighted: {weighted_input.shape}')
    activation = None

    # Loop over the weights to find activations and then calculate the next
    # weighted input
    for weight in weights[1:]:
        activation = activation_function(weighted_input)
        weighted_input = np.matmul(activation, weight)

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
    '''

    # Set default to ones, then set everything <= 0 to 0
    return np.ones(activations.shape) * activations > 0
