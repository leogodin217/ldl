import numpy as np


def feed_forward(data, neurons, activation):
    '''
    Performs the feed forward portion of the neural network in a vectorized
    manner. 
    '''
    pass


def relu_vec(data, weights):
    '''
    Vectorized rectified linear unit acativation function that utilizes
    matrix multiplication to activate one layer for multiple observations. 

    :param data: 2D array with 1 or more observations, representing the
                 activation of a previous layer. (Initial input can be
                 considered an activation)
    :param weights: 2D array with the weights of the neurons for the layer. 

    :returns 2D array with all activations according to RelU. 
    '''

    # First peform the matrix multiplication. 
    product = np.matmul(data, weights.transpose())

    # Change all negative values to 0
    return product * (product > 0)