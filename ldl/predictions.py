'''
Holds prediction functions. These are separated, because each use case will
probably have a unique prediction function.
'''
import numpy as np


def predict_digit(output):
    '''
    Predicts a digit 0-9 from an 1D numpy array of output values.

    Args:
        :param output: 1D numpy.array representing the output of a 0-9
                       classification.
    Returns:
        Integer representing the digit
    '''

    # Simply return the index of the highest output. Since we use zero-based
    # indexing, that is the digit.
    return np.argmax(output)
