'''
Utilitites not directly involved in training a neural network
'''
import numpy as np


def normalize_2d_array(data):
    '''
    Normalizes a 2D array to values betwen 0 and 1 for each column, using
    min/max normalization.

    :param data: A 2D numpy.array

    :returns A 2D numpy.array with data normulized by column
    '''

    # Get columnwise min and max
    min = np.min(data, axis=0)
    max = np.max(data, axis=0)
    diff = max - min
    # Normalize the columns
    # Add a small amount to account for columns with all zeros.
    normal_data = (data - min) / (max - min + .0000001)
    return normal_data
