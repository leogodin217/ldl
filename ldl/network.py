class Network:
    '''
    Defines the network topology and manages operations
    '''

    def __init__(self, shape):
        self.shape = shape
        # Add a bias neuron for all but the output layer
        self.layers = [l + 1 for l in shape[:-1]] + [shape[-1]]

