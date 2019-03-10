.. LDL Learn Deep Learning documentation master file, created by
   sphinx-quickstart on Sat Mar  9 10:23:00 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to LDL Learn Deep Learning's documentation!
===================================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:


Getting Started
===============

LDL is a neural-network framework that makes it easier to create a neural
network from scratch. By allowing data scientists to override important functions,
LDL makes it easier to code the neural network. Through this method, an entire
neural network can be coded from scratch, one piece at a time. Furthermore, each
individual component can automatically be incorporated into a working network,
without the need to have everything ready at once.

LDL will not teach you about neural networks. Rather, it provides a playground
for learning. If you need a resource for learning the basics of neural networks,
I highly recommend `Neural Networks and Deep Learning <http://neuralnetworksanddeeplearning.com>`_

Installation
------------

LDL is configured as a package, but not uploaded to PyPy. To install, use the
following pip command.

pip install git+https://github.com/leogodin217/ldl.git


Creating a network
------------------

The basic class in LDL is the Network classs from ldl.network. This class handles
all operations for training a network. A network takes weights, biases and a name
as arguments. By default, the network will use Relu activation functions and the
quadradic cost function.

.. code-block:: python

    from ldl.network import Network
    from algorithms import get_relu_weights, get_relu_biases
    # A 12 x 10 x 5 network
    layers = [12, 10, 5]
    weights = get_relu_weights(layers)
    biases = get_relu_biases(layers)
    network = Network(weights=weights, biases=biases, name='12x10x5 Network')

Training a network
------------------

The easiest way to train a network is to use train_and_validate_network. This
will train the network and show stats for train, val and test data. It will
also plot a graph of cost and error over the epochs.

.. code-block:: python

    # Examples for a 12x10x5 network
    epochs = 1000
    train_data = np.ones([50, 12])
    train_targets = np.ones([50, 5])
    val_data = np.ones([20, 12])
    val_targets = np.ones([20, 5])
    test_data = np.ones([20, 12])
    test_targets = np.ones([20, 5])
    test_labels = np.ones([20, 1])
    learning_rate = 0.3
    results = network.train_and_validate(epochs, train_data, train_targets,
                                         val_data, val_targets, test_data,
                                         test_targets, test_labels,
                                         learning_rate)


Building functions from scratch
-------------------------------

While LDL provides a framework for building a network, most functionality can
be overridden with custom functions. All functions are stored internally
in the network. To override a function, simply use dot notatio to provide
your own function. Please read the documenation on the default functions to
understand expected inputs and outputs.

The following functions can be overridden.

* activation_fun
* activation_derivative_fun
* cost_fun
* cost_derivative_fun
* output_error_fun
* backpropagate_errors_fun
* updated_biases_fun
* updated_weights_fun
* bias_partial_derivatives_fun
* weight_partial_derivatives_fun


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
