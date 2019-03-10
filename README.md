# LDL - Learning Deep Learning

## Summary
LDL is a neural-network framework that makes it easier to create a neural
network from scratch. By allowing data scientists to override important functions,
LDL makes it easier to code the neural network. Through this method, an entire
neural network can be coded from scratch, one piece at a time. Furthermore, each
individual component can automatically be incorporated into a working network,
without the need to have everything ready at once.

LDL is well documented in the style of Read the Docs. To learn how to use LDL
read documentation [here.](https://leogodin217.github.io/ldl/)

Three Juypter notebooks are provided in the github repository to show examples
of using LDL.

**Network Training** - Shows how to compare different learning rates using the default
settings to classify the MNIST dataset.

**Network Size** - Shows how to compare different architectures using the default
settings to classify the MNIST dataset.

**From Scratch** - Shows how to implement your own functions for activation and cost.

This README, covers the analysis of the network on the MNIST dataset.

## Exploratory Analysis

### Weights and Biases


### Architecture
First, we want to see what architecture works best on the MNIST dataset. For this
test, we will use the same method determining weights and biases for the Relu
activation function. We will use a utility function from LDL to normalize the
input data.

Setup

```python
from mnist import mnist
from ldl.algorithms import *
from ldl.utitlities import *
from ldl.predictions import *
from ldl.network import Network
import numpy as np
```

Data

```python
# Get the data
x_train, t_train, x_test, t_test = mnist.load()

# Normalize the data and separate into train, val and test
train_data = normalize_2d_array(x_train[:40000])
val_data = normalize_2d_array(x_train[40001:])
test_data = normalize_2d_array(x_test)

train_labels = t_train[:40000]
val_labels = t_train[40001:]
test_labels = t_test

# Convert labels to targets for output neurons
train_targets = np.zeros([train_labels.shape[0], 10])
for index, label in enumerate(train_labels):
    train_targets[index, label] = 1

val_targets = np.zeros([val_labels.shape[0], 10])
for index, label in enumerate(val_labels):
    val_targets[index, label] = 1

test_targets = np.zeros([test_labels.shape[0], 10])
for index, label in enumerate(test_labels):
    test_targets[index, label] = 1


learning_rate = 0.2
epochs=2000
```

One architecture. The same code is used for each architecture with only the shape
and name changing.

```python
shape = [784, 10, 10]
biases = get_relu_biases(shape)
weights = get_relu_weights(shape)
name = '784x10x10 Learning Rate 0.2'
network = Network(weights=weights, biases=biases, name=name)
three_neg_three_by_1000 = network.train_and_validate(epochs, train_data, train_targets, val_data, val_targets,
                                                     test_data, test_targets, test_labels, learning_rate)
```

Note that LDL automatically plots the train, validation and test costs with the
test error. This is useful in deciding on an architecture.
![Architecture results](https://github.com/leogodin217/ldl/raw/master/images/architecture_results.png "Architecture results")
