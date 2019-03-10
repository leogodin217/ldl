# LDL - Learning Deep Learning

## Summary
LDL is a neural-network framework that makes it easier to create a neural
network from scratch. By allowing data scientists to override important functions,
LDL makes it easier to code the neural network. Through this method, an entire
neural network can be coded from scratch, one piece at a time. Furthermore, each
individual component can automatically be incorporated into a working network,
without the need to have everything ready at once.

This README will describe LDL and show results for digit prediction on the MNIST
dataset. Top classifiers achieve > 99% accuracy. We will see how LDL performs on
the same dataset.


## Why you should use LDL

If you want to code a neural network from scratch, but find it too daunting to
start from nothing, then LDL might be for you. Instead of coding the entire thing
in one step, you can start with simple things like activation functions and derivative
functions.

If you have a difficult time understanding back propagation, LDL might be for you.
By coding the algorithm from scratch, you will prove full understanding of the theory
and practice.

If you want to practice vector programming, LDL might be for you. By implementing
various functions of a neural network, you will learn to utilize matrix multiplication
to perform incredible amounts of work. With a little patience, you will find that
almost every step in training a neural network can be accomplished with matrix
multiplication instead of for loops.

## Why you should not use LDL

If you want fast, production code, LDL is not for you. LDL is not intended for
production work. It is entirely written in plain python with the use of Numpy for
matrix multiplication and Pandas for holding and plotting results. Though the
default methods are all verctorized, this is still very slow. If you need a
production framework, I would recommend something like Pytorch or TensorFlow.


## Documentation
LDL is well documented in the style of Read the Docs. To learn how to use LDL
read documentation [here.](https://leogodin217.github.io/ldl/)

Three Juypter notebooks are provided in the github repository to show examples
of using LDL.

**Network Training** - Shows how to compare different learning rates using the default
settings to classify the MNIST dataset.

**Network Size** - Shows how to compare different architectures using the default
settings to classify the MNIST dataset.

**From Scratch** - Shows how to implement your own functions for activation and cost.

### Data

The data for this experiment comes from the MNIST data set. This dataset includes
60,000 hand-written digits with labels describing each observation. This is a good
data set to test a basic neural network as it is well known and well researched.
Modern algorithms achieve well over 99% accuracy at this time. It is unlikely
that LDL will achieve 99%.

## Exploratory Analysis

When creating a neural network from scratch, implementing the code is only the
first step. Once the code is ready, the data scientist must determine the best
architecture, biases and weights.

### Weights and Biases

My first attempts at training with random weights and biases resulted in a 90% error rate.
After playing with the weights to make the range narrower and wider, I found that
smaller weights were more likely to provide results. Since the default activation
function in LDL is the Relu function, I found there is a best method for setting
initial weights and biases for Relu-based networks. He et al (2015) proposed
a method that is based off the number of input and output neurons for a set of
weights, while starting all biases at zero. The implementation of He's algorith
was created in the ldl.algorithms module.

Once I used this method, I was able to move from 70% error rate to 90% on a network
with fifteen neurons in the hidden layer.  Upon further investigation I found
that using random weights between -1 and 1 were causing all neurons to die;
a condition when a neuron never fires. The He method was a life saver, as I was
worried that I would not get results from all my work.

```
def get_relu_weights(layers):
    '''
    Calculates optimized random weights for a relu-activated network using the
    formula devised by He et al (2016) initializes weights for each layer to a
    normal distribution with mean=0 and standard deviation = sqrt(2/nl) where
    nl=number of input neurons to the layer

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
```

### Architecture

The architecture of a neural network involves the number of layers and neurons
in the network. For MNIST, I used a three-layer network, with one hidden layer.
Then, I tested five different configurations with increasing neurons in the hidden layer
to see which one worked best.

First, we want to see what architecture works best on the MNIST dataset. For this
test, we will use the same method determining weights and biases for the Relu
activation function. We will use a utility function from LDL to normalize the
input data.

**Setup:**

```
from mnist import mnist
from ldl.algorithms import *
from ldl.utitlities import *
from ldl.predictions import *
from ldl.network import Network
import numpy as np
```

**Data:**

```
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

**One architecture:** The same code is used for each architecture with only the shape
and name changing.

```
shape = [784, 10, 10]
biases = get_relu_biases(shape)
weights = get_relu_weights(shape)
name = '784x10x10 Learning Rate 0.2'
network = Network(weights=weights, biases=biases, name=name)
three_neg_three_by_1000 = network.train_and_validate(epochs, train_data, train_targets, val_data, val_targets,
                                                     test_data, test_targets, test_labels, learning_rate)
```

Note that LDL automatically plots the train, validation and test costs with the
test error. This is useful in deciding on an architecture and troubleshooting
the network. We see some interesting results here. The 768x100x10 network performed
best with an error rate of 3.87% over two-thousand epochs. Notice the 784x10x10
network increased test errror over time. This is a sign of overfitting. Not only
did the network produce poor results, but it quickly overfitted its best case.

![Architecture results](https://github.com/leogodin217/ldl/raw/master/images/architecture_results.png "Architecture results")

### Learning Rate

Learning rate is a key metaparameter in neural networks. Since networks are trained
using gradient descent to modify weights and biases, the learning rate impacts
the magnitude of change. If the learning rate is too high, we may miss the global
minimum. Too low and we may never get to it. In this case, we will test four learning
rates ranging from 0.03 to .3.

**Setup:**

```
from mnist import mnist
from ldl.algorithms import *
from ldl.utitlities import *
from ldl.predictions import *
from ldl.network import Network
import numpy as np
```

**Data:**

```
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
```

**Defaults:**

```

shape = [784, 100, 10]
biases = get_relu_biases(shape)
weights = get_relu_weights(shape)
epochs=10000
```

We see that 0.2 provides the best learning

## Conclusion

Conclude something.
