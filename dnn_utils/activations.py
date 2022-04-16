import numpy as np

# Sigmoid Activation Function
def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    return A

def sigmoid_backward(dA, Z):
    s = sigmoid(Z)
    dg = s*(1-s)
    dZ = dA * dg
    return dZ

# Tanh Activation Function
def tanh(Z):
    A = np.tanh(Z)
    return A

def tanh_backward(dA, Z):
    s = tanh(Z)
    dg = 1 - np.square(s)
    dZ = dA * dg
    return dZ

# ReLU Activation Function
def relu(Z):
    A = np.maximum(0, Z)
    return A

def relu_backward(dA, Z):
    dg = np.ones((dA.shape))
    dg[Z <= 0] = 0
    dZ = dA * dg
    return dZ

# Leaky ReLU Activation Function
def leaky_relu(Z):
    A = np.maximum(0.01*Z, Z)
    return A

def leaky_relu_backward(dA, Z):
    dg = np.ones((dA.shape))
    dg[Z <= 0] = 0.01
    dZ = dA * dg
    return dZ

# SoftMax Activation Function
def softmax(Z):
    A = np.exp(Z) / np.sum(np.exp(Z))
    return A

def softmax_backward(dA, Z):
    s = softmax(Z)
    dg = s*(1-s)
    dZ = dA * dg
    return dZ


activations = {"sigmoid": sigmoid,
               "tanh": tanh,
               "relu": relu,
               "leaky_relu": leaky_relu,
               "softmax": softmax
               }

activations_backward = {"sigmoid": sigmoid_backward,
                        "tanh": tanh_backward,
                        "relu": relu_backward,
                        "leaky_relu": leaky_relu_backward,
                        "softmax": softmax_backward
                        }
