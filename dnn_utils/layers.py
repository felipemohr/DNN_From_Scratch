import numpy as np
from dnn_utils.activations import *
from dnn_utils.cost import *

class Layer():

    def __init__(self, n_units, activation="relu"):
        self.n_units = n_units
        self.activation = activation

    def initializeWeights(self, n_units_prev):

        # He initialization
        self.W = np.random.randn(self.n_units, n_units_prev) * np.sqrt(2/n_units_prev)
        self.b = np.zeros((self.n_units, 1))

        self.W_shape = self.W.shape
        self.b_shape = self.b.shape
        
    def printLayer(self):
        print("activation: ", self.activation)
        print("n_units: ", self.n_units)
        print("W: ", end="")
        print(self.W_shape)
        print("b: ", end="")
        print(self.b_shape)

    def linearActivationForward(self, A_prev):
        Z = np.dot(self.W, A_prev) + self.b
        linear_cache = (A_prev, self.W, self.b)

        A = activations[self.activation](Z)
        activation_cache = Z

        cache = (linear_cache, activation_cache)
        return A, cache

    def linearActivationBackward(self, dA, cache, lamb=0.1, prev_keep_prob=1.0, D_prev=None):
        
        linear_cache, activation_cache = cache

        dZ = activations_backward[self.activation](dA, activation_cache)

        A_prev, W, b = linear_cache
        m = A_prev.shape[1]

        dW = (1/m) * np.dot(dZ, A_prev.T) + (lamb/m)*self.W
        db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ) 
        if D_prev is not None:
            dA_prev = D_prev * dA_prev / prev_keep_prob

        return dA_prev, dW, db

    def updateLayerWeights(self, learning_rate, dW, db):
        self.W = self.W - learning_rate*dW
        self.b = self.b - learning_rate*db

    def getWeights(self):
        return self.W, self.b


class Dense(Layer):
    
    def __init__(self, n_units, activation="relu"):
        super().__init__(n_units, activation)


class Dropout(Layer):

    def __init__(self, n_units, keep_prob, activation="relu"):
        super().__init__(n_units, activation)
        self.keep_prob = keep_prob

    def printLayer(self):
        print("keep_prob: ", self.keep_prob)
        super().printLayer()

    def linearActivationForward(self, A_prev):

        A_without_droupout, cache = super().linearActivationForward(A_prev)
        self.D = np.random.randn(A_without_droupout.shape[0], A_without_droupout.shape[1])
        self.D = (self.D < self.keep_prob).astype(int)
        A = (A_without_droupout*self.D)/self.keep_prob

        return A, cache
