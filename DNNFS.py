import math
import numpy as np
from dnn_utils.layers import *  
from dnn_utils.activations import *
from dnn_utils.optimizers import *
from dnn_utils.cost import *

class DNNFS:
    
    def __init__(self):
        self.__layers = []
        self.__grads = {}

    def addLayer(self, layer, input_size=None):
        l = len(self.__layers)

        if input_size:
            n_units_prev = input_size
        else:
            n_units_prev = self.__layers[l-1].n_units
    
        layer.initializeWeights(n_units_prev)

        self.__layers.append(layer)

    def printNetwork(self):
        for l, layer in enumerate(self.__layers):
            print("Layer " + str(l+1) + ":")
            layer.printLayer()
            print("--")

    def normalizeInputs(self, X):
        m = X.shape[1]
        average = (1/m) * np.sum(X, axis=1, keepdims=True)
        variance = np.sqrt((1/m)*np.sum(np.square(X), axis=1, keepdims=True))
        X_normalized = (X - average)/variance
        return X_normalized

    def randomizeMiniBatches(self, X, Y, mini_batch_size=64, seed=0):        
        np.random.seed(seed)

        m = X.shape[1]
        mini_batches = []

        # Shuffle (X, Y)
        permutation = list(np.random.permutation(m))
        shuffled_X = X[:, permutation]
        shuffled_Y = Y[:, permutation].reshape((1,m))

        # Partition (shuffled_X, shuffled_Y)
        num_complete_minibatches = math.floor(m / mini_batch_size)
        for k in range(0, num_complete_minibatches):
            mini_batch_X = shuffled_X[:, mini_batch_size*k : mini_batch_size*(k+1)]
            mini_batch_Y = shuffled_Y[:, mini_batch_size*k : mini_batch_size*(k+1)]

            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

        if m % mini_batch_size != 0:
            mini_batch_X = shuffled_X[:, num_complete_minibatches*mini_batch_size:]
            mini_batch_Y = shuffled_Y[:, num_complete_minibatches*mini_batch_size:]

            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

        return mini_batches

    def forwardPropagation(self, X):
        caches = list()
        weights = list()

        A = X
        L = len(self.__layers)        

        for l in range(0, L):
            A_prev = A
            A, cache = self.__layers[l].linearActivationForward(A_prev)
            caches.append(cache)

            W = self.__layers[l].W
            b = self.__layers[l].b
            weights.append(W)

        AL = A
        return AL, caches, weights

    def backwardPropagation(self, AL, Y, caches, lamb):
        grads = dict()
        L = len(caches)
        m = AL.shape[1]
        Y = Y.reshape(AL.shape)

        dAL = -(np.divide(Y, AL) - np.divide(1-Y, 1-AL))
        grads["dA" + str(L)] = dAL

        for l in reversed(range(L)):
            current_cache = caches[l]
            if (isinstance(self.__layers[l-1], Dropout)):
                keep_prob_prev = self.__layers[l-1].keep_prob
                D_prev = self.__layers[l-1].D
                dA_prev_temp, dW_temp, db_temp = self.__layers[l].linearActivationBackward(grads["dA"+str(l+1)], current_cache, lamb, keep_prob_prev, D_prev)
            else:
                dA_prev_temp, dW_temp, db_temp = self.__layers[l].linearActivationBackward(grads["dA"+str(l+1)], current_cache, lamb)
            grads["dA" + str(l)] = dA_prev_temp
            grads["dW" + str(l+1)] = dW_temp
            grads["db" + str(l+1)] = db_temp
        
        self.__grads = grads
        return grads

    def updateWeights(self, learning_rate):
        L = len(self.__layers)
        
        for l in range(L):
            dW = self.__grads["dW" + str(l+1)]
            db = self.__grads["db" + str(l+1)]
            self.__layers[l].updateLayerWeights(learning_rate, dW, db)

    def train(self, X, Y, learning_rate, num_epochs=2000, mini_batch_size=64, 
              optimizer="momentum", normalize_inputs=True, lamb=0.1,
              print_cost_interval=100, print_cost=True):

        if normalize_inputs:
            X = self.normalizeInputs(X)

        if optimizer == "momentum":
            optimizer = Momentum(self.__layers, beta=0.9)
            optimizer.initializeOptimizerParameters()

        seed = 10
        costs = []
        for i in range(num_epochs):

            seed = seed + 1
            minibatches = self.randomizeMiniBatches(X, Y, mini_batch_size, seed)
            cost_total = 0

            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                AL, caches, weights = self.forwardPropagation(minibatch_X)
                grads = self.backwardPropagation(AL, minibatch_Y, caches, lamb)
                if optimizer == "momentum":
                    optimizer.updateParametersWithOptimizer(learning_rate, grads)
                else:
                    self.updateWeights(learning_rate)
                if lamb != 0:
                    minibatch_cost = binary_cross_entropy_with_regularization(AL, minibatch_Y, weights, lamb)
                else:
                    minibatch_cost = binary_cross_entropy(AL, minibatch_Y)
                cost_total += minibatch_cost

            if i % print_cost_interval == 0 or i == num_epochs-1:
                costs.append(cost_total)
                if print_cost:
                    print("Cost after iteration {}: {}".format(i, np.squeeze(cost_total)))
        
        return costs
