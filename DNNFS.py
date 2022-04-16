import numpy as np
from dnn_utils.layers import *  
from dnn_utils.activations import *
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

    def train(self, X, Y, learning_rate, num_iterations=2000, lamb=0.1, print_cost_interval=100, print_cost=True):

        costs = []
        for i in range(0, num_iterations):
            AL, caches, weights = self.forwardPropagation(X)
            grads = self.backwardPropagation(AL, Y, caches, lamb)
            self.updateWeights(learning_rate)

            if i % print_cost_interval == 0 or i == num_iterations-1:
                if lamb != 0:
                    cost = binary_cross_entropy_with_regularization(AL, Y, weights, lamb)
                else:
                    cost = binary_cross_entropy(AL, Y)
                costs.append(cost)
                if print_cost:
                    print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
                    
        
        return costs
