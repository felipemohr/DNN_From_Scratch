import numpy as np

class Optimizer():

    def __init__(self, layers):
        self.layers = layers
        self.parameters_shape = {}
        self.initializeParameters()

    def initializeParameters(self):
        L = len(self.layers)

        for l in range(1, L+1):
            self.parameters_shape["dW" + str(l)] = self.layers[l-1].getWeights()[0].shape
            self.parameters_shape["db" + str(l)] = self.layers[l-1].getWeights()[1].shape

class Momentum(Optimizer):

    def __init__(self, layers, beta):
        super().__init__(layers)
        self.beta = beta
        self.v = dict()
        self.initializeOptimizerParameters()

    def initializeOptimizerParameters(self):
        L = len(self.layers)

        for l in range(1, L+1):
            layer_shape_dW = self.parameters_shape["dW" + str(l)]
            layer_shape_db = self.parameters_shape["db" + str(l)]
            self.v["dW" + str(l)] = np.zeros((layer_shape_dW[0], layer_shape_dW[1]))
            self.v["db" + str(l)] = np.zeros((layer_shape_db[0], layer_shape_db[1]))

    def updateParametersWithOptimizer(self, learning_rate, grads):
        L = len(self.layers)

        for l in range(1, L+1):
            self.v["dW" + str(l)] = self.beta*self.v["dW" + str(l)] + (1 - self.beta)*grads["dW" + str(l)]
            self.v["db" + str(l)] = self.beta*self.v["db" + str(l)] + (1 - self.beta)*grads["db" + str(l)]
            
            self.layers[l-1].updateLayerWeights(learning_rate, self.v["dW" + str(l)], self.v["db" + str(l)])

