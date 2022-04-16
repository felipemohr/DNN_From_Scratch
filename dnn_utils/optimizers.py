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

    def __init__(self, layers, beta=0.9):
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


class RMSProp(Optimizer):

    def __init__(self, layers, beta=0.99):
        super().__init__(layers)
        self.beta = beta
        self.s = dict()
        self.initializeOptimizerParameters()

    def initializeOptimizerParameters(self):
        L = len(self.layers)

        for l in range(1, L+1):
            layer_shape_dW = self.parameters_shape["dW" + str(l)]
            layer_shape_db = self.parameters_shape["db" + str(l)]

            self.s["dW" + str(l)] = np.zeros((layer_shape_dW[0], layer_shape_dW[1]))
            self.s["db" + str(l)] = np.zeros((layer_shape_db[0], layer_shape_db[1]))

    def updateParametersWithOptimizer(self, learning_rate, grads):
        L = len(self.layers)
        epsilon = 1E-8

        for l in range(1, L+1):
            self.s["dW" + str(l)] = self.beta*self.s["dW" + str(l)] + (1 - self.beta)*grads["dW" + str(l)]**2
            self.s["db" + str(l)] = self.beta*self.s["db" + str(l)] + (1 - self.beta)*grads["db" + str(l)]**2

            dW = grads["dW" + str(l)] / (np.sqrt(self.s["dW" + str(l)]) + epsilon)
            db = grads["db" + str(l)] / (np.sqrt(self.s["db" + str(l)]) + epsilon)
            
            self.layers[l-1].updateLayerWeights(learning_rate, dW, db)


class Adam(Optimizer):

    def __init__(self, layers, beta1=0.9, beta2=0.99):
        super().__init__(layers)
        self.beta1 = beta1
        self.beta2 = beta2
        self.v = dict()
        self.s = dict()
        self.t = 0
        self.initializeOptimizerParameters()

    def initializeOptimizerParameters(self):
        L = len(self.layers)

        for l in range(1, L+1):
            layer_shape_dW = self.parameters_shape["dW" + str(l)]
            layer_shape_db = self.parameters_shape["db" + str(l)]

            self.v["dW" + str(l)] = np.zeros((layer_shape_dW[0], layer_shape_dW[1]))
            self.v["db" + str(l)] = np.zeros((layer_shape_db[0], layer_shape_db[1]))

            self.s["dW" + str(l)] = np.zeros((layer_shape_dW[0], layer_shape_dW[1]))
            self.s["db" + str(l)] = np.zeros((layer_shape_db[0], layer_shape_db[1]))

    def updateParametersWithOptimizer(self, learning_rate, grads):
        L = len(self.layers)
        epsilon = 1E-8

        self.t += 1
        for l in range(1, L+1):

            self.v["dW" + str(l)] = self.beta1*self.v["dW" + str(l)] + (1 - self.beta1)*grads["dW" + str(l)]
            self.v["db" + str(l)] = self.beta1*self.v["db" + str(l)] + (1 - self.beta1)*grads["db" + str(l)]

            self.s["dW" + str(l)] = self.beta2*self.s["dW" + str(l)] + (1 - self.beta2)*grads["dW" + str(l)]**2
            self.s["db" + str(l)] = self.beta2*self.s["db" + str(l)] + (1 - self.beta2)*grads["db" + str(l)]**2

            Vdw_corrected = self.v["dW" + str(l)] / (1 - self.beta1**self.t)
            Vdb_corrected = self.v["db" + str(l)] / (1 - self.beta1**self.t)

            Sdw_corrected = self.s["dW" + str(l)] / (1 - self.beta2**self.t)
            Sdb_corrected = self.s["db" + str(l)] / (1 - self.beta2**self.t)

            dW = np.divide(Vdw_corrected, np.sqrt(Sdw_corrected) + epsilon)
            db = np.divide(Vdb_corrected, np.sqrt(Sdb_corrected) + epsilon)
            
            self.layers[l-1].updateLayerWeights(learning_rate, dW, db)

