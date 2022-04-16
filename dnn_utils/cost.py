import numpy as np

def binary_cross_entropy(Y_hat, Y):
    m = Y.shape[1]
    loss = Y*np.log(Y_hat) + (1-Y)*np.log(1-Y_hat)
    cost = -(1/m) * np.sum(loss)
    cost = np.squeeze(cost)
    return cost

def binary_cross_entropy_with_regularization(Y_hat, Y, W, lamb=0.1):
    m = Y.shape[1]
    cross_entropy_cost = binary_cross_entropy(Y_hat, Y)
    L2_regularization_cost = (1/m)*(lamb/2) * np.sum(np.sum(Wsquare) for Wsquare in np.square(W))
    cost = cross_entropy_cost + L2_regularization_cost
    return cost

def multiclass_cross_entropy(Y_hat, Y):
    m = Y.shape[1]
    loss = -np.sum(Y*np.log(Y_hat))
    cost = (1/m)*np.sum(loss)
    cost = np.squeeze(cost)
    return cost

def multiclass_cross_entropy_with_regularization(Y_hat, Y, W, lamb=0.1):
    m = Y.shape[1]
    cross_entropy_cost = multiclass_cross_entropy(Y_hat, Y)
    L2_regularization_cost = (1/m)*(lamb/2) * np.sum(np.sum(Wsquare) for Wsquare in np.square(W))
    cost = cross_entropy_cost + L2_regularization_cost
    return cost
