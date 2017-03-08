import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt

class _LinearModel(object):
    
    def __init__(self):
        self.w = None
        self.alpha = 0

    def predict(self, x):
        x = np.hstack([np.ones([x.shape[0], 1]), x])
        return np.dot(x, self.w)
    
    def cost(self, x, y):
        """ Residual Sum of Squares """
        x = np.hstack([np.ones([x.shape[0], 1]), x])
        r = y - np.dot(x, self.w)

        return np.trace(np.dot(r.transpose(), r)) / float(x.shape[0])


class LinearRegression(_LinearModel):
    def __init__(self):
        super(LinearRegression, self).__init__()

    def fit(self, x, y):
        x  = np.hstack([np.ones([x.shape[0], 1]), x])
        self.w = np.linalg.pinv(x).dot(y)


class RidgeRegression(_LinearModel):
    def __init__(self, incr=0.1, min_change=0.001):
        super(RidgeRegression, self).__init__()
        self.incr       = incr
        self.min_change = min_change

    def fit(self, x, y):
        xtrain, xval = np.split(x, [int(0.7*len(x))])
        ytrain, yval = np.split(y, [int(0.7*len(y))])
        
        alpha      = 0.0
        best_alpha = 0.0
        best_cost   = float("inf")
        old_cost    = float("inf")
        new_cost    = float("inf") 

        while True:
            self._fit(xtrain, ytrain, alpha)

            new_cost = self.cost(xval, yval)
            if new_cost < best_cost:
                best_cost   = new_cost
                best_alpha = alpha
                print("cost: %f, alpha: %f" % (best_cost, best_alpha))

            if abs(new_cost - old_cost) < self.min_change:
                break

            old_cost = new_cost
            
            alpha += self.incr
        
        self._fit(xtrain, ytrain, best_alpha)
            

    def _fit(self, x, y, alpha):
        xt = x.transpose()

        self.w = np.linalg.inv(np.dot(xt, x) + alpha * np.eye(x.shape[1])).dot(xt).dot(y)
        bias   = np.mean(y, axis=0, keepdims=True) - np.dot(np.mean(x, axis=0, keepdims=True), self.w)
    
        self.w = np.vstack([bias, self.w])
