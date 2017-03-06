import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt

class LinearRegression(object):

    def __init__(self):
        self.param = None

    def predict(self, x):
        x = np.hstack([np.ones([x.shape[0], 1]), x])
        return np.dot(x, self.param)

    def rss(self, x, y):
        """ Residual Sum of Squares """
        x = np.hstack([np.ones([x.shape[0], 1]), x])
        r = y - np.dot(x, self.param)

        return np.trace(np.dot(r.transpose(), r)) / float(x.shape[0])

def train_least_squares(m, x, y):
    x  = np.hstack([np.ones([x.shape[0], 1]), x])
    xt = x.transpose()

    m.param = np.linalg.inv(np.dot(xt, x)).dot(xt).dot(y)

def train_ridge(m, x, y, alpha):
    xt = x.transpose()

    m.param = np.linalg.inv(np.dot(xt, x) + alpha * np.eye(x.shape[1])).dot(xt).dot(y)
    bias    = np.mean(y, axis=0, keepdims=True) - np.dot(np.mean(x, axis=0, keepdims=True), m.param)
    
    m.param = np.vstack([bias, m.param])

def train_lasso(m, x, y, alpha):
    pass
