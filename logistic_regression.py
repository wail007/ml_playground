import numpy as np

import solver
from activation import sigmoid, sigmoid_log


class LogisticRegression(object):
    def __init__(self):
        self.w = None

    def fit(self, x, t):
        self.w = np.zeros([x.shape[1]])
        solver.newton(self, x, t)

    def predict(self, x):
        return np.rint(sigmoid(np.dot(x, self.w)))

    def cost(self, x, t):
        a = np.dot(x, self.w)
        return -np.sum(t * sigmoid_log(a) + (1 - t) * sigmoid_log(-a), axis=0, keepdims=True)

    def gradient(self, x, t):
        y = sigmoid(np.dot(x, self.w))
        return np.dot(np.transpose(x), y - t) 

    def hessian(self, x, t):
        y = sigmoid(np.dot(x, self.w))
        return np.dot(np.transpose(x) * (y * (1 - y)), x)

    def precision(self, x, t):
        y = self.predict(x)
        return (1.0 / len(y)) * np.sum(y == t)