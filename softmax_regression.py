import numpy as np
import scipy.linalg

import solver
from activation import softmax, softmax_log

class SoftmaxRegression(object):
    def __init__(self, solver='gradient', alpha=1e-3, e=1e-3, verbose=False):
        self.w = None
        self.e = e
        self.verbose = verbose
        self.solver = solver
        self.alpha = alpha

    def predict(self, x):
        w = np.reshape(self.w, (x.shape[1], -1), 'F')
        return np.argmax(softmax(np.dot(x, w)), axis=1)

    def fit(self, x, t):
        self.w = np.zeros([x.shape[1]*t.shape[1]])

        if self.solver == 'gradient':
            solver.gradient_descent(self, x, t, self.alpha, self.e, self.verbose)
        elif self.solver == 'newton':
            solver.newton(self, x, t, self.e, self.verbose)

    def cost(self, x, t):
        w = np.reshape(self.w, (x.shape[1], -1), 'F')
        y_log = softmax_log(np.dot(x, w))
        return -np.sum(t * y_log)

    def gradient(self, x, t):
        w = np.reshape(self.w, (x.shape[1], -1), 'F')
        y = softmax(np.dot(x, w))
        return np.reshape(np.dot(np.transpose(x), y - t), -1, 'F')

    def hessian(self, x, t):
        k = t.shape[1]
        n = t.shape[0]
        d = x.shape[1]

        w = np.reshape(self.w, (x.shape[1], -1), 'F')
        y = softmax(np.dot(x, w))
        
        h = np.zeros([d*k, d*k])
        for i in xrange(k):
            for j in xrange(k):
                h[i*d:(i+1)*d,j*d:(j+1)*d] = np.dot(np.transpose(x) * (y[:,i] * ((i==j) - y[:,j])), x)

        return h

    
    def precision(self, x, t):
        y = self.predict(x)
        return (1.0 / len(y)) * np.sum(y == t)
