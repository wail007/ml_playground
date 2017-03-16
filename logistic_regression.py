import numpy as np
import solver
from activation import sigmoid, sigmoid_log


class LogisticRegression(object):
    def __init__(self, solver='gradient', alpha=1e-3, e=1e-3, verbose=False):
        self.w = None
        self.e = e
        self.verbose = verbose
        self.solver = solver
        self.alpha = alpha

    def fit(self, x, t):
        self.w = np.zeros(x.shape[1])

        if self.solver == 'gradient':
            solver.gradient_descent(self, x, t, self.alpha, self.e, self.verbose)
        elif self.solver == 'newton':
            solver.newton(self, x, t, self.e, self.verbose)

    def predict(self, x):
        return np.rint(self.probability(x))

    def probability(self, x):
        return sigmoid(np.dot(x, self.w))

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

    
class MCLogisticRegression(object):
    def __init__(self, solver='gradient', alpha=1e-3, e=1e-3, verbose=False):
        self.bin_logits = []
        self.e = e
        self.verbose = verbose
        self.solver = solver
        self.alpha = alpha

    def fit(self, x, t):
        for i in xrange(t.shape[1]):
            if self.verbose:
                print("Class: %d" % i)
            self.bin_logits.append(LogisticRegression(self.solver, self.alpha, self.e, self.verbose))
            self.bin_logits[i].fit(x, t[:,i])

    def predict(self, x):
        return np.argmax(self.probability(x), axis=1) 

    def probability(self, x):
        p = self.bin_logits[0].probability(x)
        for i in xrange(1, len(self.bin_logits)):
            p = np.vstack([p, self.bin_logits[i].probability(x)])

        return np.transpose(p)
    
    def precision(self, x, t):
        y = self.predict(x)
        return (1.0 / len(y)) * np.sum(y == t)