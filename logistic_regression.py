import numpy  as np
import solver as slvr
from activation import sigmoid, sigmoid_log


class LogisticRegression(object):
    def __init__(self, solver='gradient', reg=None, alpha=1e-3, e=1e-3, verbose=False):
        self.w = None
        self.e = e
        self.verbose = verbose
        self.alpha = alpha

        if solver == 'newton':
            self.solver = slvr.Newton(alpha, e, verbose)
        else:
            self.solver = slvr.GradientDescent(alpha, e, verbose)

        if reg:
            self.solver = slvr.ValidationSet(self.solver, 0.7, False, e, verbose)

    def fit(self, x, t):
        self.w = np.zeros(x.shape[1])
        self.solver.solve(self, x, t)

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

    def _cost(self, x, t):
        a = np.dot(x, self.w)
        return -np.sum(t * sigmoid_log(a) + (1 - t) * sigmoid_log(-a), axis=0, keepdims=True)

    def _gradient(self, x, t):
        y = sigmoid(np.dot(x, self.w))
        return np.dot(np.transpose(x), y - t)

    def _hessian(self, x, t):
        y = sigmoid(np.dot(x, self.w))
        return np.dot(np.transpose(x) * (y * (1 - y)), x)

    def _cost_L2(self, x, t, reg):
        return self._cost(x, t) + reg * np.dot(self.w[1:], self.w[1:]) 

    def _gradient_L2(self, x, t, reg):
        g = self._gradient(x, t)
        g[1:] += 2.0 * reg * self.w[1:]
        return g

    

    
    
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