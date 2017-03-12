import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

class _LinearModel(object):
    
    def __init__(self):
        self.w = None

    def fit(self, x, y):
        pass

    def predict(self, x):
        x = np.hstack([np.ones([len(x), 1]), x])
        return np.dot(x, self.w)
    
    def cost(self, x, y):
        pass

    def precision(self, x, y):
        p = self.predict(x)
        return (1.0 / len(p)) * np.sum(p == y)



class LeastSquareRegression(_LinearModel):
    def __init__(self):
        super(LeastSquareRegression, self).__init__()

    def fit(self, x, y):
        x  = np.hstack([np.ones([x.shape[0], 1]), x])
        self.w = np.linalg.pinv(x).dot(y)

    def cost(self, x, y):
        """ Residual Sum of Squares """
        x = np.hstack([np.ones([len(x), 1]), x])
        r = y - np.dot(x, self.w)
        rt= np.transpose(r)

        return (1.0 / len(x)) * np.trace(np.dot(rt, r))


class RidgeRegression(LeastSquareRegression):
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
        xt = np.transpose(x)

        self.w = np.linalg.inv(np.dot(xt, x) + alpha * np.eye(x.shape[1])).dot(xt).dot(y)
        bias   = np.mean(y, axis=0, keepdims=True) - np.dot(np.mean(x, axis=0, keepdims=True), self.w)
    
        self.w = np.vstack([bias, self.w])



class LeastSquareClassification(LeastSquareRegression):
    def __init__(self):
        super(LeastSquareClassification, self).__init__()

    def predict(self, x):
        return super(LeastSquareClassification, self).predict(x).argmax(axis=1)


class RidgeClassification(RidgeRegression):
    def __init__(self, incr=0.1, min_change=0.001):
        super(RidgeClassification, self).__init__(incr, min_change)

    def predict(self, x):
        return super(RidgeClassification, self).predict(x).argmax(axis=1)


class LDAClassification(_LinearModel):
    def __init__(self):
        self.w      = None
        self.priors = None
        self.means  = []
        self.covs   = []

    def fit(self, x, y):
        k           = y.shape[1]
        y_arg       = np.argmax(y, axis=1)
        class_count = np.sum (y, axis=0, keepdims=True)

        self.priors = (1.0 / len(y)) * np.sum (y, axis=0, keepdims=True)
        self.w      = self._lda(x, y)

        x_proj = np.dot(x, self.w)
        means  = (1.0 / class_count.T) * np.dot(y.T, x_proj)
        for i in xrange(k):
            xk_proj = x_proj[y_arg==i]
            self.means.append(np.mean(xk_proj, axis  =    0))
            self.covs .append(np.cov (xk_proj, rowvar=False))

    def predict(self, x):
        k      = self.w.shape[1]
        x_proj = np.dot(x, self.w)

        likelihood = np.column_stack([multivariate_normal.pdf(x_proj, self.means[i], self.covs[i]) for i in xrange(k)])
        posterior  = (likelihood * self.priors)
        posterior  = posterior / np.sum(posterior, axis=1, keepdims=True)

        return np.argmax(posterior, axis=1)


    def _lda(self, x, y):
        k     = y.shape[1]
        y_arg = np.argmax(y, axis=1)

        class_count= np.sum (y, axis=0, keepdims=True)
        total_mean = np.mean(x, axis=0, keepdims=True)
        class_mean = (1.0 / class_count.T) * np.dot(y.T, x)
        
        mk_m  = class_mean - total_mean
        b_cov = np.dot(class_count * mk_m.T, mk_m)
        
        w_cov = np.zeros(b_cov.shape)
        for i in xrange(k):
            xk    = x[y_arg == i]
            xk_mk = xk - class_mean[i]
            w_cov += np.dot(xk_mk.T, xk_mk)

        eig_vals, eig_vecs = np.linalg.eig(np.dot(np.linalg.pinv(w_cov), b_cov))
        eig_vals = np.abs(eig_vals)

        eig_args = np.argsort(eig_vals)[::-1][:k]
        return eig_vecs[:, eig_args]
