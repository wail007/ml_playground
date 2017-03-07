import numpy as np

class KNN(object):

    def __init__(self, k=1):
        self.k = k

    def fit(self, x, y):
        self.x = x
        self.y = y

    def predict(self, x):
        result = np.empty([x.shape[0], self.y.shape[1]])
        for i in xrange(len(x)):
            dvec = self.x - x[i]
            dsqr = np.sum(dvec * dvec, axis=1)
            knn  = dsqr.argsort()[:self.k]
            result[i] = np.sum(self.y[knn], axis=0) / float(self.k)
        
        return result
