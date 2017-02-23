import pandas as pd
import numpy  as np

class KNN(object):

    def __init__(self, k=1, dummy_var=False):
        self.k = k
        self.dummy_var = dummy_var

    def train(self, x, y):
        self.x = x
        self.y = y

    def predict(self, x):
        result = np.empty([x.shape[0], self.y.shape[1]])
        for i in xrange(len(x)):
            dvec = self.x - x[i]
            dsqr = np.sum(dvec * dvec, axis=1)
            knn  = dsqr.argsort()[:self.k]
            result[i] = np.sum(self.y[knn], axis=0)
        
        return result






def main():
    train = pd.read_table("datasets/zipcode/zip.train", 
                          delim_whitespace=True,
                          header=None,
                          index_col=0)
    test  = pd.read_table("datasets/zipcode/zip.test", 
                          delim_whitespace=True,
                          header=None,
                          index_col=0)

    knn = KNN(3, True)
    knn.train(train.values, pd.get_dummies(train.index).values)

    ptrain = knn.predict(train.values).argmax(axis=1)
    ptest  = knn.predict(test .values).argmax(axis=1)

    print("train precision: %f" % (np.sum(ptrain == train.index.values) / float(len(ptrain))))
    print("test  precision: %f" % (np.sum(ptest  == test .index.values) / float(len(ptest ))))



if __name__ == "__main__":
    main()
