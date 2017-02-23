import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt

class LinearRegression(object):

    def __init__(self):
        self.param = None

    def train(self, x, y):
        x  = np.hstack([np.ones([x.shape[0], 1]), x])
        xt = x.transpose()
        
        self.param = np.linalg.inv(np.dot(xt, x)).dot(xt).dot(y)

    def predict(self, x):
        x = np.hstack([np.ones([x.shape[0], 1]), x])
        return np.dot(x, self.param)

    def rss(self, x, y):
        """ Residual Sum of Squares """
        x = np.hstack([np.ones([x.shape[0], 1]), x])
        r = y - np.dot(x, self.param)

        return np.sum(np.dot(r.transpose(), r), axis=0)


def main():
    train = pd.read_table("datasets/zipcode/zip.train", 
                          delim_whitespace=True,
                          header=None,
                          index_col=0)
    test  = pd.read_table("datasets/zipcode/zip.test", 
                          delim_whitespace=True,
                          header=None,
                          index_col=0)

    m = LinearRegression()
    m.train(train.values, pd.get_dummies(train.index).values)

    ptrain = m.predict(train.values).argmax(axis=1)
    ptest  = m.predict(test .values).argmax(axis=1)

    print("train precision: %f" % (np.sum(ptrain == train.index.values) / float(len(ptrain))))
    print("test  precision: %f" % (np.sum(ptest  == test .index.values) / float(len(ptest ))))

if __name__ == "__main__":
    main()
