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

    m.param = np.linalg.inv(np.dot(xt, x) + alpha * np.eye(x.shape[0])).dot(xt).dot(y)
    m.param = np.hstack([np.mean(y, axis=0, keepdims=True), m.param])


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

    m.param = np.random.random([257,10])
    print(m.rss(train.values, pd.get_dummies(train.index).values))
    print(m.rss(test .values, pd.get_dummies(test .index).values))    

    train_least_squares(m, train.values, pd.get_dummies(train.index).values)
    print(m.rss(train.values, pd.get_dummies(train.index).values))
    print(m.rss(test .values, pd.get_dummies(test .index).values))

    ptrain = m.predict(train.values).argmax(axis=1)
    ptest  = m.predict(test .values).argmax(axis=1)

    print("train precision: %f" % (np.sum(ptrain == train.index.values) / float(len(ptrain))))
    print("test  precision: %f" % (np.sum(ptest  == test .index.values) / float(len(ptest ))))

if __name__ == "__main__":
    main()
