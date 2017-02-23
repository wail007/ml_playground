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
    
    def precision(self, x, y):
        pass

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
    
    xtrain = train.values
    xtest  = test .values
    ytrain = pd.get_dummies(train.index).values
    ytest  = pd.get_dummies(test .index).values

    m = LinearRegression()
    m.train(xtrain,ytrain)

    ptrain = m.predict(xtrain).argmax(axis=1)
    ptest  = m.predict(xtest ).argmax(axis=1)

    print(1.0 - (np.sum(ptrain == train.index.values) / float(len(ptrain)))) 
    print(1.0 - (np.sum(ptest  == test .index.values) / float(len(ptest )))) 

    print("end")

if __name__ == "__main__":
    main()
