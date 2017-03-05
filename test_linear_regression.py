import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
import linear_regression as lr

def test_zipcode_least_squares():
    train = pd.read_table("datasets/zipcode/zip.train", 
                          delim_whitespace=True,
                          header=None,
                          index_col=0)
    test  = pd.read_table("datasets/zipcode/zip.test", 
                          delim_whitespace=True,
                          header=None,
                          index_col=0)
    
    m = lr.LinearRegression()
    lr.train_least_squares(m, train.values, pd.get_dummies(train.index).values)

    ptrain = m.predict(train.values).argmax(axis=1)
    ptest  = m.predict(test .values).argmax(axis=1)

    print("train precision: %f" % (np.sum(ptrain == train.index.values) / float(len(ptrain))))
    print("test  precision: %f" % (np.sum(ptest  == test .index.values) / float(len(ptest ))))

def test_zipcode_ridge():
    train = pd.read_table("datasets/zipcode/zip.train", 
                          delim_whitespace=True,
                          header=None,
                          index_col=0)
    test  = pd.read_table("datasets/zipcode/zip.test", 
                          delim_whitespace=True,
                          header=None,
                          index_col=0)

    #train = train.sample(frac=1)

    xtrain, xval = np.split(train.values      , [int(0.7*len(train))])
    ytrain, yval = np.split(train.index.values, [int(0.7*len(train))])
    
    xtest = test.values
    ytest = test.index.values

    xtrain = (xtrain - np.mean(xtrain, axis=0)) / np.std(xtrain, axis=0)
    xval   = (xval   - np.mean(xval  , axis=0)) / np.std(xval  , axis=0)
    xtest  = (xtest  - np.mean(xtest , axis=0)) / np.std(xtest , axis=0)

    ytrain_dummy = pd.get_dummies(ytrain).values

    m = lr.LinearRegression()
    """
    alpha = 0.0
    max_precision = 0
    max_alpha = 0
    while True:
        lr.train_ridge(m, xtrain, ytrain_dummy, alpha)
        
        pval = m.predict(xval).argmax(axis=1)

        precision = (np.sum(pval == yval) / float(len(pval)))

        if precision > max_precision:
            max_precision = precision
            max_alpha= alpha
            print("precision: %f, alpha: %f" % (max_precision, max_alpha))

        alpha = alpha + 0.1
    """
    lr.train_ridge(m, xtrain, ytrain_dummy, 209.4)
    
    ptrain = m.predict(xtrain).argmax(axis=1)
    pval   = m.predict(xval  ).argmax(axis=1)
    ptest  = m.predict(xtest ).argmax(axis=1)
    
    print("train precision: %f" % (np.sum(ptrain == ytrain) / float(len(ptrain))))
    print("val   precision: %f" % (np.sum(pval   == yval  ) / float(len(pval  ))))    
    print("test  precision: %f" % (np.sum(ptest  == ytest ) / float(len(ptest ))))


def main():
    test_zipcode_ridge()
    test_zipcode_least_squares()


if __name__ == "__main__":
    main()