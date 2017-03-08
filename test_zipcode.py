import pandas as pd
import numpy  as np
from linear_regression import *
from knn import *

def main():
    train = pd.read_table("datasets/zipcode/zip.train", 
                          delim_whitespace=True,
                          header=None,
                          index_col=0)
    test  = pd.read_table("datasets/zipcode/zip.test", 
                          delim_whitespace=True,
                          header=None,
                          index_col=0)

    estimators = {
        "LinearRegression" : LinearRegression(),
        "RidgeRegression"  : RidgeRegression(incr=0.1, min_change=0.00001),
        #"KNN"              : KNN(3)
    }

    xtrain = np.hstack([train.values, train.values*train.values])
    ytrain = train.index.values

    xtest = np.hstack([test.values, test.values*test.values])
    ytest = test.index.values

    ytrain_dummy = pd.get_dummies(train.index).values
    ytest_dummy  = pd.get_dummies(test .index).values

    for name, estimator in estimators.items():
        estimator.fit(xtrain, ytrain_dummy)

        ptrain = estimator.predict(xtrain).argmax(axis=1)        
        ptest  = estimator.predict(xtest ).argmax(axis=1)

        print(name + ":")
        print("train precision: %f" % (np.sum(ptrain == ytrain) / float(len(ptrain))))
        print("test  precision: %f" % (np.sum(ptest  == ytest ) / float(len(ptest ))))

    
if __name__ == "__main__":
    main()