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
        "RidgeRegression"  : RidgeRegression(incr=0.7, min_change=0.0000001),
        "KNN"              : KNN(3)
    }
    
    for name, estimator in estimators.items():
        estimator.fit(train.values, pd.get_dummies(train.index).values)

        ptrain = estimator.predict(train.values).argmax(axis=1)        
        ptest  = estimator.predict(test .values).argmax(axis=1)

        print(name + ":")
        print("train precision: %f" % (np.sum(ptrain == train.index.values) / float(len(ptrain))))
        print("test  precision: %f" % (np.sum(ptest  == test .index.values) / float(len(ptest ))))

    
if __name__ == "__main__":
    main()