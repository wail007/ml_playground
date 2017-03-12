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
        #"LeastSquareClassification": LeastSquareClassification(),
        #"RidgeClassification"      : RidgeClassification(incr=0.1, min_change=0.000001),
        "LDAClassification"        : LDAClassification(),
        #"KNN"                      : KNN(3)
    }

    xtrain = np.hstack([train.values, train.values*train.values])
    ytrain = train.index.values

    xtest = np.hstack([test.values, test.values*test.values])
    ytest = test.index.values

    ytrain_dummy = pd.get_dummies(train.index).values
    ytest_dummy  = pd.get_dummies(test .index).values

    for name, estimator in estimators.items():
        estimator.fit(xtrain, ytrain_dummy)

        print(name + ":")
        print("train precision: %f" % estimator.precision(xtrain, ytrain))
        print("test  precision: %f" % estimator.precision(xtest , ytest ))
        """
        p = estimator.predict(xtest)
        wrong_arg = p != ytest

        wrong_x = test.values[wrong_arg]
        wrong_y = ytest[wrong_arg]
        wrong_p = p[wrong_arg]

        wrong_x = wrong_x.reshape([len(wrong_x), 16, -1])

        for i in xrange(len(wrong_y)):
            print("Expected: %d, Predicted: %d" % (wrong_y[i], wrong_p[i]))
            imgplot = plt.imshow(wrong_x[i], cmap='gray')
            plt.show()  
        """

    
if __name__ == "__main__":
    main()