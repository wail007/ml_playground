import pandas as pd
import numpy  as np

from linear_regression   import LeastSquareClassification
from logistic_regression import MCLogisticRegression

np.set_printoptions(precision=4, suppress=True)

def main():
    class_id_dict = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}

    dataset = pd.read_csv("datasets/iris_species/Iris.csv", 
                          index_col=[0], 
                          converters={'Species': lambda x : class_id_dict[x]})
                          
    #dataset = dataset[dataset.Species < 2]
    dataset = dataset.sample(frac=1)
    
    train = dataset[:int(0.5*len(dataset)) ]
    test  = dataset[ int(0.5*len(dataset)):]

    xtrain, ytrain = (train.values[:,:-1], train.values[:,-1])
    xtest , ytest  = (test .values[:,:-1], test .values[:,-1])
    
    """
    max, min = np.max(xtrain, axis=0, keepdims=True), np.min(xtrain, axis=0, keepdims=True)
    xtrain = (xtrain - min) / (max - min)
    xtest  = (xtest  - min) / (max - min)
    """

    """
    mean, std = (np.mean(xtrain, axis=0, keepdims=True), np.std(xtrain, axis=0, keepdims=True))
    xtrain = (xtrain - mean) / std
    xtest  = (xtest  - mean) / std
    """

    xtrain = np.hstack([np.ones([len(xtrain), 1]), xtrain])
    xtest  = np.hstack([np.ones([len(xtest ), 1]), xtest ])

    ytrain_dummy = pd.get_dummies(train['Species']).values
    ytest_dummy  = pd.get_dummies(test ['Species']).values

    estimators = {
        "Least Square Classification"                        : LeastSquareClassification(),
        "Multi-class Logistic Regression - gradient descent" : MCLogisticRegression(solver='gradient', alpha=1e-3 , e=1e-4),
        "Multi-class Logistic Regression - newton's method"  : MCLogisticRegression(solver='newton'  , alpha=1    , e=1e-4)
    }

    for name, estimator in estimators.items():
        estimator.fit(xtrain, ytrain_dummy)

        print(name + ":")
        print("train precision: %f" % estimator.precision(xtrain, ytrain))
        print("test  precision: %f" % estimator.precision(xtest , ytest ))


if __name__ == "__main__":
    main()