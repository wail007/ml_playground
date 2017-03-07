import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_olivetti_faces
from sklearn.utils.validation import check_random_state

from linear_regression import *
from knn import *


def main():
    # Load the faces datasets
    data = fetch_olivetti_faces()
    targets = data.target

    data = data.images.reshape((len(data.images), -1))
    train = data[targets < 30]
    test = data[targets >= 30]  # Test on independent people

    # Test on a subset of people
    n_faces = 5
    rng = check_random_state(4)
    face_ids = rng.randint(test.shape[0], size=(n_faces, ))
    test = test[face_ids, :]

    n_pixels = data.shape[1]
    X_train, y_train = np.split(train, [int(0.5 * n_pixels)], axis=1)
    X_test , y_test  = np.split(test , [int(0.5 * n_pixels)], axis=1)

    # Fit estimators
    ESTIMATORS = {
        "LinearRegression": LinearRegression(),
        "RidgeRegression" : RidgeRegression(incr=0.3, min_change=0.1),
        "knn"             : KNN(k=5)
    }

    y_test_predict = dict()
    for name, estimator in ESTIMATORS.items():
        estimator.fit(X_train, y_train)
        y_test_predict[name] = estimator.predict(X_test)

    # Plot the completed faces
    image_shape = (64, 64)

    n_cols = 1 + len(ESTIMATORS)
    plt.figure(figsize=(2. * n_cols, 2.26 * n_faces))
    plt.suptitle("Face completion with multi-output estimators", size=16)

    for i in range(n_faces):
        true_face = np.hstack((X_test[i], y_test[i]))

        if i:
            sub = plt.subplot(n_faces, n_cols, i * n_cols + 1)
        else:
            sub = plt.subplot(n_faces, n_cols, i * n_cols + 1,
                            title="true faces")


        sub.axis("off")
        sub.imshow(true_face.reshape(image_shape),
                cmap=plt.cm.gray,
                interpolation="nearest")

        for j, est in enumerate(sorted(ESTIMATORS)):
            completed_face = np.hstack((X_test[i], y_test_predict[est][i]))

            if i:
                sub = plt.subplot(n_faces, n_cols, i * n_cols + 2 + j)

            else:
                sub = plt.subplot(n_faces, n_cols, i * n_cols + 2 + j,
                                title=est)

            sub.axis("off")
            sub.imshow(completed_face.reshape(image_shape),
                    cmap=plt.cm.gray,
                    interpolation="nearest")

    plt.show()


if __name__ == "__main__":
    main()