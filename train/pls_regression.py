import numpy as np
from sklearn.cross_decomposition import PLSRegression
from .utils import rmse

def pls_regression(train, test, label):
    pr = PLSRegression(copy=True, max_iter=500, n_components=2, scale=True, tol=1e-06)
    pr.fit(train, label.values)

    y_prediction = pr.predict(train)
    y_test = label
    print("PLS Regression score on training set: ", rmse(y_test, y_prediction))
    y_prediction = pr.predict(test)
    y_prediction = np.exp(y_prediction)
    return y_prediction