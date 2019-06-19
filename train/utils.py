import numpy as np
from sklearn.metrics import mean_squared_error

def rmse(y,y_prediction):
    return np.sqrt(mean_squared_error(y,y_prediction))