import numpy as np
import pandas as pd
from sklearn import utils
from sklearn.preprocessing import LabelEncoder

def factorize(data_frame, factorize_data_frame, column, fill_na = None):
    """
    Impure factorize function
    """
    le = LabelEncoder()
    factorize_data_frame[column] = data_frame[column]
    if fill_na is not None:
        factorize_data_frame[column].fillna(fill_na,inplace=True)
    le.fit(factorize_data_frame[column].unique())
    factorize_data_frame[column] = le.transform(factorize_data_frame[column])
    return factorize_data_frame

def refactor(refactored_data_frame, data_frame, column, fill_na):
    """
    Impure converter categorical features
    """
    refactored_data_frame[column] = data_frame[column]
    if fill_na is not None:
        refactored_data_frame.fillna(fill_na,inplace=True)

    dummies = pd.get_dummies(refactored_data_frame[column],prefix="_" +column)
    refactored_data_frame = refactored_data_frame.join(dummies)
    refactored_data_frame = refactored_data_frame.drop([column], axis = 1)
    return refactored_data_frame

def read_csv(filename):
    return pd.read_csv(filename)