from sklearn.svm import SVC

def svm(train, test, label):
    lab_enc = preprocessing.LabelEncoder()
    training_encoded = lab_enc.fit_transform(label)
    svm_init = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
    svm_init.fit(train, training_encoded)

    y_prediction = svm_init.predict(train)
    y_test = label
    print("SVM score on training: ", rmse(y_test, y_prediction))
    y_prediction = svm_init.predict(test)
    y_prediction = np.exp(y_prediction)
    return y_prediction