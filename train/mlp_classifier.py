from sklearn.neural_network import MLPClassifier

def mlp_classifier(train, test, label):
    lab_enc = preprocessing.LabelEncoder()
    training_encoded = lab_enc.fit_transform(label)
    mlp = MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
              beta_1=0.9, beta_2=0.999, early_stopping=False,
              epsilon=1e-08, hidden_layer_sizes=(5, 2),
              learning_rate='constant', learning_rate_init=0.001,
              max_iter=200, momentum=0.9, n_iter_no_change=10,
              nesterovs_momentum=True, power_t=0.5, random_state=1,
              shuffle=True, solver='lbfgs', tol=0.0001,
              validation_fraction=0.1, verbose=False, warm_start=False)
    mlp.fit(train, training_encoded)

    y_prediction = mlp.predict(train)
    y_test = label
    print("MLP score on training set: ", rmse(y_test, y_prediction))
    y_prediction = mlp.predict(test)
    y_prediction = np.exp(y_prediction)
    return y_prediction