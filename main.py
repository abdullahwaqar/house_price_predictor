from preprocess import refactored_data_frames, read_csv
from visualize import plot_predictions
from train import pls_regression, mlp_classifier, svm

combined_train, combined_test, label = refactored_data_frames()

# y_prediction_pls = pls_regression(combined_train.fillna(0) ,combined_test.fillna(0),label)

y_prediction_mlp = mlp_classifier(combined_train.fillna(0) ,combined_test.fillna(0),label)

# y_prediction_svm = svm(combined_train.fillna(0) ,combined_test.fillna(0),label)
plot_predictions(read_csv('data/train.csv')['SalePrice'][1000:], y_prediction_mlp[1000:])