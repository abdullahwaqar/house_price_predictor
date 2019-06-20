from preprocess import refactored_data_frames, read_csv
from train import pls_regression, mlp_classifier, svm
from visualize import scatter_plot, plot_predictions

combined_train, combined_test, label = refactored_data_frames()

# y_prediction_pls = pls_regression(combined_train.fillna(0) ,combined_test.fillna(0),label)

y_prediction_mlp = mlp_classifier(combined_train.fillna(0) ,combined_test.fillna(0),label)

# print(combined_train.head())
# print(label.head())
# print(combined_test['SalePrice'].tail())

# y_prediction_svm = svm(combined_train.fillna(0) ,combined_test.fillna(0), label)
print(label.head())
scatter_plot(label[:1000]['SalePrice'], y_prediction_mlp[:1000])