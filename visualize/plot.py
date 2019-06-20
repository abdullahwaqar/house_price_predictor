import math
import numpy as np
import matplotlib.pyplot as plt

def plot_predictions(actual, prediction, title='Sales Price vs Prediction', y_label='Price USD', x_label='Number of Samples'):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    #* Add labels
    plt.ylabel(y_label)
    plt.xlabel(x_label)

    plt.plot(actual, '#00FF00', label='Orignal Sale Price')
    plt.plot(prediction, '#4286f4', label='Predicted Sale Price')

    ax.set_title(title)
    ax.legend(loc='upper left')
    plt.show()

def scatter_plot(actual, prediction, title='Sales Price vs Prediction', y_label='Price USD', x_label='Number of Samples'):
    print(np.amax(prediction))
    plt.scatter(actual, actual['Id'], label='skitscat', color='k', s=25, marker="o")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Interesting Graph\nCheck it out')
    plt.legend()
    plt.show()