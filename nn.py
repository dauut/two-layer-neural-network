import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons


def calculate_loss(model, X, y):
    print()
    # todo


def predict(model, x):
    print()
    # todo


def build_model(X, y, nn_hdim, num_passes=20000, print_loss=False):
    print()
    # todo


def build_model_691(X, y, nn_hdim, num_passes=20000, print_loss=False):
    print()
    # todo


def plot_decision_boundary(pred_func):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0] + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1] + .5
    h = 0.01

    # Generate a grid of points with distance h between them

    xx, yy = np.meshgrid(np.arrange(x_min, x_max, h), np.arrange(y_min, y_max, h))

    # Predict the function value for the whole grid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)


np.random.seed(0)
X, y = make_moons(200, noise=0.20)
plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
plt.show()

exit()