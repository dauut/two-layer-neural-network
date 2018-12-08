import numpy as np
import matplotlib.pyplot as plt


# Loss Calculation with current model
def calculate_loss(model, X, y):
    number_of_examples = len(X)
    weight1, weight2 = model['w1'], model['w2']
    bias1, bias2 = model['b1'], model['b2']

    a = X.dot(weight1) + bias1
    h = np.tanh(a)
    z = h.dot(weight2) + bias2
    y_pred = np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)  # softmax
    loss = (-1. / number_of_examples) * np.sum(np.log(y_pred[range(number_of_examples), y]))

    return loss


def predict(model, X):
    weight1, weight2 = model['w1'], model['w2']
    bias1, bias2 = model['b1'], model['b2']

    a = X.dot(weight1) + bias1
    h = np.tanh(a)
    z = h.dot(weight2) + bias2
    y_pred = np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)  # softmax

    return np.argmax(y_pred, axis=1)


def build_model(X, y, nn_hdim, num_passes=20000, print_loss=False):
    model = {}
    input_dimensions = 2
    output_dimensions = 2
    weight1 = np.random.randn(input_dimensions, nn_hdim)
    weight2 = np.random.randn(nn_hdim, output_dimensions)
    bias1 = np.zeros((1, nn_hdim))
    bias2 = np.zeros((1, output_dimensions))
    learning_rate = 0.02

    for i in range(num_passes):
        # prediction
        a = X.dot(weight1) + bias1
        h = np.tanh(a)
        z = h.dot(weight2) + bias2
        y_pred = np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)  # softmax

        # dL/dy = y_hat - y
        dLdy = y_pred
        for j in range(len(y_pred)):
            # if the label = 0 then y_hat - y apply for first output node
            if y[j] == 0:
                dLdy[j, 0] = y_pred[j, 0] - 1
                dLdy[j, 1] = y_pred[j, 1]
            # if the label = 1 then second output, y_hat - y
            else:
                dLdy[j, 0] = y_pred[j, 0]
                dLdy[j, 1] = y_pred[j, 1] - 1

        # derivatives
        # dL/da
        dLda = (1 - (np.tanh(a) * np.tanh(a))) * dLdy.dot(weight2.T)
        # dL/dWeight2
        dLdw2 = (h.T).dot(dLdy)
        # dL/dbias2
        dLdb2 = np.sum(dLdy, axis=0)
        # dL/dweight1
        dLdw1 = (X.T).dot(dLda)
        # dL/dbias1
        dLdb1 = np.sum(dLda, axis=0)

        weight1 = weight1 + (dLdw1 * (-learning_rate))
        weight2 = weight2 + (dLdw2 * (-learning_rate))
        bias1 = bias1 + (dLdb1 * (-learning_rate))
        bias2 = bias2 + (dLdb2 * (-learning_rate))

        model = {'w1': weight1, 'b1': bias1, 'w2': weight2, 'b2': bias2}

        if print_loss and i % 1000 == 0:
            print("Iteration : ", i, "Loss : ", calculate_loss(model, X, y))
    return model


def build_model_691(X, y, nn_hdim, num_passes=20000, print_loss=False):
    model = {}
    input_dimensions = 2
    output_dimensions = 3
    weight1 = np.random.randn(input_dimensions, nn_hdim)
    weight2 = np.random.randn(nn_hdim, output_dimensions)
    bias1 = np.zeros((1, nn_hdim))
    bias2 = np.zeros((1, output_dimensions))
    learning_rate = 0.02

    for i in range(num_passes):
        # prediction

        a = X.dot(weight1) + bias1
        h = np.tanh(a)
        z = h.dot(weight2) + bias2
        y_pred = np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)  # softmax

        # dL/dy = y_hat - y
        dLdy = y_pred

        # we have 3 output
        for j in range(len(y_pred)):
            # if the label = 0 then y_hat - y apply for first output node
            if y[j] == 0:
                dLdy[j, 0] = y_pred[j, 0] - 1
                dLdy[j, 1] = y_pred[j, 1]
                dLdy[j, 2] = y_pred[j, 2]
            # if the label = 1 then y_hat - y apply for second output node
            elif y[j] == 1:
                dLdy[j, 0] = y_pred[j, 0]
                dLdy[j, 1] = y_pred[j, 1] - 1
                dLdy[j, 2] = y_pred[j, 2]
            # if the label = 2 then y_hat - y apply for third output node
            else:
                dLdy[j, 0] = y_pred[j, 0]
                dLdy[j, 1] = y_pred[j, 1]
                dLdy[j, 2] = y_pred[j, 2] - 1

        # derivatives
        # dL/da
        dLda = (1 - (np.tanh(a) * np.tanh(a))) * dLdy.dot(weight2.T)
        # dL/dWeight2
        dLdw2 = (h.T).dot(dLdy)
        # dL/dbias2
        dLdb2 = np.sum(dLdy, axis=0)
        # dL/dweight1
        dLdw1 = (X.T).dot(dLda)
        # dL/dbias1
        dLdb1 = np.sum(dLda, axis=0)

        weight1 = weight1 + (dLdw1 * (-learning_rate))
        weight2 = weight2 + (dLdw2 * (-learning_rate))
        bias1 = bias1 + (dLdb1 * (-learning_rate))
        bias2 = bias2 + (dLdb2 * (-learning_rate))

        model = {'w1': weight1, 'b1': bias1, 'w2': weight2, 'b2': bias2}

        if print_loss and i % 1000 == 0:
            print("Iteration : ", i, "Loss : ", calculate_loss(model, X, y))
    return model


def plot_decision_boundary(pred_func, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01

    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
