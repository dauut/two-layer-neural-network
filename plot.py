import numpy as np
from nn import *

from sklearn.datasets import make_moons
from sklearn.datasets.samples_generator import make_blobs

np.random.seed(0)
# X, y = make_moons(200, noise=0.20)
X, y = make_blobs(n_samples =100, centers=3, n_features=2, random_state=0)

plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
# plt.show()
# model = build_model(X, y, 4, num_passes=20000, print_loss=True)
model = build_model_691(X, y, 3, num_passes=20000, print_loss=True)
plot_decision_boundary(lambda x: predict(model, x), X, y)
plt.title("Decision Boundary for hidden layer size 3")

plt.figure(figsize=(16, 32))
hidden_layer_dimensions = [1, 2, 3, 4]
# for i, nn_hdim in enumerate(hidden_layer_dimensions):
#     plt.subplot(5, 2, i + 1)
#     plt.title('HiddenLayerSize%d' % nn_hdim)
#     model = build_model(X, y, nn_hdim, 20000, False)
#     # model = build_model_691(X, y, nn_hdim, 20000, False)
#     plot_decision_boundary(lambda x: predict(model, x), X, y)
plt.show()