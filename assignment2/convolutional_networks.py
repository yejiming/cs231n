from __future__ import print_function

import matplotlib.pyplot as plt

from assignment2.cs231n.solver import Solver
from assignment2.cs231n.classifiers.cnn import *
from assignment2.cs231n.data_utils import get_CIFAR10_data

plt.rcParams["figure.figsize"] = (10.0, 8.0)
plt.rcParams["image.interpolation"] = "nearest"
plt.rcParams["image.cmap"] = "gray"


def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


def train_three_layer_network(data):
    print()
    print("*****************start train three layer network*****************")

    model = ThreeLayerConvNet(weight_scale=0.001, hidden_dim=500, reg=0.001, filter_size=3, num_filters=4)

    solver = Solver(model, data,
                    num_epochs=5, batch_size=50,
                    update_rule="adam",
                    optim_config={
                        "learning_rate": 1e-3,
                    },
                    verbose=True, print_every=20)
    solver.train()

    plt.subplot(2, 1, 1)
    plt.title("Training loss")
    plt.plot(solver.loss_history, "o")
    plt.xlabel("Iteration")

    plt.subplot(2, 1, 2)
    plt.title("Accuracy")
    plt.plot(solver.train_acc_history, "-o", label="train")
    plt.plot(solver.val_acc_history, "-o", label="val")
    plt.plot([0.5] * len(solver.val_acc_history), "k--")
    plt.xlabel("Epoch")
    plt.legend(loc="lower right")
    plt.gcf().set_size_inches(15, 12)
    plt.show()

    y_test_pred = np.argmax(model.loss(data["X_test"]), axis=1)
    y_val_pred = np.argmax(model.loss(data["X_val"]), axis=1)
    print("Validation set accuracy: ", (y_val_pred == data["y_val"]).mean())
    print("Test set accuracy: ", (y_test_pred == data["y_test"]).mean())


if __name__ == "__main__":
    data = get_CIFAR10_data()
    train_three_layer_network(data)
