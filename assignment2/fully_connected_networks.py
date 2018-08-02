import matplotlib.pyplot as plt

from assignment2.cs231n.solver import Solver
from assignment2.cs231n.classifiers.fc_net import *
from assignment2.cs231n.data_utils import get_CIFAR10_data

plt.rcParams["figure.figsize"] = (10.0, 8.0)
plt.rcParams["image.interpolation"] = "nearest"
plt.rcParams["image.cmap"] = "gray"


def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


def train_two_layer_network(data):
    print()
    print("*****************start train two layer network*****************")

    model = TwoLayerNet()
    solver = Solver(model, data, update_rule="sgd",
                    optim_config={
                        "learning_rate": 1e-3
                    },
                    lr_decay=0.95, num_epochs=10,
                    batch_size=100, print_every=100)
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


def train_multi_layer_network(data):
    print()
    print("*****************start train multi layer network*****************")

    model = FullyConnectedNet([100, 100, 100, 100], weight_scale=1e-2, use_batchnorm=True)
    solver = Solver(model, data,
                    num_epochs=10, batch_size=100,
                    update_rule="adam",
                    optim_config={
                        "learning_rate": 0.001
                    },
                    lr_decay=0.95,
                    verbose=True)
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
    train_multi_layer_network(data)
