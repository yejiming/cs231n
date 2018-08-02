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


def visualize_dropout(data):
    # Train two identical nets, one with dropout and one without
    np.random.seed(231)
    num_train = 500
    small_data = {
        "X_train": data["X_train"][:num_train],
        "y_train": data["y_train"][:num_train],
        "X_val": data["X_val"],
        "y_val": data["y_val"],
    }

    solvers = {}
    dropout_choices = [0, 0.75]
    for dropout in dropout_choices:
        model = FullyConnectedNet([100], dropout=dropout)
        print(dropout)

        solver = Solver(model, small_data,
                        num_epochs=25, batch_size=100,
                        update_rule="adam",
                        optim_config={
                            "learning_rate": 5e-4,
                        },
                        verbose=True, print_every=100)
        solver.train()
        solvers[dropout] = solver

    # Plot train and validation accuracies of the two models
    train_accs = []
    val_accs = []
    for dropout in dropout_choices:
        solver = solvers[dropout]
        train_accs.append(solver.train_acc_history[-1])
        val_accs.append(solver.val_acc_history[-1])

    plt.subplot(3, 1, 1)
    for dropout in dropout_choices:
        plt.plot(solvers[dropout].train_acc_history, "o", label="%.2f dropout" % dropout)
    plt.title("Train accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(ncol=2, loc="lower right")

    plt.subplot(3, 1, 2)
    for dropout in dropout_choices:
        plt.plot(solvers[dropout].val_acc_history, "o", label="%.2f dropout" % dropout)
    plt.title("Val accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(ncol=2, loc="lower right")

    plt.gcf().set_size_inches(15, 15)
    plt.show()


if __name__ == "__main__":
    data = get_CIFAR10_data()
    visualize_dropout(data)
