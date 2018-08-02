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


def visualize_batch_normalization(data):
    print()
    print("*****************start visualizing batch normalization*****************")

    np.random.seed(231)
    # Try training a very deep net with batchnorm
    hidden_dims = [50, 50, 50, 50, 50, 50, 50]

    num_train = 1000
    small_data = {
        "X_train": data["X_train"][:num_train],
        "y_train": data["y_train"][:num_train],
        "X_val": data["X_val"],
        "y_val": data["y_val"],
    }

    bn_solvers = {}
    solvers = {}
    weight_scales = np.logspace(-4, 0, num=20)
    for i, weight_scale in enumerate(weight_scales):
        print("Running weight scale %d / %d" % (i + 1, len(weight_scales)))
        bn_model = FullyConnectedNet(hidden_dims, weight_scale=weight_scale, use_batchnorm=True)
        model = FullyConnectedNet(hidden_dims, weight_scale=weight_scale, use_batchnorm=False)

        bn_solver = Solver(bn_model, small_data,
                           num_epochs=10, batch_size=50,
                           update_rule="adam",
                           optim_config={
                               "learning_rate": 1e-3
                           },
                           verbose=False, print_every=200)
        bn_solver.train()
        bn_solvers[weight_scale] = bn_solver

        solver = Solver(model, small_data,
                        num_epochs=10, batch_size=50,
                        update_rule="adam",
                        optim_config={
                            "learning_rate": 1e-3
                        },
                        verbose=False, print_every=200)
        solver.train()
        solvers[weight_scale] = solver

    # Plot results of weight scale experiment
    best_train_accs, bn_best_train_accs = [], []
    best_val_accs, bn_best_val_accs = [], []
    final_train_loss, bn_final_train_loss = [], []

    for ws in weight_scales:
        best_train_accs.append(max(solvers[ws].train_acc_history))
        bn_best_train_accs.append(max(bn_solvers[ws].train_acc_history))

        best_val_accs.append(max(solvers[ws].val_acc_history))
        bn_best_val_accs.append(max(bn_solvers[ws].val_acc_history))

        final_train_loss.append(np.mean(solvers[ws].loss_history[-100:]))
        bn_final_train_loss.append(np.mean(bn_solvers[ws].loss_history[-100:]))

    plt.subplot(3, 1, 1)
    plt.title("Best val accuracy vs weight initialization scale")
    plt.xlabel("Weight initialization scale")
    plt.ylabel("Best val accuracy")
    plt.semilogx(weight_scales, best_val_accs, "-o", label="baseline")
    plt.semilogx(weight_scales, bn_best_val_accs, "-o", label="batchnorm")
    plt.legend(ncol=2, loc="lower right")

    plt.subplot(3, 1, 2)
    plt.title("Best train accuracy vs weight initialization scale")
    plt.xlabel("Weight initialization scale")
    plt.ylabel("Best training accuracy")
    plt.semilogx(weight_scales, best_train_accs, "-o", label="baseline")
    plt.semilogx(weight_scales, bn_best_train_accs, "-o", label="batchnorm")
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.title("Final training loss vs weight initialization scale")
    plt.xlabel("Weight initialization scale")
    plt.ylabel("Final training loss")
    plt.semilogx(weight_scales, final_train_loss, "-o", label="baseline")
    plt.semilogx(weight_scales, bn_final_train_loss, "-o", label="batchnorm")
    plt.legend()
    plt.gca().set_ylim(1.0, 3.5)

    plt.gcf().set_size_inches(10, 15)
    plt.show()


if __name__ == "__main__":
    data = get_CIFAR10_data()
    visualize_batch_normalization(data)
