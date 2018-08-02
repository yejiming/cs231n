import numpy as np
import matplotlib.pyplot as plt

from assignment1.cs231n.data_utils import load_CIFAR10
from assignment1.cs231n.gradient_check import eval_numerical_gradient
from assignment1.cs231n.vis_utils import visualize_grid
from assignment1.cs231n.classifiers.neural_net import TwoLayerNet

plt.rcParams["figure.figsize"] = (10.0, 8.0)
plt.rcParams["image.interpolation"] = "nearest"
plt.rcParams["image.cmap"] = "gray"


def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


def show_net_weights(net):
    W1 = net.params["W1"]
    W1 = W1.reshape(32, 32, 3, -1).transpose(3, 0, 1, 2)
    plt.imshow(visualize_grid(W1, padding=3).astype("uint8"))
    plt.gca().axis("off")
    plt.show()


def visualize_loss(stats):
    plt.subplot(2, 1, 1)
    plt.plot(stats["loss_history"])
    plt.title("Loss history")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")

    plt.subplot(2, 1, 2)
    plt.plot(stats["train_acc_history"], label="train")
    plt.plot(stats["val_acc_history"], label="val")
    plt.title("Classification accuracy history")
    plt.xlabel("Epoch")
    plt.ylabel("Clasification accuracy")
    plt.legend(loc="upper right")
    plt.show()


def experiment():
    print()
    print("*****************start experiment*****************")

    def init_toy_model():
        np.random.seed(0)
        return TwoLayerNet(input_size, hidden_size, num_classes, std=1e-1)

    def init_toy_data():
        np.random.seed(1)
        X = 10 * np.random.randn(num_inputs, input_size)
        y = np.array([0, 1, 2, 2, 1])
        return X, y

    input_size = 4
    hidden_size = 10
    num_classes = 3
    num_inputs = 5

    net = init_toy_model()
    X, y = init_toy_data()

    loss, grads = net.loss(X, y, reg=0.05)

    # should be very small, we get < 1e-12
    correct_loss = 1.30378789133
    print("Difference between your loss and correct loss: {}".format(np.sum(np.abs(loss - correct_loss))))
    print()

    # these should all be less than 1e-8 or so
    for param_name in grads:
        f = lambda W: net.loss(X, y, reg=0.05)[0]
        param_grad_num = eval_numerical_gradient(f, net.params[param_name], verbose=False)
        print("%s max relative error: %e" % (param_name, rel_error(param_grad_num, grads[param_name])))
    print()

    net = init_toy_model()
    stats = net.train(X, y, X, y,
                      learning_rate=1e-1, reg=5e-6,
                      num_iters=100, verbose=False)

    print("Final training loss: ", stats["loss_history"][-1])

    # plot the loss history
    plt.plot(stats["loss_history"])
    plt.xlabel("iteration")
    plt.ylabel("training loss")
    plt.title("Training Loss history")
    plt.show()


def read_cifar_data(cifar10_dir):
    num_training = 49000
    num_validation = 1000
    num_test = 1000

    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]

    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]

    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    return X_train, y_train, X_val, y_val, X_test, y_test


def preprocess(X_train, X_val, X_test):
    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_val = np.reshape(X_val, (X_val.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))

    mean_image = np.mean(X_train, axis=0)

    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    return X_train, X_val, X_test


def tuning_hyperparameters(X_train, y_train, X_val, y_val):
    print()
    print("*****************start tuning hyperparameters*****************")

    input_size = 32 * 32 * 3
    num_classes = 10

    hidden_sizes = [200, 300]
    regularization_strengths = [0.025, 0.1]

    best_val = -1
    best_net = None
    best_stats = None

    for reg in regularization_strengths:
        for hidden_size in hidden_sizes:
            net = TwoLayerNet(input_size, hidden_size, num_classes)
            stats = net.train(X_train, y_train, X_val, y_val,
                              num_iters=3000, batch_size=100,
                              learning_rate=1e-3, learning_rate_decay=0.95,
                              reg=reg, verbose=False)
            val_acc = (net.predict(X_val) == y_val).mean()
            print("When reg is {}, h_size is {}, Validation accuracy: {}"
                  .format(reg, hidden_size, val_acc))
            if val_acc > best_val:
                best_val = val_acc
                best_net = net
                best_stats = stats

    return best_net, best_stats



if __name__ == "__main__":
    X_train, y_train, X_val, y_val, X_test, y_test = read_cifar_data("cs231n/datasets/cifar-10-batches-py")
    X_train, X_val, X_test = preprocess(X_train, X_val, X_test)

    best_net, best_stats = tuning_hyperparameters(X_train, y_train, X_val, y_val)
    visualize_loss(best_stats)
    show_net_weights(best_net)

    test_acc = (best_net.predict(X_test) == y_test).mean()
    print("Test accuract: {}".format(test_acc))
