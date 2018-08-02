import time

import numpy as np
import matplotlib.pyplot as plt

from assignment1.cs231n.data_utils import load_CIFAR10
from assignment1.cs231n.classifiers import Softmax
from assignment1.cs231n.classifiers.softmax import softmax_loss_naive
from assignment1.cs231n.classifiers.softmax import softmax_loss_vectorized

plt.rcParams["figure.figsize"] = (10.0, 8.0)
plt.rcParams["image.interpolation"] = "nearest"
plt.rcParams["image.cmap"] = "gray"


def read_cifar_data(cifar10_dir):
    num_training = 49000
    num_validation = 1000
    num_test = 1000
    num_dev = 500

    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]

    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]

    mask = np.random.choice(num_training, num_dev, replace=False)
    X_dev = X_train[mask]
    y_dev = y_train[mask]

    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    return X_train, y_train, X_val, y_val, X_test, y_test, X_dev, y_dev


def preprocess(X_train, X_val, X_test, X_dev):
    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_val = np.reshape(X_val, (X_val.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))
    X_dev = np.reshape(X_dev, (X_dev.shape[0], -1))

    mean_image = np.mean(X_train, axis=0)

    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image
    X_dev -= mean_image

    X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
    X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
    X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
    X_dev = np.hstack([X_dev, np.ones((X_dev.shape[0], 1))])

    return X_train, X_val, X_test, X_dev


def speed_up(X_dev, y_dev):
    print()
    print("*****************start speed up*****************")

    W = np.random.randn(3073, 10) * 0.0001

    tic = time.time()
    loss_naive, grad_naive = softmax_loss_naive(W, X_dev, y_dev, 0.000005)
    toc = time.time()
    print("naive loss: %e computed in %fs" % (loss_naive, toc - tic))

    tic = time.time()
    loss_vectorized, grad_vectorized = softmax_loss_vectorized(W, X_dev, y_dev, 0.000005)
    toc = time.time()
    print("vectorized loss: %e computed in %fs" % (loss_vectorized, toc - tic))

    grad_difference = np.linalg.norm(grad_naive - grad_vectorized, ord="fro")
    print("Loss difference: %f" % np.abs(loss_naive - loss_vectorized))
    print("Gradient difference: %f" % grad_difference)


def validation(X_train, y_train, X_val, y_val):
    print()
    print("*****************start validation*****************")

    learning_rates = [1e-7, 5e-7, 1e-6]
    regularization_strengths = [1e3, 5e3, 1e4]

    results = {}
    best_val = -1
    best_softmax = None

    for learning_rate in learning_rates:
        for reg in regularization_strengths:
            softmax = Softmax()
            softmax.train(X_train, y_train, learning_rate=learning_rate, reg=reg,
                          num_iters=3000, batch_size=100, verbose=False)
            train_accuracy = np.mean(y_train == softmax.predict(X_train))
            val_accuracy = np.mean(y_val == softmax.predict(X_val))
            results[(learning_rate, reg)] = (train_accuracy, val_accuracy)
            if val_accuracy > best_val:
                best_val = val_accuracy
                best_softmax = softmax

    for lr, reg in sorted(results):
        train_accuracy, val_accuracy = results[(lr, reg)]
        print("lr %e reg %e train accuracy: %f val accuracy: %f" %
              (lr, reg, train_accuracy, val_accuracy))

    print("best validation accuracy achieved during cross-validation: %f" % best_val)

    return best_softmax


def evaluation(best_softmax, X_test, y_test):
    print()
    print("*****************start evaluation*****************")

    y_test_pred = best_softmax.predict(X_test)
    test_accuracy = np.mean(y_test == y_test_pred)
    print("softmax on raw pixels final test set accuracy: %f" % (test_accuracy,))

    w = best_softmax.W[:-1, :]  # strip out the bias
    w = w.reshape(32, 32, 3, 10)
    w_min, w_max = np.min(w), np.max(w)
    classes = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    for i in range(10):
        plt.subplot(2, 5, i + 1)

        # Rescale the weights to be between 0 and 255
        wimg = 255.0 * (w[:, :, :, i].squeeze() - w_min) / (w_max - w_min)
        plt.imshow(wimg.astype("uint8"))
        plt.axis("off")
        plt.title(classes[i])

    plt.show()


if __name__ == "__main__":
    X_train, y_train, X_val, y_val, X_test, y_test, X_dev, y_dev = read_cifar_data("cs231n/datasets/cifar-10-batches-py")
    X_train, X_val, X_test, X_dev = preprocess(X_train, X_val, X_test, X_dev)

    best_softmax = validation(X_train, y_train, X_val, y_val)
    evaluation(best_softmax, X_test, y_test)

