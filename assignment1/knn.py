import numpy as np
import matplotlib.pyplot as plt

from assignment1.cs231n.data_utils import load_CIFAR10
from assignment1.cs231n.classifiers import KNearestNeighbor

plt.rcParams["figure.figsize"] = (10.0, 8.0)
plt.rcParams["image.interpolation"] = "nearest"
plt.rcParams["image.cmap"] = "gray"


def time_function(f, *args):
    """
    Call a function f with args and return the time (in seconds) that it took to execute.
    """
    import time
    tic = time.time()
    f(*args)
    toc = time.time()
    return toc - tic


def show_examples(X_train, y_train):
    print()
    print("*****************start show examples*****************")
    classes = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    num_classes = len(classes)
    samples_per_class = 7
    for y, cls in enumerate(classes):
        idxs = np.flatnonzero(y_train == y)
        idxs = np.random.choice(idxs, samples_per_class, replace=False)
        for i, idx in enumerate(idxs):
            plt_idx = i * num_classes + y + 1
            plt.subplot(samples_per_class, num_classes, plt_idx)
            plt.imshow(X_train[idx].astype("uint8"))
            plt.axis("off")
            if i == 0:
                plt.title(cls)
    plt.show()


def speed_up(X_train, y_train, X_test):
    print()
    print("*****************start speed up*****************")
    classifier = KNearestNeighbor()
    classifier.train(X_train, y_train)

    two_loop_time = time_function(classifier.compute_distances_two_loops, X_test)
    print("Two loop version took %f seconds" % two_loop_time)

    one_loop_time = time_function(classifier.compute_distances_one_loop, X_test)
    print("One loop version took %f seconds" % one_loop_time)

    no_loop_time = time_function(classifier.compute_distances_no_loops, X_test)
    print("No loop version took %f seconds" % no_loop_time)


def cross_validation(X_train, y_train):
    print()
    print("*****************start cross validation*****************")

    num_folds = 5
    k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 30]
    k_to_accuracies = {}

    indices = np.arange(len(y_train))
    index_folds = np.array_split(indices, num_folds)
    X_train_folds = [X_train[fold] for fold in index_folds]
    y_train_folds = [y_train[fold] for fold in index_folds]

    for k in k_choices:
        k_to_accuracies[k] = []
        for i in range(num_folds):
            X_small_train = np.vstack(X_train_folds[j] for j in range(num_folds) if i != j)
            y_small_train = np.hstack(y_train_folds[j] for j in range(num_folds) if i != j)

            classifier = KNearestNeighbor()
            classifier.train(X_small_train, y_small_train)
            pred = classifier.predict(X_train_folds[i], k=k, num_loops=0)
            num_correct = np.sum(pred == y_train_folds[i])
            accuracy = float(num_correct) / len(y_train_folds[i])

            k_to_accuracies[k].append(accuracy)

    # Print out the computed accuracies
    for k in sorted(k_to_accuracies):
        for accuracy in k_to_accuracies[k]:
            print("k = %d, accuracy = %f" % (k, accuracy))

    # plot the raw observations
    for k in k_choices:
        accuracies = k_to_accuracies[k]
        plt.scatter([k] * len(accuracies), accuracies)

    # plot the trend line with error bars that correspond to standard deviation
    accuracies_mean = np.array([np.mean(v) for k, v in sorted(k_to_accuracies.items())])
    accuracies_std = np.array([np.std(v) for k, v in sorted(k_to_accuracies.items())])
    plt.errorbar(k_choices, accuracies_mean, yerr=accuracies_std)
    plt.title("Cross-validation on k")
    plt.xlabel("k")
    plt.ylabel("Cross-validation accuracy")
    plt.show()


def evaluation(X_train, y_train, X_test, y_test):
    print()
    print("*****************start evaluation*****************")
    best_k = 10

    classifier = KNearestNeighbor()
    classifier.train(X_train, y_train)
    y_test_pred = classifier.predict(X_test, k=best_k)

    # Compute and display the accuracy
    num_correct = np.sum(y_test_pred == y_test)
    accuracy = float(num_correct) / num_test
    print("Got %d / %d correct => accuracy: %f" % (num_correct, num_test, accuracy))


if __name__ == "__main__":
    cifar10_dir = "cs231n/datasets/cifar-10-batches-py"
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    num_training = 5000
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]

    num_test = 500
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]

    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))

    print("Training data shape: ", X_train.shape)
    print("Training labels shape: ", y_train.shape)
    print("Test data shape: ", X_test.shape)
    print("Test labels shape: ", y_test.shape)

    evaluation(X_train, y_train, X_test, y_test)
