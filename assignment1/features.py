import numpy as np
import matplotlib.pyplot as plt

from assignment1.cs231n.features import *
from assignment1.cs231n.data_utils import load_CIFAR10
from assignment1.cs231n.classifiers.linear_classifier import LinearSVM
from assignment1.cs231n.classifiers.neural_net import TwoLayerNet

plt.rcParams["figure.figsize"] = (10.0, 8.0)
plt.rcParams["image.interpolation"] = "nearest"
plt.rcParams["image.cmap"] = "gray"


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


def feature_extraction(X_train, X_val, X_test):
    num_color_bins = 10  # Number of bins in the color histogram
    feature_fns = [hog_feature, lambda img: color_histogram_hsv(img, nbin=num_color_bins)]
    X_train_feats = extract_features(X_train, feature_fns, verbose=True)
    X_val_feats = extract_features(X_val, feature_fns)
    X_test_feats = extract_features(X_test, feature_fns)

    # Preprocessing: Subtract the mean feature
    mean_feat = np.mean(X_train_feats, axis=0, keepdims=True)
    X_train_feats -= mean_feat
    X_val_feats -= mean_feat
    X_test_feats -= mean_feat

    # Preprocessing: Divide by standard deviation.
    std_feat = np.std(X_train_feats, axis=0, keepdims=True)
    X_train_feats /= std_feat
    X_val_feats /= std_feat
    X_test_feats /= std_feat

    # Preprocessing: Add a bias dimension
    X_train_feats = np.hstack([X_train_feats, np.ones((X_train_feats.shape[0], 1))])
    X_val_feats = np.hstack([X_val_feats, np.ones((X_val_feats.shape[0], 1))])
    X_test_feats = np.hstack([X_test_feats, np.ones((X_test_feats.shape[0], 1))])

    return X_train_feats, X_val_feats, X_test_feats


def train_svm(X_train_feats, y_train, X_val_feats, y_val, X_test, X_test_feats, y_test):
    print()
    print("*****************start train svm*****************")

    learning_rates = [1e-8, 5e-8, 1e-7]
    regularization_strengths = [5e5]

    results = {}
    best_val = -1  # The highest validation accuracy that we have seen so far.
    best_svm = None  # The LinearSVM object that achieved the highest validation rate.

    for learning_rate in learning_rates:
        for reg in regularization_strengths:
            svm = LinearSVM()
            svm.train(X_train_feats, y_train, learning_rate=learning_rate, reg=reg,
                      num_iters=2000, verbose=False)
            train_accuracy = np.mean(y_train == svm.predict(X_train_feats))
            val_accuracy = np.mean(y_val == svm.predict(X_val_feats))
            results[(learning_rate, reg)] = (train_accuracy, val_accuracy)
            if val_accuracy > best_val:
                best_val = val_accuracy
                best_svm = svm

    for lr, reg in sorted(results):
        train_accuracy, val_accuracy = results[(lr, reg)]
        print("lr %e reg %e train accuracy: %f val accuracy: %f" %
              (lr, reg, train_accuracy, val_accuracy))

    print("best validation accuracy achieved during cross-validation: %f" % best_val)
    
    y_test_pred = best_svm.predict(X_test_feats)
    test_accuracy = np.mean(y_test == y_test_pred)
    print("Test accuract: {}".format(test_accuracy))

    examples_per_class = 8
    classes = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    for cls, cls_name in enumerate(classes):
        idxs = np.where((y_test != cls) & (y_test_pred == cls))[0]
        idxs = np.random.choice(idxs, examples_per_class, replace=False)
        for i, idx in enumerate(idxs):
            plt.subplot(examples_per_class, len(classes), i * len(classes) + cls + 1)
            plt.imshow(X_test[idx].astype("uint8"))
            plt.axis("off")
            if i == 0:
                plt.title(cls_name)
    plt.show()


def train_neural_net(X_train_feats, y_train, X_val_feats, y_val, X_test_feats, y_test):
    print()
    print("*****************start train neural net*****************")

    input_size = X_train_feats.shape[1]
    hidden_size = 300
    num_classes = 10

    learning_rates = [0.1, 0.25, 0.5]
    regularization_strengths = [0.01, 0.005, 0.001]
    num_iters = [3000, 4000, 5000]

    results = {}
    best_val = -1
    best_net = None

    for learning_rate in learning_rates:
        for reg in regularization_strengths:
            for num in num_iters:
                net = TwoLayerNet(input_size, hidden_size, num_classes)
                net.train(X_train_feats, y_train, X_val_feats, y_val,
                          num_iters=num, batch_size=400,
                          learning_rate=learning_rate, learning_rate_decay=0.95,
                          reg=reg, verbose=False)
                train_accuracy = np.mean(y_train == net.predict(X_train_feats))
                val_accuracy = np.mean(y_val == net.predict(X_val_feats))
                results[(learning_rate, reg, num)] = (train_accuracy, val_accuracy)
                if val_accuracy > best_val:
                    best_val = val_accuracy
                    best_net = net

    for lr, reg, num in sorted(results):
        train_accuracy, val_accuracy = results[(lr, reg, num)]
        print("lr %e reg %e num %e train accuracy: %f val accuracy: %f" %
              (lr, reg, num, train_accuracy, val_accuracy))

    test_accuracy = np.mean(y_test == best_net.predict(X_test_feats))
    print("Test accuract: {}".format(test_accuracy))


if __name__ == "__main__":
    X_train, y_train, X_val, y_val, X_test, y_test = read_cifar_data("cs231n/datasets/cifar-10-batches-py")
    X_train_feats, X_val_feats, X_test_feats = feature_extraction(X_train, X_val, X_test)
    print(X_train_feats.shape)

    train_neural_net(X_train_feats, y_train, X_val_feats, y_val, X_test_feats, y_test)
