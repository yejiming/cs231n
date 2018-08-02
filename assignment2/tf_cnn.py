import os
import sys
import pickle
import platform
import math

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/learn")

import numpy as np
import tensorflow as tf
import tflearn
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation


def load_pickle(f):
    version = platform.python_version_tuple()
    if version[0] == "2":
        return pickle.load(f)
    elif version[0] == "3":
        return pickle.load(f, encoding="latin1")
    raise ValueError("invalid python version: {}".format(version))


def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, "rb") as f:
        datadict = load_pickle(f)
        X = datadict["data"]
        Y = datadict["labels"]
        X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
        Y = np.array(Y)
        return X, Y


def load_CIFAR10(ROOT):
    """ load all of cifar """
    xs = []
    ys = []
    for b in range(1,6):
        f = os.path.join(ROOT, "data_batch_%d" % (b, ))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, "test_batch"))
    return Xtr, Ytr, Xte, Yte


def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=10000):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the two-layer neural net classifier. These are the same steps as
    we used for the SVM, but condensed to a single function.
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = "cifar-10-batches-py"
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # Subsample the data
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


def simple_model(X, y):
    # define our weights (e.g. init_two_layer_convnet)

    # setup variables
    Wconv1 = tf.get_variable("Wconv1", shape=[7, 7, 3, 32])
    bconv1 = tf.get_variable("bconv1", shape=[32])
    W1 = tf.get_variable("W1", shape=[5408, 10])
    b1 = tf.get_variable("b1", shape=[10])

    # define our graph (e.g. two_layer_convnet)
    a1 = tf.nn.conv2d(X, Wconv1, strides=[1, 2, 2, 1], padding="VALID") + bconv1
    h1 = tf.nn.relu(a1)
    h1_flat = tf.reshape(h1, [-1, 5408])
    y_out = tf.matmul(h1_flat, W1) + b1
    return y_out


def my_model(X, y):
    y = tf.one_hot(y, 10)

    network = tflearn.conv_2d(X, nb_filter=32, filter_size=3, strides=1, activation="linear",
                              padding="valid", name="conv1", weight_decay=0.01)
    network = tflearn.batch_normalization(network, name="bn1")
    network = tflearn.relu(network)
    network = tflearn.max_pool_2d(network, kernel_size=2, strides=2, padding="same", name="pool1")

    network = tflearn.conv_2d(network, nb_filter=64, filter_size=3, strides=1, activation="linear",
                             padding="valid", name="conv2", weight_decay=0.01)
    network = tflearn.batch_normalization(network, name="bn2")
    network = tflearn.relu(network)
    network = tflearn.max_pool_2d(network, kernel_size=2, strides=2, padding="same", name="pool2")

    network = tflearn.flatten(network, name="flat1")

    network = tflearn.fully_connected(network, 1024, activation="linear", name="fc1", weight_decay=0.01)
    network = tflearn.batch_normalization(network, name="bn2")
    network = tflearn.relu(network)

    network = tflearn.fully_connected(network, 1024, activation="linear", name="fc2", weight_decay=0.01)
    network = tflearn.batch_normalization(network, name="bn3")
    network = tflearn.relu(network)

    logits = tflearn.fully_connected(network, 10, activation="softmax", name="output", weight_decay=0.01)
    loss = tflearn.categorical_crossentropy(logits, y)
    train_op = tflearn.Adam(0.0001, 0.9)().minimize(loss)

    return logits, loss, train_op


def resnet(X, y):
    y = tf.one_hot(y, 10)

    network = tflearn.conv_2d(X, 16, 3, regularizer="L2", weight_decay=0.0001)
    network = tflearn.residual_block(network, 5, 16)
    network = tflearn.residual_block(network, 1, 32, downsample=True)
    network = tflearn.residual_block(network, 5 - 1, 32)
    network = tflearn.residual_block(network, 1, 64, downsample=True)
    network = tflearn.residual_block(network, 5 - 1, 64)
    network = tflearn.batch_normalization(network)
    network = tflearn.activation(network, "relu")
    network = tflearn.global_avg_pool(network)

    logits = tflearn.fully_connected(network, 10, activation="softmax")
    loss = tflearn.categorical_crossentropy(logits, y)

    step_tensor = tf.Variable(0., name="Training_step", trainable=False)
    optimizer = tflearn.Momentum(0.1, lr_decay=0.1, decay_step=32000, staircase=True)
    optimizer.build(step_tensor)
    train_op = optimizer().minimize(loss)

    return logits, loss, train_op


def run_model(session, predict, mean_loss, train_op,
              X_train, y_train, X_val, y_val, X_test, y_test,
              epochs=1, batch_size=64, print_every=100):

    correct_prediction = tf.equal(tf.argmax(predict, 1), y)

    iter_cnt = 0
    for e in range(epochs):
        print("======================Epoch {}======================".format(e + 1))

        # shuffle indicies
        train_indicies = np.arange(X_train.shape[0])
        np.random.shuffle(train_indicies)

        tflearn.is_training(True, session=session)
        # keep track of losses and accuracy
        correct = 0
        losses = []
        # make sure we iterate over the dataset once
        for i in range(int(math.ceil(X_train.shape[0] / batch_size))):
            # generate indicies for the batch
            start_idx = (i * batch_size) % X_train.shape[0]
            idx = train_indicies[start_idx:start_idx + batch_size]

            # create a feed dictionary for this batch
            feed_dict = {X: X_train[idx, :],
                         y: y_train[idx]}
            # get batch size
            actual_batch_size = y_train[idx].shape[0]

            # have tensorflow compute loss and correct predictions
            # and (if given) perform a training step
            loss, corr, _ = session.run([mean_loss, correct_prediction, train_op], feed_dict=feed_dict)

            # aggregate performance stats
            losses.append(loss * actual_batch_size)
            correct += np.sum(corr)

            # print every now and then
            if iter_cnt % print_every == 0:
                print("Iteration {0}: with minibatch training loss = {1:.3g} and accuracy of {2:.2g}" \
                      .format(iter_cnt, loss, np.sum(corr) / float(actual_batch_size)))
            iter_cnt += 1

        total_correct = correct / float(X_train.shape[0])
        total_loss = np.sum(losses) / float(X_train.shape[0])
        print("Train: Overall loss = {0:.3g} and accuracy of {1:.3g}" \
              .format(total_loss, total_correct))

        tflearn.is_training(False, session=session)

        feed_dict = {X: X_val, y: y_val}
        val_loss, corr = session.run([mean_loss, correct_prediction], feed_dict=feed_dict)
        val_correct = np.sum(corr) / float(X_val.shape[0])
        print("Validation: Overall loss = {0:.3g} and accuracy of {1:.3g}".format(val_loss, val_correct))

        feed_dict = {X: X_test, y: y_test}
        test_loss, corr = session.run([mean_loss, correct_prediction], feed_dict=feed_dict)
        test_correct = np.sum(corr) / float(X_test.shape[0])
        print("Test: Overall loss = {0:.3g} and accuracy of {1:.3g}".format(test_loss, test_correct))


if __name__ == "__main__":
    X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()

    # clear old variables
    tf.reset_default_graph()

    # Real-time data preprocessing
    img_prep = tflearn.ImagePreprocessing()
    img_prep.add_featurewise_zero_center(per_channel=True)

    # Real-time data augmentation
    img_aug = tflearn.ImageAugmentation()
    img_aug.add_random_flip_leftright()
    img_aug.add_random_crop([32, 32], padding=4)

    # setup input (e.g. the data that changes every batch)
    # The first dim is None, and gets sets automatically based on batch size fed in
    X = tflearn.input_data(shape=[None, 32, 32, 3],
                           data_preprocessing=img_prep,
                           data_augmentation=img_aug)
    y = tf.placeholder(tf.int64, [None])

    logits, loss, train_op = my_model(X, y)

    with tf.Session() as sess:
        with tf.device("/gpu:0"):  # "/cpu:0" or "/gpu:0"
            sess.run(tf.global_variables_initializer())
            run_model(sess, logits, loss, train_op, X_train, y_train, X_val, y_val, X_test, y_test, 50, 128, 100)
