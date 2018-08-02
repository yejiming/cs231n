import numpy as np
from random import shuffle


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
        that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    num_classes = W.shape[1]
    num_train = X.shape[0]
    for i in range(num_train):
        scores = X[i].dot(W)
        num = np.exp(scores[y[i]])
        denom = 0.
        for j in range(num_classes):
            denom += np.exp(scores[j])
        for j in range(num_classes):
            if j != y[i]:
                dW[:, j] += X[i] * np.exp(scores[j]) / denom
            else:
                dW[:, j] += -X[i] * (1 - np.exp(scores[j]) / denom)
        loss += -np.log(num / denom)

    loss /= num_train
    loss += reg * np.sum(W * W)

    dW /= num_train
    dW += 2 * reg * W

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    num_classes = W.shape[1]
    num_train = X.shape[0]

    y_bin = np.eye(num_classes)[y].astype(bool)

    exp_scores = np.exp(X.dot(W))
    num = exp_scores[y_bin]
    denom = np.sum(exp_scores, axis=1)

    loss = np.sum(-np.log(num / denom)) / num_train + reg * np.sum(W * W)

    gradients = exp_scores / denom.reshape(-1, 1)
    gradients[y_bin] -= 1
    dW = np.dot(X.T, gradients) / num_train + 2 * reg * W

    return loss, dW
