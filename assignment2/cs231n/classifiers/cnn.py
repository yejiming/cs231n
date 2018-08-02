import numpy as np

from assignment2.cs231n.layers import *
from assignment2.cs231n.fast_layers import *
from assignment2.cs231n.layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        C, H, W = input_dim
        stride, pad = 1, (filter_size - 1) // 2
        conv_dim = (int((H + 2 * pad - filter_size) / stride) + 1, int((W + 2 * pad - filter_size) / stride) + 1)
        pool_dim = (int((conv_dim[0] - 2) / 2) + 1, int((conv_dim[1] - 2) / 2) + 1)

        self.params["W1"] = weight_scale * np.random.randn(num_filters, C, filter_size, filter_size)
        self.params["b1"] = np.zeros(num_filters)

        self.params["W2"] = weight_scale * np.random.randn(np.prod(pool_dim)*num_filters, hidden_dim)
        self.params["b2"] = np.zeros(hidden_dim)
        self.params["W3"] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params["b3"] = np.zeros(num_classes)

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]
        W3, b3 = self.params["W3"], self.params["b3"]

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {"stride": 1, "pad": (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {"pool_height": 2, "pool_width": 2, "stride": 2}

        h1, h1_cache = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
        h2, h2_cache = affine_relu_forward(h1, W2, b2)
        scores, o_cache = affine_forward(h2, W3, b3)

        if y is None:
            return scores

        loss, dout = softmax_loss(scores, y)
        dx3, dW3, db3 = affine_backward(dout, o_cache)
        dx2, dW2, db2 = affine_relu_backward(dx3, h2_cache)
        dx1, dW1, db1 = conv_relu_pool_backward(dx2, h1_cache)

        loss += 0.5 * (self.reg * np.square(W1).sum() + self.reg * np.square(W2).sum() + self.reg * np.square(W3).sum())
        grads = {
            "W1": dW1 + self.reg * W1, "b1": db1,
            "W2": dW2 + self.reg * W2, "b2": db2,
            "W3": dW3 + self.reg * W3, "b3": db3
        }

        return loss, grads
