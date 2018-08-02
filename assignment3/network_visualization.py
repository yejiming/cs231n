import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.ndimage.filters import gaussian_filter1d

from assignment3.cs231n.classifiers.squeezenet import SqueezeNet
from assignment3.cs231n.image_utils import preprocess_image, deprocess_image
from assignment3.cs231n.image_utils import SQUEEZENET_MEAN, SQUEEZENET_STD
from assignment3.cs231n.data_utils import load_imagenet_val

plt.rcParams["figure.figsize"] = (10.0, 8.0) # set default size of plots
plt.rcParams["image.interpolation"] = "nearest"
plt.rcParams["image.cmap"] = "gray"


def get_session():
    """Create a session that dynamically allocates memory."""
    # See: https://www.tensorflow.org/tutorials/using_gpu#allowing_gpu_memory_growth
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    return session


def blur_image(X, sigma=1):
    X = gaussian_filter1d(X, sigma, axis=1)
    X = gaussian_filter1d(X, sigma, axis=2)
    return X


def compute_saliency_maps(X, y, model):
    """
    Compute a class saliency map using the model for images X and labels y.

    Input:
    - X: Input images, numpy array of shape (N, H, W, 3)
    - y: Labels for X, numpy of shape (N,)
    - model: A SqueezeNet model that will be used to compute the saliency map.

    Returns:
    - saliency: A numpy array of shape (N, H, W) giving the saliency maps for the
    input images.
    """
    # Compute the score of the correct class for each example.
    # This gives a Tensor with shape [N], the number of examples.
    #
    # Note: this is equivalent to scores[np.arange(N), y] we used in NumPy
    # for computing vectorized losses.
    correct_scores = tf.gather_nd(model.classifier,
                                  tf.stack((tf.range(X.shape[0]), model.labels), axis=1))

    gradient = tf.gradients(correct_scores, model.image)
    gradient = tf.reduce_sum(tf.abs(gradient), axis=0)
    gradient = tf.reduce_max(gradient, axis=3)

    saliency = sess.run(gradient, feed_dict={model.image: X, model.labels: y})
    return saliency


def show_saliency_maps(model, X, y, mask):
    mask = np.asarray(mask)
    Xm = X[mask]
    ym = y[mask]

    saliency = compute_saliency_maps(Xm, ym, model)

    for i in range(mask.size):
        plt.subplot(2, mask.size, i + 1)
        plt.imshow(deprocess_image(Xm[i]))
        plt.axis("off")
        plt.title(class_names[ym[i]])
        plt.subplot(2, mask.size, mask.size + i + 1)
        plt.title(mask[i])
        plt.imshow(saliency[i], cmap=plt.cm.hot)
        plt.axis("off")
        plt.gcf().set_size_inches(10, 4)
    plt.show()


def make_fooling_image(X, target_y, model):
    """
    Generate a fooling image that is close to X, but that the model classifies
    as target_y.

    Inputs:
    - X: Input image, of shape (1, 224, 224, 3)
    - target_y: An integer in the range [0, 1000)
    - model: Pretrained SqueezeNet model

    Returns:
    - X_fooling: An image that is close to X, but that is classifed as target_y
    by the model.
    """
    X_fooling = X.copy()
    learning_rate = 1.

    y_pred = sess.run(model.classifier, feed_dict={model.image: X_fooling})
    y_pred = np.argmax(y_pred[0])

    while y_pred != target_y:
        score = model.classifier[0, target_y]

        gradient = tf.gradients(score, model.image)[0]
        gradient = sess.run(gradient, feed_dict={model.image: X_fooling})

        X_fooling += learning_rate * gradient

        y_pred = sess.run(model.classifier, feed_dict={model.image: X_fooling})
        y_pred = np.argmax(y_pred[0])

    return X_fooling


def show_fooling_image(idx=0, target_y=0):
    Xi = X[idx][None]
    X_fooling = make_fooling_image(Xi, target_y, model)

    # Make sure that X_fooling is classified as y_target
    scores = sess.run(model.classifier, {model.image: X_fooling})
    assert scores[0].argmax() == target_y, "The network is not fooled!"

    # Show original image, fooling image, and difference
    orig_img = deprocess_image(Xi[0])
    fool_img = deprocess_image(X_fooling[0])
    # Rescale
    plt.subplot(1, 4, 1)
    plt.imshow(orig_img)
    plt.axis("off")
    plt.title(class_names[y[idx]])
    plt.subplot(1, 4, 2)
    plt.imshow(fool_img)
    plt.title(class_names[target_y])
    plt.axis("off")
    plt.subplot(1, 4, 3)
    plt.title("Difference")
    plt.imshow(deprocess_image((Xi - X_fooling)[0]))
    plt.axis("off")
    plt.subplot(1, 4, 4)
    plt.title("Magnified difference (10x)")
    plt.imshow(deprocess_image(10 * (Xi - X_fooling)[0]))
    plt.axis("off")
    plt.gcf().tight_layout()

    plt.show()


def create_class_visualization(target_y, model, **kwargs):
    """
    Generate an image to maximize the score of target_y under a pretrained model.

    Inputs:
    - target_y: Integer in the range [0, 1000) giving the index of the class
    - model: A pretrained CNN that will be used to generate the image

    Keyword arguments:
    - l2_reg: Strength of L2 regularization on the image
    - learning_rate: How big of a step to take
    - num_iterations: How many iterations to use
    - blur_every: How often to blur the image as an implicit regularizer
    - max_jitter: How much to gjitter the image as an implicit regularizer
    - show_every: How often to show the intermediate result
    """
    l2_reg = kwargs.pop("l2_reg", 1e-3)
    learning_rate = kwargs.pop("learning_rate", 25)
    num_iterations = kwargs.pop("num_iterations", 5000)
    blur_every = kwargs.pop("blur_every", 10)
    max_jitter = kwargs.pop("max_jitter", 16)
    show_every = kwargs.pop("show_every", 1000)

    X = 255 * np.random.rand(224, 224, 3)
    X = preprocess_image(X)[None]

    loss = model.classifier[0, target_y] - 0.5 * l2_reg * (model.image * model.image)
    grad = tf.gradients(loss, model.image)[0]
    dx = learning_rate * grad / tf.norm(grad)

    for t in range(num_iterations):
        # Randomly jitter the image a bit; this gives slightly nicer results
        ox, oy = np.random.randint(-max_jitter, max_jitter + 1, 2)
        Xi = X.copy()
        X = np.roll(np.roll(X, ox, 1), oy, 2)

        gradient = sess.run(dx, {model.image: Xi})
        X += gradient

        # Undo the jitter
        X = np.roll(np.roll(X, -ox, 1), -oy, 2)

        # As a regularizer, clip and periodically blur
        X = np.clip(X, -SQUEEZENET_MEAN / SQUEEZENET_STD, (1.0 - SQUEEZENET_MEAN) / SQUEEZENET_STD)
        if t % blur_every == 0:
            X = blur_image(X, sigma=0.5)

        # Periodically show the image
        if t == 0 or (t + 1) % show_every == 0 or t == num_iterations - 1:
            plt.imshow(deprocess_image(X[0]))
            class_name = class_names[target_y]
            plt.title("%s\nIteration %d / %d" % (class_name, t + 1, num_iterations))
            plt.gcf().set_size_inches(4, 4)
            plt.axis("off")
            plt.show()

    return X


if __name__ == "__main__":
    tf.reset_default_graph()
    sess = get_session()

    SAVE_PATH = "cs231n/datasets/squeezenet.ckpt"
    model = SqueezeNet(save_path=SAVE_PATH, sess=sess)

    X_raw, y, class_names = load_imagenet_val(num=5)
    X = np.array([preprocess_image(img) for img in X_raw])

    show_fooling_image(idx=0, target_y=0)
