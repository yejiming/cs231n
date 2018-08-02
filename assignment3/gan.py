import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

plt.rcParams["figure.figsize"] = (10.0, 8.0) # set default size of plots
plt.rcParams["image.interpolation"] = "nearest"
plt.rcParams["image.cmap"] = "gray"


def show_images(images):
    images = np.reshape(images, [images.shape[0], -1])  # images reshape to (batch_size, D)
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))
    sqrtimg = int(np.ceil(np.sqrt(images.shape[1])))

    _ = plt.figure(figsize=(sqrtn, sqrtn))
    gs = gridspec.GridSpec(sqrtn, sqrtn)
    gs.update(wspace=0.05, hspace=0.05)

    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis("off")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect("equal")
        plt.imshow(img.reshape([sqrtimg,sqrtimg]))


def preprocess_img(x):
    return 2 * x - 1.0


def deprocess_img(x):
    return (x + 1.0) / 2.0


def rel_error(x,y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


def count_params():
    """Count the number of parameters in the current TensorFlow graph """
    param_count = np.sum([np.prod(x.get_shape().as_list()) for x in tf.global_variables()])
    return param_count


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    return session


def leaky_relu(x, alpha=0.01):
    """Compute the leaky ReLU activation function.

    Inputs:
    - x: TensorFlow Tensor with arbitrary shape
    - alpha: leak parameter for leaky ReLU

    Returns:
    TensorFlow Tensor with the same shape as x
    """
    return tf.nn.leaky_relu(tf.cast(x, tf.float32), alpha)


def test_leaky_relu(x, y_true):
    tf.reset_default_graph()
    with get_session() as sess:
        y_tf = leaky_relu(tf.constant(x))
        y = sess.run(y_tf)
        print("Maximum error: %g"%rel_error(y_true, y))


def sample_noise(batch_size, dim):
    """Generate random uniform noise from -1 to 1.

    Inputs:
    - batch_size: integer giving the batch size of noise to generate
    - dim: integer giving the dimension of the the noise to generate

    Returns:
    TensorFlow Tensor containing uniform noise in [-1, 1] with shape [batch_size, dim]
    """
    return tf.random_uniform([batch_size, dim], -1, 1)


def test_sample_noise():
    batch_size = 3
    dim = 4
    tf.reset_default_graph()
    with get_session() as sess:
        z = sample_noise(batch_size, dim)
        # Check z has the correct shape
        assert z.get_shape().as_list() == [batch_size, dim]
        # Make sure z is a Tensor and not a numpy array
        assert isinstance(z, tf.Tensor)
        # Check that we get different noise for different evaluations
        z1 = sess.run(z)
        z2 = sess.run(z)
        assert not np.array_equal(z1, z2)
        # Check that we get the correct range
        assert np.all(z1 >= -1.0) and np.all(z1 <= 1.0)
        print("All tests passed!")


def discriminator(x):
    """Compute discriminator score for a batch of input images.

    Inputs:
    - x: TensorFlow Tensor of flattened input images, shape [batch_size, 784]

    Returns:
    TensorFlow Tensor with shape [batch_size, 1], containing the score 
    for an image being real for each input image.
    """
    with tf.variable_scope("discriminator"):
        x = tf.reshape(x, [-1, 28, 28, 1])

        conv_1 = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=(2, 2))
        relu_1 = leaky_relu(conv_1, alpha=0.01)

        conv_2 = tf.layers.conv2d(inputs=relu_1, filters=128, kernel_size=4, strides=(2, 2))
        relu_2 = leaky_relu(conv_2, alpha=0.01)

        bn_1 = tf.layers.batch_normalization(inputs=relu_2)
        flat_1 = tf.layers.flatten(inputs=bn_1)

        dense_1 = tf.layers.dense(inputs=flat_1, units=1024, use_bias=True)
        relu_3 = leaky_relu(dense_1, alpha=0.01)

        logits = tf.layers.dense(inputs=relu_3, units=1, use_bias=True)
        return logits


def test_discriminator(true_count=1102721):
    tf.reset_default_graph()
    with get_session() as sess:
        y = discriminator(tf.ones((2, 784)))
        cur_count = count_params()
        if cur_count != true_count:
            print("Incorrect number of parameters in discriminator. {0} instead of {1}. Check your achitecture."
                  .format(cur_count,true_count))
        else:
            print("Correct number of parameters in discriminator.")


def generator(z):
    """Generate images from a random noise vector.

    Inputs:
    - z: TensorFlow Tensor of random noise with shape [batch_size, noise_dim]

    Returns:
    TensorFlow Tensor of generated images, with shape [batch_size, 784].
    """
    with tf.variable_scope("generator"):
        dense_1 = tf.layers.dense(inputs=z, units=1024, activation=tf.nn.relu)
        bn_1 = tf.layers.batch_normalization(inputs=dense_1)

        dense_2 = tf.layers.dense(inputs=bn_1, units=7*7*128, activation=tf.nn.relu)
        bn_2 = tf.layers.batch_normalization(inputs=dense_2)

        z_img = tf.reshape(bn_2, [-1, 7, 7, 128])

        trans_1 = tf.layers.conv2d_transpose(inputs=z_img, filters=64, strides=2, kernel_size=4,
                                             padding="same", activation=tf.nn.relu)
        bn_3 = tf.layers.batch_normalization(inputs=trans_1)
        img = tf.layers.conv2d_transpose(inputs=bn_3, filters=1, strides=2, kernel_size=4, activation=tf.nn.tanh,
                                         padding="same")

        img = tf.reshape(img, shape=[-1, 784])
        return img


def test_generator(true_count=6595521):
    tf.reset_default_graph()
    with get_session() as sess:
        y = generator(tf.ones((1, 4)))
        cur_count = count_params()
        if cur_count != true_count:
            print("Incorrect number of parameters in generator. {0} instead of {1}. Check your achitecture."
                  .format(cur_count,true_count))
        else:
            print("Correct number of parameters in generator.")


def gan_loss(logits_real, logits_fake):
    """Compute the GAN loss.

    Inputs:
    - logits_real: Tensor, shape [batch_size, 1], output of discriminator
        Log probability that the image is real for each real image
    - logits_fake: Tensor, shape[batch_size, 1], output of discriminator
        Log probability that the image is real for each fake image

    Returns:
    - D_loss: discriminator loss scalar
    - G_loss: generator loss scalar
    """
    labels_real = tf.ones_like(logits_real)
    labels_fake = tf.zeros_like(logits_fake)

    G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_real, logits=logits_fake))
    D_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_real, logits=logits_real)) + \
             tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_fake, logits=logits_fake))

    return D_loss, G_loss


def test_gan_loss(logits_real, logits_fake, d_loss_true, g_loss_true):
    tf.reset_default_graph()
    with get_session() as sess:
        d_loss, g_loss = sess.run(gan_loss(tf.constant(logits_real), tf.constant(logits_fake)))
    print("Maximum error in d_loss: %g"%rel_error(d_loss_true, d_loss))
    print("Maximum error in g_loss: %g"%rel_error(g_loss_true, g_loss))


def lsgan_loss(score_real, score_fake):
    """Compute the Least Squares GAN loss.

    Inputs:
    - score_real: Tensor, shape [batch_size, 1], output of discriminator
        score for each real image
    - score_fake: Tensor, shape[batch_size, 1], output of discriminator
        score for each fake image    

    Returns:
    - D_loss: discriminator loss scalar
    - G_loss: generator loss scalar
    """
    G_loss = 0.5 * tf.reduce_mean((score_fake - 1) ** 2)
    D_loss = 0.5 * tf.reduce_mean((score_real - 1) ** 2) + 0.5 * tf.reduce_mean(score_fake ** 2)
    return D_loss, G_loss


def test_lsgan_loss(score_real, score_fake, d_loss_true, g_loss_true):
    with get_session() as sess:
        d_loss, g_loss = sess.run(
            lsgan_loss(tf.constant(score_real), tf.constant(score_fake)))
    print("Maximum error in d_loss: %g"%rel_error(d_loss_true, d_loss))
    print("Maximum error in g_loss: %g"%rel_error(g_loss_true, g_loss))


def wgangp_loss(logits_real, logits_fake, batch_size, x, G_sample):
    """Compute the WGAN-GP loss.

    Inputs:
    - logits_real: Tensor, shape [batch_size, 1], output of discriminator
        Log probability that the image is real for each real image
    - logits_fake: Tensor, shape[batch_size, 1], output of discriminator
        Log probability that the image is real for each fake image
    - batch_size: The number of examples in this batch
    - x: the input (real) images for this batch
    - G_sample: the generated (fake) images for this batch

    Returns:
    - D_loss: discriminator loss scalar
    - G_loss: generator loss scalar
    """
    D_loss = - tf.reduce_mean(logits_real) + tf.reduce_mean(logits_fake)
    G_loss = - tf.reduce_mean(logits_fake)

    # lambda from the paper
    lamda = 10

    # random sample of batch_size (tf.random_uniform)
    eps = tf.random_uniform(shape=[batch_size, 1], minval=0, maxval=1)
    x_hat = eps * x + (1 - eps) * G_sample

    # Gradients of Gradients is kind of tricky!
    with tf.variable_scope("", reuse=True) as scope:
        grad_D_x_hat = tf.gradients(discriminator(x_hat), x_hat)

    grad_norm = tf.norm(grad_D_x_hat[0], axis=1, ord="euclidean")
    grad_pen = tf.reduce_mean(tf.square(grad_norm - 1.))

    D_loss += lamda * grad_pen

    return D_loss, G_loss


def get_solvers(learning_rate=1e-3, beta1=0.5, beta2=0.999):
    """Create solvers for GAN training.

    Inputs:
    - learning_rate: learning rate to use for both solvers
    - beta1: beta1 parameter for both solvers (first moment decay)

    Returns:
    - D_solver: instance of tf.train.AdamOptimizer with correct learning_rate and beta1
    - G_solver: instance of tf.train.AdamOptimizer with correct learning_rate and beta1
    """
    D_solver = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1, beta2=beta2)
    G_solver = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1, beta2=beta2)
    return D_solver, G_solver


def build_gan():
    tf.reset_default_graph()

    # number of images for each batch
    batch_size = 128
    # our noise dimension
    noise_dim = 96

    # placeholder for images from the training dataset
    x = tf.placeholder(tf.float32, [None, 784])
    # random noise fed into our generator
    z = sample_noise(batch_size, noise_dim)
    # generated images
    G_sample = generator(z)

    with tf.variable_scope("") as scope:
        # scale images to be -1 to 1
        logits_real = discriminator(preprocess_img(x))
        # Re-use discriminator weights on new inputs
        scope.reuse_variables()
        logits_fake = discriminator(G_sample)

    # Get the list of variables for the discriminator and generator
    D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "discriminator")
    G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "generator")

    D_solver, G_solver = get_solvers(learning_rate=1e-3, beta1=0., beta2=0.9)
    D_loss, G_loss = wgangp_loss(logits_real, logits_fake, batch_size, x, G_sample)
    D_train_step = D_solver.minimize(D_loss, var_list=D_vars)
    G_train_step = G_solver.minimize(G_loss, var_list=G_vars)
    D_extra_step = tf.get_collection(tf.GraphKeys.UPDATE_OPS, "discriminator")
    G_extra_step = tf.get_collection(tf.GraphKeys.UPDATE_OPS, "generator")

    return x, G_sample, G_train_step, G_loss, D_train_step, D_loss, G_extra_step, D_extra_step


def run_a_gan(sess, x, G_sample, G_train_step, G_loss, D_train_step, D_loss, G_extra_step, D_extra_step,
              show_every=500, print_every=50, batch_size=128, num_epoch=5):
    """Train a GAN for a certain number of epochs.

    Inputs:
    - sess: A tf.Session that we want to use to run our data
    - G_train_step: A training step for the Generator
    - G_loss: Generator loss
    - D_train_step: A training step for the Generator
    - D_loss: Discriminator loss
    - G_extra_step: A collection of tf.GraphKeys.UPDATE_OPS for generator
    - D_extra_step: A collection of tf.GraphKeys.UPDATE_OPS for discriminator
    Returns:
        Nothing
    """
    mnist = input_data.read_data_sets("MNIST_data", one_hot=False)
    show_images(mnist.train.next_batch(16)[0])

    # compute the number of iterations we need
    max_iter = int(mnist.train.num_examples * num_epoch / batch_size)
    for it in range(max_iter):
        # every show often, show a sample result
        if it % show_every == 0:
            samples = sess.run(G_sample)
            show_images(samples[:16])
            plt.show()
            print()
        # run a batch of data through the network
        minibatch, minbatch_y = mnist.train.next_batch(batch_size)
        _, D_loss_curr = sess.run([D_train_step, D_loss], feed_dict={x: minibatch})
        _, G_loss_curr = sess.run([G_train_step, G_loss])

        # print loss every so often.
        # We want to make sure D_loss doesn"t go to 0
        if it % print_every == 0:
            print("Iter: {}, D: {:.4}, G:{:.4}".format(it, D_loss_curr, G_loss_curr))

    print("Final images")
    samples = sess.run(G_sample)

    show_images(samples[:16])
    plt.show()


if __name__ == "__main__":
    x, G_sample, G_train_step, G_loss, D_train_step, D_loss, G_extra_step, D_extra_step = build_gan()
    with get_session() as sess:
        sess.run(tf.global_variables_initializer())
        run_a_gan(sess, x, G_sample, G_train_step, G_loss, D_train_step, D_loss, G_extra_step, D_extra_step)
