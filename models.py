import math
import tensorflow as tf
import numpy as np

from utils import get_dim


class Encoder(object):

    def __init__(self, layer_list, z_dim):
        self.layer_list = layer_list  # should be [784, ..., z_dim]
        self.z_dim = z_dim

    def __call__(self, X, is_training):
        """
        INPUT:
            X: shape = (None, hieght x width x depth)
        Return:

        """
        hidden = X

        with tf.variable_scope('encoder'):

            for i, (in_dim, out_dim) in enumerate(zip(self.layer_list, self.layer_list[1:])):
                shape = (in_dim, out_dim)
                hidden = _linear(hidden, shape, i)
                hidden = leaky_relu(batch_normalization(hidden, i, is_training))

            """
            Instead of generating a vector of real values, encoder will generate
            a vector of means and a vector of standard deviations.
            """
            h_out_dim = get_dim(hidden)
            mu = _linear(hidden, (h_out_dim, self.z_dim), 'mu')  # mean
            log_sigma = _linear(hidden, (h_out_dim, self.z_dim), 'log_sigma')  # log std

        return mu, log_sigma

    def get_variables(self):
        _vars = tf.trainable_variables()
        ret = []

        for var in _vars:
            if 'encoder' in var.name:
                ret.append(var)

        return ret


class Decoder(object):

    def __init__(self, layer_list):
        # should be [z_dim, ..., 784]
        self.layer_list = layer_list

    def __call__(self, z, is_training):

        hidden = z

        with tf.variable_scope('decoder'):
            for i, (in_dim, out_dim) in enumerate(zip(self.layer_list, self.layer_list[1:])):
                shape = (in_dim, out_dim)
                output = _linear(hidden, shape, i)
                hidden = leaky_relu(batch_normalization(output, i, is_training))

        return output

    def get_variables(self):
        _vars = tf.trainable_variables()
        ret = []

        for var in _vars:
            if 'decoder' in var.name:
                ret.append(var)

        return ret


class Discriminator(object):

    def __init__(self, layer_list):
        self.layer_list = layer_list  # should be [z_dim + classes + 1, ..., 1]

    def __call__(self, z, y, is_training):
        """
        INPUTS:
            z: latent space
            y: one-hot vectors with shpae [num_class + 1]  (+1: for unlabelled data)
        RETURN:
            `logits` (i.e. no activation function like sigmoid, softmax, ...)
        """

        hidden = tf.concat([z, y], axis=1)

        with tf.variable_scope('discriminator'):
            for i, (in_dim, out_dim) in enumerate(zip(self.layer_list, self.layer_list[1:])):
                shape = (in_dim, out_dim)
                output = _linear(hidden, shape, i)
                hidden = leaky_relu(batch_normalization(output, i, is_training))

        return output

    def get_variables(self):
        _vars = tf.trainable_variables()
        ret = []

        for var in _vars:
            if 'discriminator' in var.name:
                ret.append(var)

        return ret


class Sampler(object):
    """
    Draw "true" samples z' from the predifined prior p(z)
    """

    def __init__(self, class_num):
        # 0 -- class_num - 1: calssification index
        # class_num: for unlabeled index
        self.class_num = class_num

        self.x_variance = 0.5
        self.y_variance = 0.05
        self.radial = 2.0

    def __call__(self, class_indexes):
        ret = []
        for class_index in class_indexes:
            x = np.random.normal(0.0, self.x_variance) + self.radial
            y = np.random.normal(0.0, self.y_variance)
            rad = self._get_radian(class_index)
            x, y = self._rotate(x, y, rad)
            ret.append([x, y])

        return np.asarray(ret)

    def _get_radian(self, class_index):
        return 2 * np.pi * float(class_index) / float(self.class_num)

    def _rotate(self, x, y, radian):
        mod_x = x * math.cos(radian) - y * math.sin(radian)
        mod_y = x * math.sin(radian) + y * math.cos(radian)
        return mod_x, mod_y


def _linear(x, shape, _id):
    if len(shape) != 2 or not isinstance(shape, (tuple, list)):
        raise ValueError("`shape` should be a list of (input_dim, output_dim)")

    weights = tf.get_variable(name='weights_{}'.format(_id),
                              shape=shape,
                              dtype=tf.float32,
                              initializer=tf.random_normal_initializer(stddev=1.0 / np.sqrt(float(shape[0]))),
                              trainable=True)

    biases = tf.get_variable(name='biases_{}'.format(_id),
                             shape=shape[1],
                             dtype=tf.float32,
                             initializer=tf.constant_initializer(0.0),
                             trainable=True)

    return tf.matmul(x, weights) + biases


def leaky_relu(x, leak=0.2):
    return tf.maximum(x, leak * x)


def batch_normalization(x, _ids, is_training=True):
    decay_rate = 0.99

    shape = x.get_shape().as_list()
    dim = shape[-1]

    if len(shape) == 2:
        mean, var = tf.nn.moments(x, [0], name='moments_bn_{}'.format(_ids))
    elif len(shape) == 4:
        mean, var = tf.nn.moments(x, [0, 1, 2], name='moments_bn_{}'.format(_ids))

    avg_mean = tf.get_variable(name='avg_mean_bn_{}'.format(_ids),
                               shape=(1, dim),
                               dtype=tf.float32,
                               initializer=tf.constant_initializer(0.0),
                               trainable=False)

    avg_var = tf.get_variable(name='avg_var_bn_{}'.format(_ids),
                              shape=(1, dim),
                              dtype=tf.float32,
                              initializer=tf.constant_initializer(1.0),
                              trainable=False)

    beta = tf.get_variable(name='beta_bn_{}'.format(_ids),
                           shape=(1, dim),
                           dtype=tf.float32,
                           initializer=tf.constant_initializer(0.0),
                           trainable=True)

    gamma = tf.get_variable(name='gamma_bn_{}'.format(_ids),
                            shape=(1, dim),
                            dtype=tf.float32,
                            initializer=tf.constant_initializer(1.0),
                            trainable=True)

    if is_training:
        avg_mean_assign_op = tf.assign(avg_mean, decay_rate * avg_mean + (1 - decay_rate) * mean)
        avg_var_assign_op = tf.assign(avg_var, decay_rate * avg_var + (1 - decay_rate) * var)

        with tf.control_dependencies([avg_mean_assign_op, avg_var_assign_op]):
            ret = gamma * (x - mean) / tf.sqrt(1e-6 + var) + beta
    else:
        ret = gamma * (x - avg_mean) / tf.sqrt(1e-6 + avg_var) + beta

    return ret
