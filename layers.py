import numpy as np
import tensorflow as tf


def linear(x, shape, name):
    if len(shape) != 2 or not isinstance(shape, (tuple, list)):
        raise ValueError("`shape` should be a list of (input_dim, output_dim)")

    weights = tf.get_variable(name='weights_{}'.format(name),
                              shape=shape,
                              dtype=tf.float32,
                              initializer=tf.random_normal_initializer(stddev=1.0 / np.sqrt(float(shape[0]))),
                              trainable=True)

    biases = tf.get_variable(name='biases_{}'.format(name),
                             shape=shape[1],
                             dtype=tf.float32,
                             initializer=tf.constant_initializer(0.0),
                             trainable=True)

    return tf.matmul(x, weights) + biases


def leaky_relu(x, leak=0.2):
    return tf.maximum(x, leak * x)


def batch_normalization(x, dim, name, is_training=True):
    decay_rate = 0.99

    mean, var = tf.nn.moments(x, [0], name='moments_bn_{}'.format(name))

    avg_mean = tf.get_variable(name='avg_mean_bn_{}'.format(name),
                               shape=(1, dim),
                               dtype=tf.float32,
                               initializer=tf.constant_initializer(0.0),
                               trainable=False)

    avg_var = tf.get_variable(name='avg_var_bn_{}'.format(name),
                              shape=(1, dim),
                              dtype=tf.float32,
                              initializer=tf.constant_initializer(1.0),
                              trainable=False)

    beta = tf.get_variable(name='beta_bn_{}'.format(name),
                           shape=(1, dim),
                           dtype=tf.float32,
                           initializer=tf.constant_initializer(0.0),
                           trainable=True)

    gamma = tf.get_variable(name='gamma_bn_{}'.format(name),
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
