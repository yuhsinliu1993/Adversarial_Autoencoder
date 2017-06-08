import tensorflow as tf

import nn


class Encoder(object):

    def __init__(self, z_dim):
        self.z_dim = z_dim

    def __call__(self, inputs, is_training):

        with tf.variable_scope('encoder'):
            enc_l1 = nn.linear(inputs, (784, 1000), 'enc_l1')
            enc_b1 = nn.batch_normalization(enc_l1, 1000, 'enc_b1', is_training)
            enc_r1 = nn.leaky_relu(enc_b1)

            enc_l2 = nn.linear(enc_r1, (1000, 1000), 'enc_l2')
            enc_b2 = nn.batch_normalization(enc_l2, 1000, 'enc_b2', is_training)
            enc_r2 = nn.leaky_relu(enc_b2)

            enc_l3 = nn.linear(enc_r2, (1000, self.z_dim), 'enc_l3')
            encoded = nn.batch_normalization(enc_l3, self.z_dim, 'enc_b3', is_training)

            return encoded

    def get_variables(self):
        _vars = tf.trainable_variables()
        ret = []

        for var in _vars:
            if 'encoder' in var.name:
                ret.append(var)

        return ret


class Decoder(object):

    def __init__(self, z_dim):
        # should be [z_dim, ..., 784]
        self.z_dim = z_dim

    def __call__(self, inputs, is_training):

        with tf.variable_scope('decoder'):
            dec_l1 = nn.linear(inputs, (self.z_dim, 1000), 'dec_l1')
            dec_b1 = nn.batch_normalization(dec_l1, 1000, 'dec_b1', is_training)
            dec_r1 = nn.leaky_relu(dec_b1)

            dec_l2 = nn.linear(dec_r1, (1000, 1000), 'dec_l2')
            dec_b2 = nn.batch_normalization(dec_l2, 1000, 'dec_b2', is_training)
            dec_r2 = nn.leaky_relu(dec_b2)

            dec_l3 = nn.linear(dec_r2, (1000, 784), 'dec_l3')

            return tf.sigmoid(dec_l3)

    def get_variables(self):
        _vars = tf.trainable_variables()
        ret = []

        for var in _vars:
            if 'decoder' in var.name:
                ret.append(var)

        return ret


class Discriminator(object):

    def __init__(self, z_dim, num_classes):
        self.z_dim = z_dim
        self.num_classes = num_classes

    def __call__(self, inputs, y, is_training):
        """
        INPUTS:
            z: latent space
            y: one-hot vectors with shpae [num_class + 1]  (+1: for unlabelled data)
        RETURN:
            `logits` (i.e. no activation function like sigmoid, softmax, ...)
        """

        h = tf.concat([inputs, y], axis=1)   # inputs's shape: (batch_size, z_dim + num_classes + 1)

        with tf.variable_scope('discriminator'):
            h = nn.linear(h, (self.z_dim + self.num_classes + 1, 500), 'disc_l1')
            h = nn.batch_normalization(h, 500, 'disc_b1')
            h = nn.leaky_relu(h)

            h = nn.linear(h, (500, 500), 'disc_l2')
            h = nn.batch_normalization(h, 500, 'disc_b2')
            h = nn.leaky_relu(h)

            logits = nn.linear(h, (500, 1), 'disc_l3')

            return logits

    def get_variables(self):
        _vars = tf.trainable_variables()
        ret = []

        for var in _vars:
            if 'discriminator' in var.name:
                ret.append(var)

        return ret
