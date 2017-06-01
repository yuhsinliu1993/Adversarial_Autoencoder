import numpy as np
import tensorflow as tf

from models import Encoder, Decoder, Discriminator, Sampler

from utils import TINY


class AAE(object):
    """
    INPUTS:
        X: images
        y_labels: number of classes and +1 for unlabelled images
        z_input: True images draw from the prior z

        p(z): the prior of true images
        q(z): the prior of latent space
    """

    def __init__(self, input_dim, z_dim, num_classes, batch_size, learning_rate):
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.step = 0

        self.X = tf.placeholder(dtype=tf.float32, shape=(self.batch_size, self.input_dim), name='X_input')
        self.y_labels = tf.placeholder(tf.float32, [self.batch_size, self.num_classes + 1])
        self.z_input = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.z_dim])

        self.encoder = Encoder([self.input_dim, 1200, 600, 100], self.z_dim)
        self.decoder = Decoder([self.z_dim, 100, 600, 1200, self.input_dim])
        self.sampler = Sampler(self.num_classes)
        self.discriminator = Discriminator([self.z_dim + self.num_classes + 1, 50, 20, 10, 1])  # input dimension is the concatenation of `z_dim` and `sample`  and  `+1` for unlabelled images

    def build(self, L_G_type=1):

        self.q_z = self._build_latent_space()
        self._build_VAE_network()
        self._build_GAN_network(L_G_type)

        # Get training variables
        encoder_train_vars = self.encoder.get_variables()
        decoder_train_vars = self.decoder.get_variables()
        disc_train_vars = self.discriminator.get_variables()

        self.vae_train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.recon_loss, var_list=encoder_train_vars + decoder_train_vars)
        self.disc_train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.L_D, var_list=disc_train_vars)
        self.gen_train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.L_G, var_list=encoder_train_vars)

        # ???
        # self.mu, _ = self.encoder(self.X, is_training=False)
        # self.gen_imgs = self.decoder(self.z_input, is_training=False)

    def _build_VAE_network(self):
        # Decoding: reconstruct images by drawing samples from q(z)
        recon_imgs = self.decoder(self.q_z, is_training=True)

        # Reconstruction Loss between generating images and real images
        self.recon_loss = tf.reduce_mean(tf.reduce_sum(tf.square(recon_imgs - self.X), axis=1))

        self.recon_loss_summary = tf.summary.scalar('Reconstruct_Loss', self.recon_loss)

    def _build_GAN_network(self, L_G_type):
        """ Build GAN network
        sigmoid_cross_entropy_with_logits: x: logits, t: labels
            t * -log(sigmoid(x)) + (1 - t) * -log(1 - sigmoid(x))

        Loss of Generator:
            1. minimize  log(1 - sigmoid(D(z)))
            2. minimize  -log(sigmoid(D(z)))

        Loss of Discriminator:
            1. minimize  -log(sigmoid(D(z'))) - log(1 - sigmoid(D(z)))

        ps: z => fake imgs    z' => true imgs
        """
        with tf.variable_scope(tf.get_variable_scope()):
            fake_logits = self.discriminator(self.q_z, self.y_labels, is_training=True)
            tf.get_variable_scope().reuse_variables()  # reuse variables for "only" Discriminator
            true_logits = self.discriminator(self.z_input, self.y_labels, is_training=True)

            self.L_D = -tf.reduce_mean(tf.log(tf.sigmoid(true_logits) + TINY) + tf.log(1. - tf.sigmoid(fake_logits) + TINY))

            if L_G_type == 1:
                self.L_G = tf.reduce_mean(1. - tf.log(tf.sigmoid(fake_logits) + TINY))
            elif L_G_type == 2:
                self.L_G = -tf.reduce_mean(tf.log(tf.sigmoid(fake_logits) + TINY))

            self.L_D_summary = tf.summary.scalar('Discriminator_Loss', self.L_D)
            self.L_G_summary = tf.summary.scalar('Generator_Loss', self.L_G)

    def _build_latent_space(self):
        """
        Adopting re-parameterization trick for backpropagation

        Calculate the latent variable `z`  (use for generating the fake image by drawing samples from latent space z with the prior q(z) )
        """
        mu, log_sigma = self.encoder(self.X, is_training=True)
        noise = tf.random_normal([self.batch_size, self.z_dim])
        return noise * tf.exp(log_sigma) + mu

    def train_VAE(self, X, sess, writer=None):
        feed_dict = {
            self.X: X
        }

        _, recon_loss, summary = sess.run([self.vae_train_op, self.recon_loss, self.recon_loss_summary], feed_dict=feed_dict)

        if writer:
            writer.add_summary(summary, self.step)

        return recon_loss

    def train_GENERATOR(self, X, y, sess, writer=None):
        feed_dict = {
            self.X: X,
            self.y_labels: y
        }

        _, gen_loss, summary = sess.run([self.gen_train_op, self.L_G, self.L_G_summary], feed_dict=feed_dict)

        if writer:
            writer.add_summary(summary, self.step)

        return gen_loss

    def train_DISCRIMINATOR(self, X, y, sess, writer=None):
        z = self.sampler(np.argmax(y, axis=1))

        feed_dict = {
            self.X: X,
            self.y_labels: y,
            self.z_input: z
        }

        _, disc_loss, summary = sess.run([self.disc_train_op, self.L_D, self.L_D_summary], feed_dict=feed_dict)

        if writer:
            writer.add_summary(summary, self.step)

        return disc_loss
