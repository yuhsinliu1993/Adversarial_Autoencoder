import os
import sys
import argparse

import numpy as np
import tensorflow as tf

from aae import AAE

from utils import to_categorical


def train(input_dim, z_dim, num_epochs, num_classes, batch_size, learning_rate, shuffle=False, data_dir=None):
    # Load data
    X_train = np.load(os.path.join(data_dir, 'data.npy'))
    y_train = np.load(os.path.join(data_dir, 'label.npy'))
    y_train = to_categorical(y_train, num_classes + 1)
    print('Number of training images: %d' % X_train.shape[0])

    with tf.Graph().as_default():
        sess = tf.Session()

        with sess.as_default():
            # Build model
            aae = AAE(input_dim, z_dim, num_classes, batch_size, learning_rate)
            aae.build(L_G_type=FLAGS.L_G_type)

            tf.logging.info('Create new session')
            sess.run(tf.global_variables_initializer())

            num_batches_per_epoch = X_train.shape[0] // batch_size

            for epoch in range(num_epochs):
                total_vae_loss, total_gen_loss, total_disc_loss = 0.0, 0.0, 0.0

                if shuffle:
                    shuffle_indices = np.random.permutation(np.arange(X_train.shape[0]))
                    X_shuffled = X_train[shuffle_indices]
                    y_shuffled = y_train[shuffle_indices]
                else:
                    X_shuffled = X_train
                    y_shuffled = y_train

                for i in range(num_batches_per_epoch):
                    start_index = i * batch_size
                    end_index = min((i + 1) * batch_size, X_train.shape[0])
                    X_batch = X_shuffled[start_index:end_index]
                    y_batch = y_shuffled[start_index:end_index]

                    vae_loss, gen_loss, disc_loss = train_step(X_batch, y_batch, sess, aae)

                    total_vae_loss += vae_loss
                    total_gen_loss += gen_loss
                    total_disc_loss += disc_loss

                print("Epoch %d ==> vae_loss: %.4f\tgen_loss: %.4f\tdisc_loss: %.4f" % (epoch, total_vae_loss / num_batches_per_epoch, total_gen_loss / num_batches_per_epoch, total_disc_loss / num_batches_per_epoch))


def train_step(X, y, sess, model):
    vae_loss, gen_loss, disc_loss = 0.0, 0.0, 0.0

    disc_loss += model.train_DISCRIMINATOR(X, y, sess)

    vae_loss += model.train_VAE(X, sess)

    model.train_GENERATOR(X, y, sess)
    model.train_GENERATOR(X, y, sess)
    model.train_GENERATOR(X, y, sess)
    model.train_GENERATOR(X, y, sess)
    model.train_GENERATOR(X, y, sess)
    gen_loss += model.train_GENERATOR(X, y, sess)

    return vae_loss, gen_loss, disc_loss


def run(_):
    if FLAGS.mode == 'train':
        input_dim = 28 * 28 * 1
        z_dim = 2
        train(input_dim, z_dim, FLAGS.num_epochs, FLAGS.num_classes, FLAGS.batch_size, FLAGS.learning_rate, FLAGS.shuffle, FLAGS.data_dir)
    else:
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=10,
        help='Specify number of epochs'
    )
    parser.add_argument(
        '--num_classes',
        type=int,
        default=10,
        help='Specify number of classes'
    )
    parser.add_argument(
        '--L_G_type',
        type=int,
        default=1,
        help='Specify the type of Generator Loss'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=128,
        help='Batch size. Must divide evenly into the dataset sizes.'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help='Specify learning rate'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data',
        help='Specify the directory of data',
    )
    parser.add_argument(
        '--mode',
        type=str,
        help='Specify mode: `train` or `eval`',
        required=True
    )
    parser.add_argument(
        '--shuffle',
        action='store_true',
        default=False,
        help='Whether shuffle the data or not',
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=run, argv=[sys.argv[0]] + unparsed)
