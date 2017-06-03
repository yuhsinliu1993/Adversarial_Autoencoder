import os
import sys
import argparse
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf

from aae import AAE

from utils import to_categorical

np.random.seed(123)


def train(input_dim, z_dim, num_epochs, num_classes, batch_size, learning_rate, shuffle=False, data_dir=None, summary_dir=None):

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
            aae.build(G_type=FLAGS.G_type)

            loss_summary_writer = tf.summary.FileWriter(summary_dir, sess.graph)

            num_batches_per_epoch = X_train.shape[0] // batch_size

            tf.logging.info('Create new session')
            sess.run(tf.global_variables_initializer())

            for epoch in range(num_epochs):
                total_vae_loss, total_gen_loss, total_disc_loss = 0.0, 0.0, 0.0

                for i in range(num_batches_per_epoch):
                    start_index = i * batch_size
                    end_index = min((i + 1) * batch_size, X_train.shape[0])

                    X_batch = X_train[start_index:end_index]
                    y_batch = y_train[start_index:end_index]

                    vae_loss, gen_loss, disc_loss = train_step(X_batch, y_batch, sess, aae, loss_summary_writer)

                    total_vae_loss += vae_loss
                    total_gen_loss += gen_loss
                    total_disc_loss += disc_loss

                print("Epoch %3d ==> vae_loss: %.4f\tgen_loss: %.4f\tdisc_loss: %.4f" % (epoch, total_vae_loss / num_batches_per_epoch, total_gen_loss / num_batches_per_epoch, total_disc_loss / num_batches_per_epoch))

            if FLAGS.plot:
                indices = np.random.choice(X_train.shape[0], size=batch_size)
                X_sample = X_train[indices]
                y_sample = y_train[indices]
                plot_reconstructed_images(X_sample, y_sample, aae, sess)

                plot_generated_images(aae, sess)

                indices = np.random.choice(X_train.shape[0], size=5000)
                X_sample = X_train[indices]
                y_sample = y_train[indices]

                X_latent_space = aae.get_latent_space(sess, X_sample)
                X_latent_space = X_latent_space.astype('float64')
                plot_tsne(X_latent_space, y_sample)


def train_step(X, y, sess, model, writer):

    vae_loss = model.train_VAE(X, sess, writer)

    disc_loss = model.train_DISCRIMINATOR(X, y, sess, writer)

    model.train_GENERATOR(X, y, sess)
    model.train_GENERATOR(X, y, sess)
    model.train_GENERATOR(X, y, sess)
    gen_loss = model.train_GENERATOR(X, y, sess, writer)

    model.step += 1

    return vae_loss, gen_loss, disc_loss


def plot_tsne(X_sample, y_sample):
    from tsne import bh_sne

    vis_data = bh_sne(X_sample)

    vis_x = vis_data[:, 0]
    vis_y = vis_data[:, 1]

    plt.scatter(vis_x, vis_y, c=np.argmax(y_sample, 1), cmap=plt.cm.get_cmap("jet", 10))
    plt.colorbar(ticks=range(10))
    plt.clim(-0.5, 9.5)
    plt.show()


def plot_reconstructed_images(X_sample, y_sample, model, sess):
    X_reconstruct = model.get_reconstructed_images(sess, X_sample)

    plt.figure(figsize=(8, 12))
    for i in range(3):
        plt.subplot(5, 2, 2 * i + 1)
        plt.imshow(X_sample[i].reshape(28, 28), vmin=0, vmax=1, cmap="gray")
        plt.title("Test input")
        plt.colorbar()
        plt.subplot(5, 2, 2 * i + 2)
        plt.imshow(X_reconstruct[i].reshape(28, 28), vmin=0, vmax=1, cmap="gray")
        plt.title("Reconstruction")
        plt.colorbar()
    plt.tight_layout()
    plt.show()


def plot_generated_images(model, sess):
    X_reconstruct = model.get_generated_images(sess)

    plt.figure(figsize=(8, 12))
    plt.subplot(5, 2, 2)
    plt.imshow(X_reconstruct[0].reshape(28, 28), vmin=0, vmax=1, cmap="gray")
    plt.title("Reconstruction")
    plt.colorbar()
    plt.tight_layout()
    plt.show()


def run(_):
    if FLAGS.mode == 'train':
        input_dim = 28 * 28 * 1
        train(input_dim, FLAGS.z_dim, FLAGS.num_epochs, FLAGS.num_classes, FLAGS.batch_size, FLAGS.learning_rate, FLAGS.shuffle, FLAGS.data_dir, FLAGS.summary_dir)
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
        '--G_type',
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
        '--z_dim',
        type=int,
        default=100,
        help='Specify the dimension of the latent space'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=5e-4,
        help='Specify learning rate'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data',
        help='Specify the directory of data',
    )
    parser.add_argument(
        '--summary_dir',
        type=str,
        default='summary',
        help='Specify the directory of summaries',
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
    parser.add_argument(
        '--plot',
        action='store_true',
        default=True,
        help='Plot the t-sne, reconstructed images and generated images',
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=run, argv=[sys.argv[0]] + unparsed)
