import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from tf1_estimator import AbstractEstimator


class MnistEstimator(AbstractEstimator):
    def __init__(self):
        super().__init__()

    def get_network(self, input):
        dp_keep_prob = 0.8

        h = tf.layers.conv2d(
            input,
            filters=4,
            kernel_size=4,
            strides=2,
            padding='same',
            activation=tf.tanh
        )
        h = tf.nn.dropout(h, dp_keep_prob)

        h = tf.layers.conv2d(
            h,
            filters=4,
            kernel_size=4,
            strides=2,
            padding='same',
            activation=tf.tanh
        )

        h = tf.nn.dropout(h, dp_keep_prob)

        h = tf.layers.flatten(h)
        y = tf.layers.dense(h, 10, activation=tf.tanh)
        y = tf.nn.softmax(y)

        return y

    def get_loss_and_optimizer(self, y, labels, learning_rate=0.001):
        # get loss
        loss = tf.losses.mean_squared_error(y, labels)

        optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
        train_op = optimizer.minimize(
            loss, global_step=tf.train.get_global_step())

        return loss, train_op


# --------------------------------------------------------------------------------
def make_empty_dir(dir):
    """
    空のディレクトリを生成。
    すでに存在しているディレクトリの場合、中身を削除

    Parameters
    ----------
    dir : str
        ディレクトリのパス
    """
    os.makedirs(dir, exist_ok=True)
    shutil.rmtree(dir)
    os.makedirs(dir, exist_ok=False)


def main(argv):
    model_dir = './model'
    batch_size = 1000
    epochs = 3

    make_empty_dir(model_dir)

    mnist = input_data.read_data_sets("./data/mnist", one_hot=True)
    train_x = mnist.train.images.reshape(-1, 28, 28, 1)
    train_y = mnist.train.labels
    test_x = mnist.test.images.reshape(-1, 28, 28, 1)
    test_y = mnist.test.labels

    estimator = MnistEstimator()
    estimator.train(train_x, train_y, test_x, test_y,
                    epochs=epochs, batch_size=batch_size, model_dir=model_dir)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
