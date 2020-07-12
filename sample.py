import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import argparse
import os
import shutil


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


def get_data_as_np(mnist):
    train_x = mnist.train.images
    train_y = mnist.train.labels
    test_x = mnist.test.images
    test_y = mnist.test.labels

    # reshape inputs to 4D array
    train_x = train_x.reshape(-1, 28, 28, 1)
    test_x = test_x.reshape(-1, 28, 28, 1)

    return train_x, train_y, test_x, test_y


def get_network(input):
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

    return y


def get_loss_and_optimizer(y, labels):
    # get loss
    loss = tf.losses.mean_squared_error(y, labels)

    optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    return loss, train_op


def model_fn(features, labels, mode, params):

    # get tensor input
    input = features['x']
    y = get_network(input)

    # returns pred
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'y': y
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # calc loss
    loss, optimizer = get_loss_and_optimizer(y, labels)

    # mode for evaluation
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss)

    # create train_op
    assert mode == tf.estimator.ModeKeys.TRAIN

    # summaries to be shown in tensorboard
    tf.summary.scalar('train_loss', loss)

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=optimizer)


def get_input(x, y, epochs, batch_size, shuffle=True):
    return tf.estimator.inputs.numpy_input_fn(
        x={'x': x},
        y=y,
        num_epochs=epochs,
        batch_size=batch_size,
        shuffle=shuffle,
    )


def main(argv):
    model_dir = './model'
    batch_size = 1000
    epochs = 3

    make_empty_dir(model_dir)

    mnist = input_data.read_data_sets("./data/mnist", one_hot=True)
    train_x, train_y, test_x, test_y = get_data_as_np(mnist)

    # configuration of the model
    my_config = tf.estimator.RunConfig(
        save_checkpoints_steps=1000,
        keep_checkpoint_max=5,
        log_step_count_steps=1000,
        save_summary_steps=1000,
    )

    # create model
    model = tf.estimator.Estimator(
        model_fn=model_fn,
        params={

        },
        model_dir=model_dir,
        config=my_config)

    # create input pipeline for training.
    train_input_fn = get_input(train_x, train_y, epochs, batch_size)
    eval_input_fn = get_input(
        test_x[:batch_size], test_y[:batch_size], 1, batch_size)

    # start the training.
    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=10)
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn)
    tf.estimator.train_and_evaluate(model, train_spec, eval_spec)

    # get prediction after training.
    # predict() returns generator
    test_input_fn = get_input(test_x, test_y, 1, batch_size, shuffle=False)

    preds = model.predict(
        input_fn=test_input_fn,
    )

    # this is numpy array
    for pred in preds:
        print(pred['y'])


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
