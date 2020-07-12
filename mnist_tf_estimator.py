import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
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


def get_network2(x):
    h = tf.layers.conv2d(
        x,
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
    return x


def get_network(x):
    # weight initialization
    def weight_variable(name, shape):
        # http://www.366service.com/jp/qa/b6eaa42e0d670d6dc0ea792e09ddc04a
        return tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    # convolution
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    # pooling
    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Create the model
    #W = tf.Variable(tf.zeros([784, 10]))
    #b = tf.Variable(tf.zeros([10]))
    #y = tf.nn.softmax(tf.matmul(x, W) + b)

    # first convolutinal layer
    w_conv1 = weight_variable('w_conv1', [5, 5, 1, 32])
    b_conv1 = bias_variable([32])

    x_image = tf.reshape(x, [-1, 28, 28, 1])

    h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # second convolutional layer
    w_conv2 = weight_variable('w_conv2', [5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # densely connected layer
    w_fc1 = weight_variable('w_fc1', [7*7*64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

    # dropout
    h_fc1_drop = tf.nn.dropout(h_fc1, 0.8)

    # readout layer
    w_fc2 = weight_variable('w_fc2', [1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)

    return y_conv


def get_loss_and_optimizer(y, labels, learning_rate):
    loss = tf.losses.mean_squared_error(y, labels)

    optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return loss, train_op


def model_fn(features, labels, mode, params):
    x = features['x']
    y = get_network(x)

    # returns pred
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'y': y
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # get loss and optimizer
    learning_rate = params['learning_rate']
    loss, train_op = get_loss_and_optimizer(y, labels, learning_rate)

    # mode for evaluation
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss)

    # create train_op
    assert mode == tf.estimator.ModeKeys.TRAIN

    # summaries to be shown in tensorboard
    tf.summary.scalar('train_loss', loss)

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


######################################################################################
if __name__ == '__main__':
    mnist = input_data.read_data_sets("./data/mnist", one_hot=True)

    model_dir = './model'
    batch_size = 1000
    epochs = 3

    make_empty_dir(model_dir)

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
            'learning_rate': 0.0001,
        },
        model_dir=model_dir,
        config=my_config)

    # create input pipeline for training.
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': mnist.train.images.astype(np.float32)},
        #x={'x': train_x},
        y=mnist.train.labels.astype(np.float32),
        # y=train_y,
        batch_size=batch_size,
        shuffle=True,
        num_epochs=epochs
    )

    # create input pipeline for evaluation.
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': mnist.train.images[:50].astype(np.float32)},
        #x={'x': test_x},
        y=mnist.train.labels[:50].astype(np.float32),
        # y=test_y,
        batch_size=batch_size,
        shuffle=True,
        num_epochs=1
    )

    # start tensorflow interactiveSession
    sess = tf.InteractiveSession()

    # start the training.
    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn)
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn)

    tf.estimator.train_and_evaluate(model, train_spec, eval_spec)

    # get prediction after training.
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': mnist.test.images.astype(np.float32)},
        y=mnist.test.images.astype(np.float32),
        batch_size=batch_size,
        shuffle=False
    )

    # predict() returns generator
    preds = model.predict(
        input_fn=test_input_fn,
    )

    # this is numpy array
    for pred in preds:
        print(pred['y'])
