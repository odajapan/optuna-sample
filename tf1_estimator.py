import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from abc import ABCMeta, abstractmethod


class AbstractEstimator(metaclass=ABCMeta):
    def __init__(self):
        # configuration of the model
        self.RunConfig = tf.estimator.RunConfig(
            save_checkpoints_steps=1000,
            keep_checkpoint_max=5,
            log_step_count_steps=1000,
            save_summary_steps=1000,
        )

        self.TrainParams = {
            'learning_rate': 0.001
        }

    @abstractmethod
    def get_network(self, input):
        pass

    @abstractmethod
    def get_loss_and_optimizer(self, y, labels, learning_rate=0.001):
        pass

    def get_model(self, model_dir='./model'):
        return tf.estimator.Estimator(
            model_dir=model_dir,
            model_fn=self.model_fn,
            params=self.TrainParams,
            config=self.RunConfig
        )

    def get_train_and_eval_specs_inputs(self, train_x, train_y, val_x, val_y, epochs, batch_size):
        # create input pipeline for training.
        train_input_fn = self.get_input(train_x, train_y, epochs, batch_size)
        eval_input_fn = self.get_input(val_x, val_y, 1, batch_size)

        # start the training.
        train_spec = tf.estimator.TrainSpec(
            input_fn=train_input_fn, max_steps=50)
        eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn)

        return train_spec, eval_spec, train_input_fn, eval_input_fn

    def train(self, train_x, train_y, test_x, test_y, val_x=None, val_y=None, model_dir='./model', batch_size=1000, epochs=3):
        if val_x == None:
            data_len = int(len(test_x) * 0.1)
            val_x = test_x[:data_len]
            val_y = test_y[:data_len]

        # create model
        model = self.get_model(model_dir)

        train_spec, eval_spec, _, _ = self.get_train_and_eval_specs_inputs(train_x=train_x, train_y=train_y,
                                                                           val_x=val_x, val_y=val_y,
                                                                           epochs=epochs, batch_size=batch_size
                                                                           )

        tf.estimator.train_and_evaluate(model, train_spec, eval_spec)

        # get prediction after training.
        # predict() returns generator
        test_input_fn = self.get_input(
            test_x, test_y, 1, batch_size, shuffle=False)

        preds = model.predict(input_fn=test_input_fn)

        # this is numpy array
        for pred in preds:
            print(pred['y'])

    def get_loss_and_optimizer(self, y, labels, learning_rate=0.001):
        # get loss
        loss = tf.losses.mean_squared_error(y, labels)

        optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
        train_op = optimizer.minimize(
            loss, global_step=tf.train.get_global_step())

        return loss, train_op

    def model_fn(self, features, labels, mode, params):

        # get tensor input
        input = features['x']
        y = self.get_network(input)

        # returns pred
        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {
                'y': y
            }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)

        # calc loss
        learning_rate = params['learning_rate']
        loss, optimizer = self.get_loss_and_optimizer(y, labels, learning_rate)

        # mode for evaluation
        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(mode, loss=loss)

        # create train_op
        assert mode == tf.estimator.ModeKeys.TRAIN

        # summaries to be shown in tensorboard
        tf.summary.scalar('train_loss', loss)

        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=optimizer)

    def get_input(self, x, y, epochs, batch_size, shuffle=True):
        return tf.estimator.inputs.numpy_input_fn(
            x={'x': x},
            y=y,
            num_epochs=epochs,
            batch_size=batch_size,
            shuffle=shuffle,
        )
