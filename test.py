import pickle
from optuna.integration.tensorflow import TensorFlowPruningHook
from mnist_estimator import MnistEstimator
import plotly
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import optuna
from optuna import trial
import os
import shutil
import sys
from glob import glob
from datetime import datetime
sys.path.append('../')
#from mnist import train


#from utils import *

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


model_dir = './model'
batch_size = 1000
epochs = 100

make_empty_dir(model_dir)

mnist = input_data.read_data_sets("./data/mnist", one_hot=True)
train_x = mnist.train.images.reshape(-1, 28, 28, 1)
train_y = mnist.train.labels
test_x = mnist.test.images.reshape(-1, 28, 28, 1)
test_y = mnist.test.labels

#train_x, train_y = train_x[:10000], train_y[:10000]
#test_x, test_y = test_x[:1000], test_y[:1000]

estimator = MnistEstimator()
# estimator.train(train_x, train_y, test_x, test_y,
#                epochs=epochs, batch_size=batch_size, model_dir=model_dir)


# Implement integration for TensorFlow's Estimator API #292
# https://github.com/optuna/optuna/pull/292
# https://github.com/optuna/optuna/blob/b83f771014060fe44f5411c419a9c6209218382a/optuna/integration/tensorflow.py

def objective(trial):
    save_steps = 50
    estimator.TrainParams['learning_rate'] = trial.suggest_loguniform(
        "learning_rate", 1e-5, 1e-2)
    clf = estimator.get_model(model_dir)
    train_spec, eval_spec, train_input_fn, eval_input_fn = \
        estimator.get_train_and_eval_specs_inputs(
            train_x, train_y, test_x, test_y, epochs, batch_size)

    # Create hooks
    early_stopping_hook = tf.estimator.experimental.stop_if_no_decrease_hook(
        clf, "loss", save_steps)
    optuna_pruning_hook = TensorFlowPruningHook(
        trial=trial,
        estimator=clf,
        metric="loss",
        run_every_steps=10,
    )
    hooks = []
    #hooks = [optuna_pruning_hook]
    hooks = [early_stopping_hook, optuna_pruning_hook]

    # Run training and evaluation
    tf.estimator.train_and_evaluate(clf, train_spec, eval_spec)
    result = clf.evaluate(input_fn=eval_input_fn)
    #result = clf.evaluate(input_fn=eval_input_fn, steps=100)
    print(result)
    return result['loss']
    #accuracy = result["accuracy"]
    # return 1.0 - accuracy


study = optuna.create_study()
study.optimize(objective, n_trials=10)

print(study.best_params)
print(study.best_value)
print(study.best_trial)

for t in study.trials:
    print(t.number, t.params, 1-t.value)

for t in study.trials:
    print(t)

with open('./study.pickle', 'wb') as wb:
    pickle.dump(study, wb)
