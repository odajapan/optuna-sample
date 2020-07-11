import os
import random
import shutil
from datetime import datetime
from glob import glob

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import get_file
from tensorflow.python.client import device_lib


def gpu_available():
    """
    GPUの有無を確認
    
    Parameters
    -------
    ___ : bool
    GPUの有無
    """
    
    devices = '.'.join([str(x) for x in device_lib.list_local_devices()])
    print(devices)
    return 'gpu' in devices.lower()


def keras_setup(seed=0):
    """
    random seed固定
    メモリ使用量抑制
    # tensorflow2.0 + kerasでGPUメモリの使用量を抑える方法
    # https://qiita.com/studio_haneya/items/4dfaf2fb2ac44818e7e0

    TODO: kerasで学習再現
    Parameters
    -------
    seed : int
        乱数seed
    """
    print('tensorflow version: ', tf.__version__)
    # ランダムシード設定
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # メモリ使用量抑制
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        for k in range(len(physical_devices)):
            tf.config.experimental.set_memory_growth(physical_devices[k], True)
            print('device {}: memory growth:{}'.format(k, tf.config.experimental.get_memory_growth(
                physical_devices[k])))
    else:
        print("Not enough GPU hardware devices available")
        
    # kerasで学習が再現できない人へ
    # https://qiita.com/okotaku/items/8d682a11d8f2370684c9
    # keras2.2 用に修正が必要
    # tf.ConfigProto -> tf.compat.v1.ConfigProto
    # from keras import backend as K -> from tensorflow.keras import backend as K

    os.environ['PYTHONHASHSEED'] = '0'
    session_conf = tf.compat.v1.ConfigProto(
        intra_op_parallelism_threads=1,
        inter_op_parallelism_threads=1
    )

    '''
    # TODO: tf.Sessionがtensorflow 2.2にない。結果の再現性で悩んだたら対応する。
    from tensorflow.keras import backend as K
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)
    '''


def start_training(seed = 0):
    """
    kerasを初期化し実験を開始する

    Parameters
    -------
    seed : int
        乱数seed

    Returns
    -------
    start_time : datetime
        実験の開始時刻
    """
    start_time = datetime.now()
    keras_setup(seed)
    return start_time



def save_results(start_time, model, history, label='mri_segmentation'):
    """
    モデルを保存する
    Parameters
    -------
    start_time : datetime
        実験開始時刻
    model : tensorflow.python.keras.engine.training.Model
        学習モデル
    history : 
        history
    label :
        実験結果を保存するディレクトリのラベル
    Returns
    -------
    save_path : str
        実験結果を保存したディレクトリ 
    """
    
    # Save Model & Results
    save_path = os.path.join('results', start_time.strftime(
        '%Y%m%d%H%M%S_') + label)
    os.makedirs(save_path, exist_ok=True)

    print('[execute time]', str(datetime.now() - start_time))
    with open(os.path.join(save_path, 'execute_time.txt'), 'w') as f:
        f.write(str(datetime.now() - start_time))

    model.save_weights(os.path.join(save_path, 'model.hdf5'))
    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv(os.path.join(save_path, 'history.csv'), index=False)

    return save_path

if __name__ == '__main__':
    print(gpu_available())
