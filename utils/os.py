import zipfile
import os
import shutil
from glob import glob
from tensorflow.keras.utils import get_file

def download_zip(origin, cache_subdir, dst):
    """
    zipファイルをダウンロードして展開する。
    https://keras.io/ja/utils/data_utils/
    Parameters
    ----------
    origin : str
        ファイルの置かれているURL
    cache_subdir : str
        キャッシュ先のディレクトリ
    dst : str
        zipの展開先
    """

    path = get_file(os.path.basename(origin), origin=origin,
                    cache_subdir=cache_subdir)
    os.makedirs(dst, exist_ok=True)
    with zipfile.ZipFile(path) as existing_zip:
        existing_zip.extractall(dst)


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