import os
import numpy as np
from glob import glob
from PIL import Image

def read_imgs(dir_path, search_pattern = '*.png'):
    """
    複数画像をnumpyに読み込み、3次元のnumpy.ndarrayを返す。
    https://qiita.com/panda531/items/2e9cec94cd9b399d1725
    
    (n_samples,height,width,channels)
    
    Parameters
    -------
    dir_path : str
        ディレクトリ のパス
    search_pattern : search_pattern
        検索パターン
        ex) '*.png'
    """

    imgs = []
    for file_path in sorted(glob(os.path.join(dir_path, search_pattern))):
        img = np.asarray(Image.open(file_path))
        img_expanded = np.expand_dims(img, axis=0)
        imgs.append(img_expanded)
    
    assert len(imgs) > 0, 'no {} images in {}'.format(search_pattern, dir_path)
    
    result = np.concatenate(imgs, axis=0)
    return result    
