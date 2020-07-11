import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


def resize3d(img, method = tf.image.ResizeMethod.BILINEAR, width=48, height=48):
    """
    3D画像の幅と高さを変更する

    Parameters
    ----------
    
    Returns
    -------
    img : numpy.darray
        変換した画像
    """
    
    def _resize(img, method = tf.image.ResizeMethod.BILINEAR, width=48, height=48):
        img = img[:,:,np.newaxis]
        new_img = tf.image.resize(img, (width, height), method=method)
        return new_img.numpy().transpose(2, 0, 1)
    
    return np.vstack([_resize(x, method, width, height) for x in img])



def create_animation_gif(src, dst, title, figsize=(5, 5),duration=40):
    """
    3D画像からアニメーションgifを生成する

    Parameters
    ----------
    
    
    
    """
    imgs=[]
    
    shapes = [v.shape for k, v in src.items()]
    _s = shapes[0]
    for s in shapes:
        assert _s == s, '{} and {} are not equal'.format(_s, s)
    
    frame_count = _s[0]
    cols = len(shapes)
    buf_img_name = 'animation_buf.png'

    for i in range(0, frame_count):
        plt.figure(figsize=figsize)
            
        for index, (label, v) in enumerate(src.items()):
            plt.subplot(1, cols, index + 1)    
            plt.title(label)
            plt.imshow(v[i])
        
        plt.savefig(buf_img_name)
        plt.close()
        _img = Image.open(buf_img_name)
        draw = ImageDraw.Draw(_img)
        text = '{} ({}/{})'.format(title, i, frame_count)
        draw.text((0, 0), text, fill=(0, 0, 0, 0))
        imgs.append(_img)

    imgs[0].save(dst, save_all=True, append_images=imgs[1:], optimize=False, duration=duration, loop=0)
    os.remove(buf_img_name)

    

if __name__ == '__main__':
    print('image3d.py')
    save_anmation_gif()
