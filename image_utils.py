import imageio
from PIL import Image
import numpy as np
import os


def read_images(subdir: str):

    """
    reads images from a given subdir
    """

    imgs_subdir = os.curdir + subdir
    out = []
    for filename in os.listdir(imgs_subdir):
        with open(imgs_subdir+filename, 'r'):
            img = imageio.imread(imgs_subdir+filename)           
            out.append(np.asarray(img))
    return np.asarray(out)


def sample_to_image(pred: np.ndarray, img_name: str):

    """
    save generated image
    """
    im = Image.fromarray(pred, mode="L")
    imageio.imsave(img_name, im)