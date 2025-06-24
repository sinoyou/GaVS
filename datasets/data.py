from PIL import Image
import numpy as np
import torch


def process_projs(proj):
    # pose in dataset is normalised by resolution
    # need to unnormalise it for metric projection
    K = np.eye(3, dtype=np.float32)
    K[0, 0] = proj[0]
    K[1, 1] = proj[1]
    K[0, 2] = proj[2]
    K[1, 2] = proj[3]
    return K


def data_to_c2w(w2c):
    w2c = pose_to_4x4(w2c)
    c2w = np.linalg.inv(w2c)
    return c2w


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')