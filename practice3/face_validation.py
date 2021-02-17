import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.fftpack import dct
from PIL import Image
from numpy import random


def get_histogram(image, param = 30):
    hist, _ = np.histogram(image, bins=np.linspace(0, 1, param))
    return hist

def get_dft(image, mat_side = 13):
    f = np.fft.fft2(image)
    f = f[0:mat_side, 0:mat_side]
    return np.abs(f)

def get_dct(image, mat_side = 13):
    c = dct(image, axis=1)
    c = dct(c, axis=0)
    c = c[0:mat_side, 0:mat_side]
    return c

def get_gradient(image, n = 2):
    shape = image.shape[0]
    i, l = 0, 0
    r = n
    result = []

    while r <= shape:
        window = image[l:r, :]
        result.append(np.sum(window))
        i += 1
        l = i * n
        r = (i + 1) * n
    result = np.array(result)
    return result

def get_scale(image, scale = 0.35):
    h = image.shape[0]
    w = image.shape[1]
    new_size = (int(h * scale), int(w * scale))
    image = np.array(Image.fromarray(image).resize(new_size))
    return image

