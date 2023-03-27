import cv2
import numpy as np


# Линейное отображение
def linear_mapping(img):
    return (img - img.min()) / (img.max() - img.min())


# Препроцессинг изображения
def pre_process(img):
    height, width = img.shape
    img = np.log(img + 1)
    img = (img - np.mean(img)) / (np.std(img) + 1e-5)
    window = hanning_window(height, width)
    img = img * window
    return img


# Окно Хеннинга
def hanning_window(height, width):
    win_col = np.hanning(width)
    win_row = np.hanning(height)
    mask_col, mask_row = np.meshgrid(win_col, win_row)
    win = mask_col * mask_row
    return win


# Аугментация
def random_warp(img):
    a = -180 / 16
    b = 180 / 16
    r = a + (b - a) * np.random.uniform()
    matrix_rot = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), r, 1)
    img_rot = cv2.warpAffine(np.uint8(img * 255), matrix_rot, (img.shape[1], img.shape[0]))
    img_rot = img_rot.astype(np.float32) / 255
    return img_rot
