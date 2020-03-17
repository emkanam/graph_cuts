import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from graphModel import GraphModel
from alphaExpansion import alpha_expansion

data_path = '../data/'


def add_gaussian_noise(_img, mean=0, std=10):
    shape = _img.shape
    noise = np.zeros(shape, np.int)
    cv.randn(noise, mean, std)
    _noisy_img = _img + noise
    return _noisy_img


def binary_image(_img):
    _bin_img = _img/255.0
    _bin_img = np.around(_bin_img, decimals=0)
    return _bin_img


def level_image(_img, _levels):
    h, w = _img.shape
    n = len(_levels)
    _levels = np.array(_levels)
    # _levels = _levels.reshape((-1, 1))
    cat_img = np.repeat(_img.reshape((h, w, 1)), n, axis=2)
    cat_img = np.abs(cat_img - _levels)
    cat_img = np.argmin(cat_img, axis=2)

    res = np.zeros(_img.shape)
    for i in range(n):
        res[cat_img == i] = _levels[i]
    return res


def show_images(img1, img2):
    # show image
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(img1, cmap='gray')
    axs[1].imshow(img2, cmap='gray')
    plt.show()


if __name__ == '__main__':
    # read and add noise to image
    img = cv.imread(data_path+'img_1.png', cv.IMREAD_GRAYSCALE)
    img = img.astype(np.int64)
    noisy_img = add_gaussian_noise(img, std=50)
    # show image and noisy image
    show_images(img, noisy_img)

    gm = GraphModel(img, 0)
    levels = [0, 51, 102, 153, 255]
    labels = np.random.choice(levels, size=img.shape)
    gm.init_weights(labels)

    # show alpha-expansion results
    alp_rim = alpha_expansion(img, max_it=50, levels=levels)
    noisy_alp_rim = alpha_expansion(noisy_img, max_it=50, levels=levels)
    show_images(alp_rim, noisy_alp_rim)

    # show level_image results
    lv_rim = level_image(img, levels)
    noisy_lv_rim = level_image(noisy_img, levels)
    show_images(lv_rim, noisy_lv_rim)
