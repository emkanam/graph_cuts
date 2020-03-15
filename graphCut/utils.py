import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from graphCut.graphModel import GraphModel

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


if __name__ == '__main__':
    # read and add noise to image
    img = cv.imread(data_path+'img_1.png', cv.IMREAD_GRAYSCALE)
    noisy_img = add_gaussian_noise(img, std=10)
    bin_img = binary_image(img)

    # show image
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(img, cmap='gray')
    axs[1].imshow(bin_img, cmap='gray')
    plt.show()

    gm = GraphModel(img)
    print(gm.G.in_edges("target"))
