import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

data_path = '../data/'


def add_gaussian_noise(img, mean=0, std=10):
    shape = img.shape
    noise = np.zeros(shape, np.float)
    cv.randn(noise, mean, std)
    noise_img = img + noise
    return noise_img


if __name__ == '__main__':
    # read and add noise to image
    img = cv.imread(data_path+'img_1.png', cv.IMREAD_GRAYSCALE)
    noisy_img = add_gaussian_noise(img, std=10)

    # show image
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(img, cmap='gray')
    axs[1].imshow(noisy_img, cmap='gray')
    plt.show()
