import math

import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.util import random_noise


class ImageDegradation:
    def __init__(self, path):
        self.image = cv2.imread(path, 0)
        # dispay original image
        #plt.figure('Original Image')
        #plt.imshow(self.image)
        self.width, self.height = self.image.shape[:2]

        blurred_img = self.motion_blur(self.image, 16, 12)
        gaus_img = self.gaussian_blur(1.1, 0.8, blurred_img[0])

        self.plot_motion_and_gaussian(blurred_img[0], gaus_img)

        #deblurred = self.direct_inverse_filtering(blurred_img[0], blurred_img[1])
        #plt.figure('Deblurred Image')
        #plt.imshow(deblurred, cmap='gray', vmin=0, vmax=255)

        #self.wiener_filtering(self.image, blurred_img[0], blurred_img[1], 'Motion Blur')
        self.wiener_filtering(self.image, gaus_img, blurred_img[1], 'Motion and Gaussian Blur')

    def motion_blur(self, image, alpha, beta):
        width, height = image.shape
        x = image.astype(np.double)
        width_2 = int(width/2)
        height_2 = int(height/2)

        # create a width x height mesh grid
        [u, v] = np.mgrid[-width_2:width_2, -height_2:height_2]

        # calculate u and v
        u = 2*u/width
        v = 2*v/height

        # define H
        H = np.sinc((u*alpha) + (v*beta)) * np.exp(-1j*np.pi*(u*alpha + v*beta))

        # find the FFT of the image and display it
        F = np.fft.fft2(x)

        # shift the FFT
        F_shift = np.fft.fftshift(F)

        # apply the filter
        blurred_img = F_shift * H

        return np.abs(np.fft.ifft2(blurred_img)), H

    def gaussian_blur(self, mu, sigma, z):
        gaussian = np.random.normal(mu, sigma, z.shape)
        z = z.astype(np.uint8)
        z = z + gaussian

        return z

    def plot_motion_and_gaussian(self, blurred_img, gaus_img):
        plt.subplots(1, 3, figsize=(10, 3.8))
        plt.subplot(1, 3, 1)
        plt.imshow(self.image, cmap='gray')
        plt.title('Original Image')

        plt.subplot(1, 3, 2)
        plt.imshow(blurred_img, cmap='gray')
        plt.title('Motion Blur')

        plt.subplot(1, 3, 3)
        plt.imshow(gaus_img, cmap='gray')
        plt.title('Gaussian Blur')

    def direct_inverse_filtering(self, blur_img, H):
        G = np.fft.fft2(blur_img)
        F_estimate = G/H

        return np.fft.ifft(F_estimate).real

    def wiener_filtering(self, og_img, blur_img, H, name):
        # find the power spectrums of the original and blurred images
        noise_power_spectrum = (np.fft.fft2(blur_img)) ** 2
        power_spectrum = (np.fft.fft2(og_img)) ** 2

        abs_H_2 = np.abs(H) ** 2  # to make it neater I create this variable

        # find K to approximate the ratio between the power spectrums
        K = 1  # if it is only motion blur
        if name != 'Motion Blur':
            K = np.mean(og_img, blur_img)

        H_wiener = (1 / H) * (abs_H_2 / (abs_H_2 + K*(noise_power_spectrum / power_spectrum)))
        fft_shift = np.fft.fftshift(np.fft.fft2(blur_img))
        result = np.fft.ifft2(fft_shift*H_wiener)

        plt.figure('Wiener Filtering')
        plt.imshow(np.abs(result), cmap='gray')
        plt.title('Wiener Filtering of ' + name)

        return H_wiener


bird = ImageDegradation('images/bird.jpg')
plt.show()
