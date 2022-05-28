import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.util import random_noise
from sklearn.metrics import mean_squared_error


class ImageDegradation:
    def __init__(self, path):
        self.noise = 0
        self.image = cv2.imread(path)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        # dispay original image
        # plt.figure('Original Image')
        # plt.imshow(self.image)
        self.width, self.height = self.image.shape[:2]

        # for each color channel, apply motion blur to it
        (blue, green, red) = cv2.split(self.image)
        blue_blurred, blue_H = self.motion_blur(blue)
        green_blurred, green_H = self.motion_blur(green)
        red_blurred, red_H = self.motion_blur(red)
        # now merge the channels to make one image
        # and normalize the image
        blurred_img = cv2.merge((blue_blurred, green_blurred, red_blurred))
        blurred_img = blurred_img / np.max(blurred_img)

        gaus_and_motion = random_noise(blurred_img, mode='gaussian', mean=0.1, var=0.8)
        self.gaussian = random_noise(self.image, mode='gaussian', mean=0.1, var=0.8)
        # self.plot_motion_and_gaussian(blurred_img, gaus_and_motion)

        (blue2, green2, red2) = cv2.split(blurred_img)
        blue_deblurred = self.direct_inverse_filtering(blue2, blue_H)
        green_deblurred = self.direct_inverse_filtering(green2, green_H)
        red_deblurred = self.direct_inverse_filtering(red2, red_H)
        deblurred_motion = cv2.merge((blue_deblurred, green_deblurred, red_deblurred))
        deblurred_motion = deblurred_motion / np.max(deblurred_motion)

        # plt.figure('Deblurred Motion Image')
        # plt.imshow(deblurred_motion)

        # de_motion_and_gaus = self.direct_inverse_filtering(gaus_and_motion, blurred_img[1])
        # plt.figure('Deblurred Motion and Gaussian Image')
        # plt.imshow(de_motion_and_gaus, cmap='gray', vmin=0, vmax=255)

        (blue3, green3, red3) = cv2.split(deblurred_motion)
        blue_wiener = self.wiener_filtering(blue, blue3, blue_H, "Motion Blur")
        green_wiener = self.wiener_filtering(green, green3, green_H, "Motion Blur")
        red_wiener = self.wiener_filtering(red, red3, red_H, "Motion Blur")
        wiener_motion = cv2.merge((blue_wiener, green_wiener, red_wiener))
        wiener_motion = wiener_motion / np.max(wiener_motion)

        (blue4, green4, red4) = cv2.split(gaus_and_motion)
        blue_wiener2 = self.wiener_filtering(blue, blue4, blue_H, "Gaussian Blur")
        green_wiener2 = self.wiener_filtering(green, green4, green_H, "Gaussian Blur")
        red_wiener2 = self.wiener_filtering(red, red4, red_H, "Gaussian Blur")
        wiener_gaus = cv2.merge((blue_wiener2, green_wiener2, red_wiener2))
        wiener_gaus = wiener_gaus / np.max(wiener_gaus)
        #self.wiener_filtering(self.image, gaus_and_motion, blurred_img[1], 'Motion and Gaussian Blur')

    def motion_blur(self, channel, alpha=20, beta=12):
        width, height = channel.shape
        width_2 = int(width/2)
        height_2 = int(height/2)

        # create a width x height mesh grid
        [u, v] = np.mgrid[-width_2:width_2, -height_2:height_2]

        # calculate u and v
        u = 2 * u / self.width
        v = 2 * v / self.height

        # define H
        H = np.sinc((u * alpha) + (v * beta)) * np.exp(-1j * np.pi * (u * alpha + v * beta))

        # find the FFT of the image and display it
        F = np.fft.fft2(channel)

        # shift the FFT
        F_shift = np.fft.fftshift(F)

        # apply the filter
        blurred_img = F_shift * H

        return np.abs(np.fft.ifft2(blurred_img)), H

    def plot_motion_and_gaussian(self, blurred_img, gaus_img):
        plt.subplots(1, 3, figsize=(10, 3.8))
        plt.subplot(1, 3, 1)
        plt.imshow(self.image)
        plt.title('Original Image')

        plt.subplot(1, 3, 2)
        plt.imshow(blurred_img)
        plt.title('Motion Blur')

        plt.subplot(1, 3, 3)
        plt.imshow(gaus_img)
        plt.title('Gaussian Blur')

    def direct_inverse_filtering(self, channel, H):
        F = np.fft.fft2(channel)
        # apply the inverse filter
        deblurred = F * (1/H)

        return np.abs(np.fft.ifft2(deblurred))

    def wiener_filtering(self, og_img, blur_img, H, name):
        # find the power spectrums of the original and blurred images
        noise_power_spectrum = (np.fft.fft2(blur_img)) ** 2
        power_spectrum = (np.fft.fft2(og_img)) ** 2

        abs_H_2 = np.abs(H) ** 2  # to make it neater I create this variable

        # find K to approximate the ratio between the power spectrums
        K = 1  # if it is only motion blur
        if name != 'Motion Blur':
            K = np.sum(noise_power_spectrum) / np.sum(power_spectrum)

        H_wiener = (1 / H) * (abs_H_2 / (abs_H_2 + (noise_power_spectrum / power_spectrum)*K))

        fft_shift = np.fft.fftshift(np.fft.fft2(blur_img))
        result = np.fft.ifft2(fft_shift * H_wiener)

        plt.figure('Wiener Filtering')
        plt.imshow(np.abs(result))
        plt.title('Wiener Filtering of ' + name)

        return H_wiener


bird = ImageDegradation('images/bird.jpg')
plt.show()
