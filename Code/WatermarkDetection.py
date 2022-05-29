import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack as f


class WatermarkDetection:
    def __init__(self, path1, path2):
        self.image1 = cv2.imread(path1, 0)
        self.image2 = cv2.imread(path2, 0)
        self.bs = 8

        # compute 2D DCT
        dct1 = self.blockwise_dct(self.image1, 'DCT of the image 1')
        dct2 = self.blockwise_dct(self.image2, 'DCT of the image 2')

        # threshold the non-DCT coefficients
        k_largest1 = self.threshold(self.image1, dct1, 0.01)
        k_largest2 = self.threshold(self.image2, dct2, 0.01)

    def dct_2d(self, image):
        return f.dct(f.dct(image.T, norm='ortho').T, norm='ortho')

    def blockwise_dct(self, image, name):
        blocks = np.zeros(image.shape)
        width, height = image.shape
        # create blocks
        # this is done by dividing the image into blocks of size block_size
        for i in np.r_[:width:self.bs]:
            for j in np.r_[:width:self.bs]:
                blocks[i:i+self.bs, j:j+self.bs] = self.dct_2d(image[i:i+self.bs, j:j+self.bs])

        return blocks

    def threshold(self, non_dct, dct, thresh=0.01):
        non_dct_thresh = non_dct * (abs(non_dct) > (thresh*np.max(non_dct)))

        for i in np.r_[:non_dct_thresh.shape[0]: self.bs]:
            for j in np.r_[:non_dct_thresh.shape[1]: self.bs]:
                    dct[i, j] = 0

        return dct

    def watermark_approximation(self, dct, image, alpha=0.03, K=6):
        watermark_approx = []
        c_cappy = []
        c = []

        for i in range(dct.shape[0]):
            for j in range(dct.shape[1]):
              c_cappy = dct[i:i + self.bs, j:j + self.bs]
              c = image[i:i + self.bs, j:j + self.bs]

              watermark_approx = np.add(watermark_approx, self.getBlockCoefficients(c_cappy, c))

        return watermark_approx


    # def watermark_approximation(self, dct, mean=16.673039316413192, alpha=0.02, K=6):


images = WatermarkDetection("images/Nail_art.jpg", "images/watermarked.png")
plt.show()
