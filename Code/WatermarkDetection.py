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
        dct1 = self.blockwise_dct(self.image1)
        dct2 = self.blockwise_dct(self.image2)

        # threshold the non-DCT coefficients
        k_largest1 = self.threshold(self.image1, dct1, 0.01)
        k_largest2 = self.threshold(self.image2, dct2, 0.01)

        approximation1 = self.watermark_approximation(k_largest1, self.image1)
        similarity1 = self.similarity(approximation1, self.image1)

        if similarity1 >= 0.03:
            print("Image 1 is watermarked")
        else:
            print("Image 1 is not watermarked")

        approximation2 = self.watermark_approximation(k_largest2, self.image2)
        similarity2 = self.similarity(approximation2, self.image2)


        if similarity2 >= 0.03:
            print("Image 2 is watermarked")
        else:
            print("Image 2 is not watermarked")

    def dct_2d(self, image):
        return f.dct(f.dct(image.T, norm='ortho').T, norm='ortho')

    def blockwise_dct(self, image):
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

    def watermark_approximation(self, dct, image, alpha=0.03):
        width, height = image.shape
        watermark_approx = []
        c_cappy = dct.copy()
        c = image

        for i in np.r_[:width:self.bs]:
            for j in np.r_[:width:self.bs]:
                top = c_cappy[i, j] - c[i, j]
                bottom = alpha*c[i, j]
                watermark_approx.append(top/bottom)

        return watermark_approx

    def similarity(self, watermark_approx, watermark_real, watermark_mean=16.673039316413192):
        width, height = watermark_real.shape
        similarity = []
        mean_approx = sum(watermark_approx)/len(watermark_approx)
        for i in np.r_[:width:self.bs]:
            for j in np.r_[:width:self.bs]:

                top = (watermark_approx[i] - mean_approx)*(watermark_real[i, j] - watermark_mean)
                #print('Top ' + str(top))
                bottom1 = (watermark_approx[i] - mean_approx)**2
                #print('Bottom1 ' + str(bottom1))

                bottom2 = (watermark_real[i] - watermark_mean)**2
                similarity.append(top/np.sqrt(bottom1*bottom2))

        return np.sum(similarity)


images = WatermarkDetection("images/Nail_art.jpg", "images/watermarked.png")
plt.show()
