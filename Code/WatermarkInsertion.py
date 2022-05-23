import cv2
import matplotlib.pyplot as plt
from scipy import fftpack as f
import numpy as np


class WatermarkInsertion:
    def __init__(self, path, block_size=8):
        self.image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        self.width, self.height = self.image.shape
        self.bs = block_size

    # from lab 6
    def dct_2d(self, image):
        return f.dct(f.dct(image.T, norm='ortho').T, norm='ortho')

    # from lab 6
    def idct_2d(self, image):
        return f.idct(f.idct(image.T, norm='ortho').T, norm='ortho')

    def blockwise_dct(self, image, name):
        blocks = np.zeros(self.image.shape)

        # create blocks
        # this is done by dividing the image into blocks of size block_size
        for i in np.r_[:self.width:self.bs]:
            for j in np.r_[:self.width:self.bs]:
                blocks[i:i+self.bs, j:j+self.bs] = self.dct_2d(image[i:i+self.bs, j:j+self.bs])

        # display the image DCT
        plt.figure(name)
        plt.subplot(1, 2, 1)
        plt.imshow(blocks, vmax=np.max(blocks)*0.01, vmin=0)
        title = [self.bs, 'x', self.bs, ' DCT of the image']
        plt.title(''.join(map(str, title)))

        # display a DCT block
        pos = np.random.randint(0, blocks.shape[0] - self.bs)
        plt.subplot(1, 2, 2)
        plt.imshow(blocks[pos:pos+self.bs, pos:pos+self.bs], vmax=np.max(blocks)*0.01, vmin=0)
        title2 = [self.bs, 'x', self.bs, ' DCT of a block']
        plt.title(''.join(map(str, title2)))

        return blocks

    def blockwise_idct(self, dct, name):
        image = np.zeros(dct.shape)

        # perform the same blockwise transformation as in blockwise_dct except invert the process
        # this will give us the original image if the threshold is at an appropriate value
        for i in np.r_[:self.width: self.bs]:
            for j in np.r_[:self.height: self.bs]:
                image[i:(i + self.bs), j:(j + self.bs)] = self.idct_2d(dct[i:(i + self.bs), j:(j + self.bs)])

        if name == 'DCT after thresholding':
            # display the image IDCT and original image
            plt.figure(name)
            plt.subplot(1, 2, 1)
            plt.imshow(self.image, cmap='gray')
            plt.title('Original image')

            plt.subplot(1, 2, 2)
            plt.imshow(image, cmap='gray')
            title = [self.bs, 'x', self.bs, ' IDCT of the image']
            plt.title(''.join(map(str, title)))

        else:
            # display the image IDCT
            plt.figure(name)
            plt.imshow(image, cmap='gray')
            plt.title('Watermarked image')

        return image

    def threshold(self, dct, thresh):
        dct_thresh = dct * (abs(dct) > (thresh*np.max(dct)))

        plt.figure(2)
        plt.imshow(dct_thresh, vmax=np.max(dct)*0.01, vmin=0)
        title = ["Threshold  of ", thresh, " applied to the DCT"]
        plt.title(''.join(map(str, title)))

        return dct_thresh, 64 - np.floor(thresh*64)

    def watermark(self, dct, var=5, mean=0, alpha=0.02):
        random_numbers = np.random.normal(mean, var, size=(8, 8))

        watermarked_img = dct
        # check if the block is = 0, if it is then add a random number to the block
        for i in np.r_[:self.width: self.bs]:
            for j in np.r_[:self.height: self.bs]:
                # If the sum of the block is not 0 then add a random number to the block
                block = watermarked_img[i:(i + self.bs), j:(j + self.bs)]
                if np.abs(np.sum(block)) != 0:
                    # accessing each element of the block
                    for k in range(block.shape[0]):
                        for l in range(block.shape[1]):
                            watermarked_img[i+k, j+l] = block[k, l] + (1 + alpha*random_numbers[k, l])

        print(np.mean(watermarked_img)) #16.673039316413192

        return watermarked_img

    def difference_img(self, img1, img2):
        # the width and height of the two images have to be the same for this to work
        if img1.shape != img2.shape:
            print("The images have different dimensions")

            return None

        width, height = img1.shape
        diff_img = np.zeros(img1.shape)
        # calculate the difference between each pixel of the two images
        for i in range(width):
            for j in range(height):
                diff_img[i, j] = img1[i, j] - img2[i, j]

        return diff_img

    def compare_all(self, og_img, watermarked_img, diff_img):
        # compare the original image with the watermarked image
        plt.figure('Difference between original and watermarked image')
        plt.subplot(1, 3, 1)
        plt.imshow(og_img, cmap='gray')
        plt.title('Original image')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(watermarked_img, cmap='gray')
        plt.title('Watermarked image')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(diff_img, cmap='gray')
        plt.title('Difference')
        plt.axis('off')

    def img_histogram(self, image1, image2):
        plt.figure('Histograms')
        plt.subplot(1, 3, 1)
        hist1 = image1.flatten()
        hist2 = image2.flatten()
        plt.ylim(0, 10000)
        plt.hist(hist1, bins=255)

        plt.subplot(1, 3, 2)
        plt.ylim(0, 10000)
        plt.hist(hist2, bins=255)

        plt.subplot(1, 3, 3)
        hist3 = hist1 - hist2
        plt.hist(hist3, bins=255)


nails = WatermarkInsertion('images/Nail_art.jpg')

dct = nails.blockwise_dct(nails.image, 'Original DCT')
dct_thresh, threshold = nails.threshold(dct, 0.01)
idct = nails.blockwise_idct(dct_thresh, 'DCT after thresholding')
watermark = nails.watermark(dct_thresh, threshold)

plt.figure('Watermark')
plt.imshow(watermark, cmap='gray')
idct_watermark = nails.blockwise_idct(watermark, 'Watermarked IDCT')

diff_img = nails.difference_img(idct, idct_watermark)
nails.compare_all(nails.image, idct_watermark, diff_img)
nails.img_histogram(nails.image, idct_watermark)
plt.show()
