import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import io, color, measure


class Granulometry:
    def __init__(self, path1):
        self.original_image = cv2.imread(path1, 0)
        self.apply_changes(self.original_image)

    # this method will apply some changes to the original image
    def apply_changes(self, image):
        # convert the image to binary
        ret, threshold = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # dilate and erode the image to clean it up
        kernel = np.ones((5, 5), np.uint8)
        eroded = cv2.erode(threshold, kernel, iterations=1)
        dilated = cv2.dilate(eroded, kernel, iterations=1)

        # sets true for all pixels with a value
        mask = dilated == 255
        print(mask)


lights = Granulometry('images/lights.jpg')