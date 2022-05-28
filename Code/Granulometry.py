import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import io, color, measure

## need to change cause its plagaried
class Granulometry:
    def __init__(self, path1):
        self.original_image = cv2.imread(path1, 0)
        contrast, freqencies = self.apply_changes(self.original_image, 3, 5, 20)

        plt.plot(freqencies[:, 0], freqencies[:, 1])
        plt.title("Image Lights Sizes")
        plt.show()

    # this method will apply some changes to the original image
    # 3, 5 20
    def apply_changes(self, image, start, factor, iterations):
        frequencies = np.zeros((iterations, 2))
        # Calculate initial surface area
        surface_area = sum(sum(image))
        for i in range(iterations):
            diameter = start + (i * factor)

            erosion = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))
            dilation = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))

            # Erode the image and then dilate the eroded image
            opening = cv2.morphologyEx(image, cv2.MORPH_ERODE, erosion)
            opening = cv2.morphologyEx(opening, cv2.MORPH_DILATE, dilation)

            image = opening
            final_surface_area = sum(sum(image))
            frequencies[i, 0] = diameter/2
            frequencies[i, 1] = (abs(surface_area - final_surface_area))
            surface_area = final_surface_area

        return opening, frequencies


lights = Granulometry('images/lights.jpg')
