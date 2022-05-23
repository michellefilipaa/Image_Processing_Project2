import cv2
import numpy as np


class Morphology:
    def __init__(self, path1, path2):
        oranges = cv2.imread(path1, 0)
        orange_tree = cv2.imread(path2)

        # creating an image with only black and white
        _, thresh1 = cv2.threshold(oranges, 127, 255, cv2.THRESH_BINARY)
        _, thresh2 = cv2.threshold(orange_tree, 127, 255, cv2.THRESH_BINARY)


        #self.count_orange(thresh1)
        # the two array inputs are the lower bound and upper bound for the color orange
        self.count_oranges_on_tree(orange_tree, np.array([1, 100, 80]), np.array([22, 256, 256]))

    def count_orange(self, threshold):
        # create a kernal
        kernel = np.ones((2, 2), np.uint8)
        # this removes black distortions from the image
        dilation = cv2.dilate(threshold, kernel, iterations=2)

        cv2.imshow("Dilation", dilation)

        # find all the contours in the image
        contours, other = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        print("The number of oranges in the picture is " + str(len(contours)))
        cv2.waitKey(0)

    def count_oranges_on_tree(self, image, lowerbound, upperbound):
        # create kernels for dilation and erosion
        kernel1 = np.ones((2, 2), np.uint8)
        kernel2 = np.ones((2, 2), np.uint8)

        # first we need to convert from grayscale to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # create a mask
        mask = cv2.inRange(hsv, lowerbound, upperbound)

        # now we need to erode the image
        erosion = cv2.erode(mask, kernel1, iterations=23)
        # and then dilate the eroded image
        dilation = cv2.dilate(erosion, kernel2, iterations=8)

        # find all the contours in the image
        contours, other = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(hsv, contours, -1, (0, 255, 0), 3)

        cv2.imshow("Contours", hsv)

        print("The number of oranges in the picture is " + str(len(contours)))
        cv2.waitKey(0)


oranges = Morphology("images/oranges.jpg", "images/orangetree.jpg")

