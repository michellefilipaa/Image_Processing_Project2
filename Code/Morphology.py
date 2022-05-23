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
        self.count_oranges_on_tree(orange_tree, np.array([1, 160, 100]), np.array([18, 255, 255]))

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
        kernel = np.ones((2, 2), np.uint8)

        # first we need to convert from grayscale to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # create a mask
        mask = cv2.inRange(hsv, lowerbound, upperbound)

        # show the image as only the orange objects
        result = cv2.bitwise_and(image, image, mask=mask)

        # first convert from hsv to grayscale
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

        # now apply thresholding so the image is only black and white
        _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        cv2.imshow("Threshold", threshold)

        dilation = cv2.dilate(threshold, kernel, iterations=10)
        cv2.imshow("Dilation", dilation)

        cv2.waitKey()



oranges = Morphology("images/oranges.jpg", "images/orangetree.jpg")

