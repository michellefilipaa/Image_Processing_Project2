import cv2
import numpy as np


class Morphology:
    def __init__(self, path1, path2):
        oranges = cv2.imread(path1, 0)
        orange_tree = cv2.imread(path2)

        # creating an image with only black and white
        _, thresh1 = cv2.threshold(oranges, 127, 255, cv2.THRESH_BINARY)

        # self.count_orange(thresh1)

        oranges_on_tree = self.isolate_oranges(orange_tree)
        #cv2.imshow("oranges_on_tree", oranges_on_tree)
        #cv2.waitKey()

        self.count_oranges_on_tree(oranges_on_tree)

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

    # this method will take away most of the background from the image
    def isolate_oranges(self, image):
        lowerBound = np.array([1, 100, 80])
        upperBound = np.array([22, 256, 256])

        mask = cv2.inRange(image, lowerBound, upperBound)
        # morphology
        only_oranges = cv2.bitwise_and(image, image, mask=mask)
        return cv2.cvtColor(only_oranges, cv2.COLOR_BGR2GRAY)

    def count_oranges_on_tree(self, image):
        # threshold the image
        _, thresh = cv2.threshold(image, 167, 255, cv2.THRESH_BINARY)
        cv2.imshow("Threshold", thresh)
        # morphology
        erosion_structure = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilation_structure = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (40, 40))

        maskOpen = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, dilation_structure)
        maskClose = cv2.morphologyEx(maskOpen, cv2.MORPH_ERODE, erosion_structure)

        conts, h = cv2.findContours(maskClose, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in conts:
            area = cv2.contourArea(c)
            if area > 100:
                print("The number of oranges on the tree is " + str(len(conts)))
                break

        cv2.waitKey(0)
        cv2.drawContours(image, conts, -1, (255, 0, 0), 3)

        cv2.imshow("Final", image)
        cv2.waitKey(0)


oranges = Morphology("images/oranges.jpg", "images/orangetree.jpg")

