import cv2
import numpy as np
import skimage

class FindMask:
    def __init__(self):
        self.inner_heart_lower = []
        self.inner_heart_upper = []

    @staticmethod
    def to_gray(image):
        """Convert the image to grayscale"""
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    @staticmethod
    def to_hsv(image):
        """Convert the image to HSV color space"""
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    def find_by_hsv_threshold(self, hsv, lower, upper):
        """Finding the mask by HSV threshold"""
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        return mask

    def find_inner_heart_mask(self, image):
        hsv = FindMask.to_hsv(image)
        return self.find_by_hsv_threshold(hsv, self.inner_heart_lower, self.inner_heart_upper)


        
