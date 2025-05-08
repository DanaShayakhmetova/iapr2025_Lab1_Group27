import cv2
import numpy as np
import skimage

class FindMask:
    def __init__(self):
        self.inner_heart_lower = [20, 5, 150]
        self.inner_heart_upper = [50, 70, 225]
        self.brown_lower = [0, 60, 60]
        self.brown_upper = [15, 200, 200]

    @staticmethod
    def to_gray(image):
        """Convert the image to grayscale"""
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    @staticmethod
    def to_hsv(image):
        """Convert the image to HSV color space"""
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    @staticmethod
    def to_rgb(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def find_by_hsv_threshold(self, hsv, lower, upper):
        """Finding the mask by HSV threshold"""
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        return mask

    def find_inner_heart_mask(self, image):
        """Find the inner heart shape of the Passion_au_lait chocolate"""
        hsv = FindMask.to_hsv(image)
        return self.find_by_hsv_threshold(hsv, self.inner_heart_lower, self.inner_heart_upper)
    
    def find_brown_mask(self, image):
        """Find the mask for the brown choco like Arabia, Noblesse and Triangolo"""
        hsv = FindMask.to_hsv(image)
        rgb = FindMask.to_rgb(image)
        mask_brown = self.find_by_hsv_threshold(hsv, self.brown_lower, self.brown_upper)
        mask_rgb = cv2.bitwise_and(rgb, rgb, mask = mask_brown)
        mask_gray = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(mask_gray, (5, 5), 0)
        edges = cv2.Canny(blurred, threshold1=10, threshold2=25)
        kernel = np.ones((5, 5), np.uint8)
        edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        min_length = 1500
        filtered_contours = [cnt for cnt in contours if cv2.arcLength(cnt, closed=False) > min_length]
        
        final_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(final_mask, filtered_contours, -1, 255, thickness=cv2.FILLED)  

        return final_mask

        
