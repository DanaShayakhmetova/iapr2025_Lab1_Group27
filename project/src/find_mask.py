import cv2
import numpy as np
import skimage

class FindMask:
    def __init__(self):
        self.inner_heart_lower = [10, 5, 150]
        self.inner_heart_upper = [50, 70, 225]
        self.brown_lower = [0, 40, 40]
        self.brown_upper = [10, 180, 180]
        self.highlight_lower = [140, 30, 40]
        self.highlight_upper = [200, 150, 150]
        self.Noblesse_inner_circle_lower = [0, 0, 0]
        self.Noblesse_inner_circle_upper = [10,160,120]
        self.inner_heart_min_area = 20000
        self.inner_heart_max_area = 40000
        self.brown_mask_min_area = 0
        self.brown_mask_max_area = None

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
    
    @staticmethod
    def filter_contours_by_length(contours, min_length, max_length):
        filtered_contours = [cnt for cnt in contours if min_length <= cv2.arcLength(cnt, closed=False) <= max_length]
        return filtered_contours
    
    @staticmethod
    def filter_contours_by_area(contours, min_area, max_area):
        filtered_contours = [cnt for cnt in contours if min_area <= cv2.contourArea(cnt) <= max_area]
        return filtered_contours

    def find_by_hsv_threshold(self, hsv, lower, upper):
        """Finding the mask by HSV threshold"""
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        return mask

    def find_Noblesse_inner_circle(self, image):
        """Find the inner circle of the Noblesse Choco"""
        hsv = FindMask.to_hsv(image)
        mask_1 = self.find_by_hsv_threshold(hsv, self.Noblesse_inner_circle_lower, self.Noblesse_inner_circle_upper)
        mask_2 = self.find_by_hsv_threshold(hsv, self.highlight_lower, self.highlight_upper)
        mask_Noblesse = cv2.bitwise_or(mask_1, mask_2)

        contours, _ = cv2.findContours(mask_Noblesse, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        filtered_contours = FindMask.filter_contours_by_length(contours, 800, 1500)
        
        final_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(final_mask, filtered_contours, -1, 255, thickness=cv2.FILLED)  

        kernel = np.ones((20, 20), np.uint8)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)

        return final_mask

    def find_inner_heart(self, image):
        """Find the inner heart shape of the Passion_au_lait chocolate"""
        hsv = FindMask.to_hsv(image)
        inner_heart_mask = self.find_by_hsv_threshold(hsv, self.inner_heart_lower, self.inner_heart_upper)
        contours, _ = cv2.findContours(inner_heart_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        filtered_contours = FindMask.filter_contours_by_area(contours, self.inner_heart_min_area, self.inner_heart_max_area)

        final_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(final_mask, filtered_contours, -1, 255, thickness=cv2.FILLED)  
        return final_mask
    
    def find_highlight_mask(self, hsv):
        """Find the mask for the hightlight region"""
        highlight_mask = self.find_by_hsv_threshold(hsv, self.highlight_lower, self.highlight_upper)
        return highlight_mask
    
    def find_brown_mask(self, hsv):
        mask_brown = self.find_by_hsv_threshold(hsv, self.brown_lower, self.brown_upper)
        return mask_brown

    def find_brown_choco(self, image):
        """Find the mask for the brown choco like Arabia, Noblesse and Triangolo"""
        hsv = FindMask.to_hsv(image)
        rgb = FindMask.to_rgb(image)

        mask_brown = self.find_brown_mask(hsv)
        mask_highlight = self.find_highlight_mask(hsv)
        mask_choco = cv2.bitwise_or(mask_brown, mask_highlight)
        mask_rgb = cv2.bitwise_and(rgb, rgb, mask = mask_choco)
        mask_gray = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2GRAY)

        blurred = cv2.GaussianBlur(mask_gray, (5, 5), 0)
        edges = cv2.Canny(mask_gray, threshold1=10, threshold2=25)
        kernel = np.ones((5, 5), np.uint8)
        edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(edges_closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        filtered_contours = FindMask.filter_contours_by_length(contours, 500, 100000)
        
        final_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(final_mask, filtered_contours, -1, 255, thickness=cv2.FILLED)  

        kernel = np.ones((15, 15), np.uint8)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)
        
        return final_mask
        
    def find_Hough_circle(self, image, range = [250, 350]):
        """Find the circle in the image with a certain size"""
        gray = FindMask.to_gray(image)

        # Detect circles
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=50,
            param1=100,
            param2=50,
            minRadius=range[0],
            maxRadius=range[1]
        )

        # Create an empty mask
        mask = np.zeros_like(gray)

        # Draw the detected circles on the mask
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles[0, :]:
                center = (circle[0], circle[1])
                radius = circle[2]
                cv2.circle(mask, center, radius, 255, thickness=-1)

        return mask, circles
    