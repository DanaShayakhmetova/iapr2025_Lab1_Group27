# src/utils.py

import cv2

def read_image(path):
    """Load the image from the path"""
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Fail to load the image: {path}")
    return image
