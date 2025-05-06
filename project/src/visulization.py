import cv2
import matplotlib.pyplot as plt
import numpy as np

class Visualization:
    @staticmethod
    def show_image(image, title="Image", size=(6,6)):
        plt.figure(figsize=size)
        plt.imshow(image)
        plt.title(title)
        plt.axis('off')
        plt.show()