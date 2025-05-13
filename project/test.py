from src.find_mask import FindMask
from src.visulization import Visualization
from src.utils import *

find_mask = FindMask()
image_path  = "data/data_project/train/L1000870.JPG"
image = read_image(image_path)
hsv = find_mask.to_hsv(image)
# mask, circles = find_mask.find_Hough_circle(image)
# Visualization.draw_detected_circle(image, circles)
mask, circles= find_mask.find_Hough_circle(image)
Visualization.draw_detected_circle(image, circles)
# Visualization.show_image_with_hsv(hsv)