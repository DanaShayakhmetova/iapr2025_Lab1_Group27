from src.find_mask import FindMask
from src.visulization import Visualization
from src.utils import *

find_mask = FindMask()
image_path  = "data/data_project/test/L1000891.JPG"
image = read_image(image_path)
hsv = find_mask.to_hsv(image)
mask = find_mask.find_brown_mask(image)
Visualization.show_image(mask)