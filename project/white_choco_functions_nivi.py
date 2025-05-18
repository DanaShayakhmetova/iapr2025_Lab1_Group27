import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from skimage import color, exposure
from skimage.morphology import closing, square, dilation, disk, remove_small_objects, opening
import skimage
from skimage import io, color, filters, feature
import os
from skimage.measure import regionprops, label

## run segment_white_choco with the argument as the path to img; returns the masked chocolates

def mask_background(img_rgb, threshs=[100,100,100], hue_thresh_max=0.1):
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    blur_h = cv2.GaussianBlur(img_hsv[:,:,0],(81,81),100)
    mask_h = (blur_h/255<hue_thresh_max).astype(np.uint8)
    kernel = np.ones((25, 25), np.uint8)
    mask_h = cv2.erode(mask_h, kernel, iterations=1)
    mask_h = cv2.dilate(mask_h, kernel, iterations=2)

    mask_rgb = np.ones_like(img_rgb[:,:,0])
    for i in range(3):
        blur = cv2.GaussianBlur(img_rgb[:,:,i],(81,81),100)
        mask = (blur>threshs[i]).astype(np.uint8)
        kernel = np.ones((100, 100), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)
        mask_rgb = np.logical_and(mask_rgb,mask)
    mask_rgb = np.logical_and(mask_rgb,mask_h)
    masked_bg = np.zeros_like(img_rgb)
    for i in range(3):
        masked_bg[:,:,i] = mask_rgb*img_rgb[:,:,i]
    return masked_bg, mask_rgb

def mask_hsv_white(img_rgb, val_sq_thresh_min=0.65,sat_thresh_max=0.1):
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    #value thresholding
    val_choco=img_hsv[:,:,2]/255
    val_choco_thresh = val_choco*val_choco > val_sq_thresh_min
    #saturation thresholding
    sat_choco = img_hsv[:,:,1]/255
    sat_choco_thresh = sat_choco< sat_thresh_max

    #combine to get white chocolate mask
    arr = np.clip(val_choco_thresh+1-sat_choco_thresh,0,1)
    masked_choco = np.zeros_like(img_rgb)
    for i in range(3):
        masked_choco[:,:,i] = arr*img_rgb[:,:,i]
    
    return masked_choco, arr


def segment_white_chocos(img_path):
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    masked_bg,_ = mask_background(img_rgb)
    masked_choco,_ = mask_hsv_white(masked_bg)

    return masked_choco

#eg implementation
data_path = data_path = "D:/sem2/EE451/iapr2025_Lab1_Group27/project/dataset_project_iapr2025"

path = data_path + "/train/L1000952.JPG"
masked_choco = segment_white_chocos(path)
plt.imshow(masked_choco)
