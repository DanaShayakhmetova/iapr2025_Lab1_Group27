#removing blue book background
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

data_path = "D:/sem2/EE451/iapr2025_Lab1_Group27/project/dataset_project_iapr2025"

blue_book = "L1000993.JPG"
path_img = data_path + f"/train/{blue_book}"
img = cv2.imread(path_img)

lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
l2 = clahe.apply(l)
lab = cv2.merge((l2, a, b))
enhanced_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
kernel = np.ones((5,5),np.float32)/25
enhanced_img = cv2.filter2D(enhanced_img, -1, kernel)
enhanced_img_rgb = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB)

mask_r =np.logical_and(np.logical_and(enhanced_img[:,:,2]>160, enhanced_img[:,:,1]<100),enhanced_img[:,:,0]<100)
mask_g = np.logical_and(np.logical_and(enhanced_img[:,:,2]<100, enhanced_img[:,:,1]>160),enhanced_img[:,:,0]<100)
mask_b = np.logical_and(np.logical_and(enhanced_img[:,:,2]<100, enhanced_img[:,:,1]<160),enhanced_img[:,:,0]>80)
mask_y =np.logical_and(np.logical_and(enhanced_img[:,:,2]>100, enhanced_img[:,:,1]>100),enhanced_img[:,:,0]<100)
mask_w =np.logical_and(np.logical_and(enhanced_img[:,:,2]>140, enhanced_img[:,:,1]>140),enhanced_img[:,:,0]>140)
mask_bk = np.logical_and(np.logical_and(enhanced_img[:,:,2]<45, enhanced_img[:,:,1]<45),enhanced_img[:,:,0]<45)

mask = np.logical_or(np.logical_or(np.logical_or(mask_r, mask_g), np.logical_or(mask_b, mask_y)),np.logical_or(mask_w,mask_bk))
#mask = np.logical_or(np.logical_or(mask_r, mask_g), np.logical_or(mask_b, mask_y))

kernel = np.ones((30, 30), np.uint8)
mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)

kernel = np.ones((100, 100), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

mask = 1-mask

img_masked = np.zeros_like(enhanced_img_rgb)
for i in range(3):
    img_masked[:,:,i] = enhanced_img[:,:,i]*mask

img_masked[img_masked==0] = 200

img_masked_rgb = cv2.cvtColor(img_masked, cv2.COLOR_BGR2RGB)

fig, ax = plt.subplots(1,3)
ax[0].imshow(img_masked_rgb)
ax[1].imshow(mask, cmap='gray')
ax[2].imshow(enhanced_img_rgb)