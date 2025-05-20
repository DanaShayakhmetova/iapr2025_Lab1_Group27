# this was in here before and idk what to do with it lol 
# testing the ref IDS
# from src.constants import ID2NAME, NAME2ID
# print(ID2NAME[4])  # should say 'Jelly_black'

####################################################################################################
#Imports 
# general 
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math 
import pandas as pd
import os
# classifier model 
from skimage.feature import hog, local_binary_pattern
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import random # For augmentation
import joblib
# CSV 
from collections import Counter

####################################################################################################

# DoG Segmentation 
# I commented out the plotting. It was used for debugging. 
# Area thresholds were found by analyzing out roboflow boundary box sizes.
# Also, trial and error with the train image segmentations since roboflow was perfect segmentation. 

def extract_chocolate_segments(
    image_path, 
    std_thresh=5.0, 
    iou_thresh=0.7,
    area_thresholds=[(110000, 400000), (100000, 300000), (40000, 200000)],
    final_max_area=400000
):
    def difference_of_Gaussians(img, k1, s1, k2, s2):
        b1 = cv2.GaussianBlur(img, (k1, k1), s1)
        b2 = cv2.GaussianBlur(img, (k2, k2), s2)
        return b1 - b2

    def detect_chocolates(thresholded_img, area_min=110000, area_max=350000, excluded_mask=None):
        if excluded_mask is not None:
            th = thresholded_img.copy()
            th[excluded_mask > 0] = 0
        else:
            th = thresholded_img

        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        boxes = []
        for c in contours:
            area = cv2.contourArea(c)
            if area_min < area < area_max:
                rect = cv2.minAreaRect(c)
                (_, (w, h), _) = rect
                area_box = w * h
                if 30000 < area_box < 600000:
                    box = cv2.boxPoints(rect)
                    box = np.int64(box)
                    boxes.append(box)
        return boxes
    
    def box_mostly_inside(box_inner, box_outer, threshold=0.75):
        box_outer = box_outer.astype(np.float32)
        inside_count = sum(cv2.pointPolygonTest(box_outer, (float(pt[0]), float(pt[1])), False) >= 0 for pt in box_inner)
        return inside_count / len(box_inner) >= threshold

    def bounding_rect_to_xywh(box):
        x, y, w, h = cv2.boundingRect(box)
        return x, y, w, h

    def iou(boxA, boxB):
        xA, yA, wA, hA = bounding_rect_to_xywh(boxA)
        xB, yB, wB, hB = bounding_rect_to_xywh(boxB)

        x1 = max(xA, xB)
        y1 = max(yA, yB)
        x2 = min(xA + wA, xB + wB)
        y2 = min(yA + hA, yB + hB)

        interW = max(0, x2 - x1)
        interH = max(0, y2 - y1)
        interArea = interW * interH

        boxAArea = wA * hA
        boxBArea = wB * hB

        unionArea = boxAArea + boxBArea - interArea

        if unionArea == 0:
            return 0.0
        return interArea / unionArea

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Unable to read image at path: {image_path}")
    
    img_orig = img.copy()  # Save original image for final outputs
    img = (img * 0.7).astype(np.uint8)  # Darkened version for processing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((3, 3), np.uint8)
    exclusion_mask = np.zeros_like(gray)
    all_boxes = []

    for i in range(3):
        area_min, area_max = area_thresholds[i]

        gray_masked = gray.copy()
        gray_masked[exclusion_mask > 0] = 0

        DoG_norm = cv2.normalize(
            difference_of_Gaussians(gray_masked, 9, 7, 25, 15),
            None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        _, th = cv2.threshold(DoG_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        if i == 0:
            th_processed = cv2.dilate(cv2.erode(th, kernel, iterations=2), kernel, iterations=6)
        else:
            th_processed = cv2.dilate(th, kernel, iterations=2)

        boxes = detect_chocolates(th_processed, area_min=area_min, area_max=area_max)
        all_boxes += [(f"DoG_{i+1}", box) for box in boxes]

        for box in boxes:
            cv2.fillPoly(exclusion_mask, [box], 255)

        # Plot each round's detection result
        # img_dog = img_orig.copy()
        # for box in boxes:
        #     cv2.drawContours(img_dog, [box], 0, (0, 255, 0), 2)
        # plt.figure(figsize=(10, 6))
        # plt.imshow(cv2.cvtColor(img_dog, cv2.COLOR_BGR2RGB))
        # plt.title(f'Detected Chocolates after DoG Round {i+1}: {len(boxes)} boxes')
        # plt.axis('off')
        # plt.show()

    # Filter overlapping boxes by IoU, keep bigger box
    filtered_boxes = []

    for origin, box in all_boxes:
        x, y, w, h = cv2.boundingRect(box)
        area = w * h

        overlap_found = False
        to_remove = []
        for idx, (f_origin, f_box) in enumerate(filtered_boxes):
            iou_val = iou(box, f_box)
            if iou_val > iou_thresh:
                fx, fy, fw, fh = cv2.boundingRect(f_box)
                f_area = fw * fh
                if area > f_area:
                    to_remove.append(idx)
                else:
                    overlap_found = True
                    break

        for idx in reversed(to_remove):
            filtered_boxes.pop(idx)

        if not overlap_found:
            filtered_boxes.append((origin, box))

    final_boxes = []
    for i, (origin_i, box_i) in enumerate(filtered_boxes):
        keep = True
        for j, (origin_j, box_j) in enumerate(filtered_boxes):
            if i != j and box_mostly_inside(box_i, box_j):
                keep = False
                break
        if keep:
            final_boxes.append((origin_i, box_i))

    filtered_boxes = final_boxes

    segments = []
    for origin, box in filtered_boxes:
        x, y, w, h = cv2.boundingRect(box)
        area = w * h
        crop = img_orig[y:y+h, x:x+w]  # Use original image for segments
        segments.append(crop)

    # Plot final filtered detection
    # img_plot = img_orig.copy()
    # for origin, box in filtered_boxes:
    #     color = (0, 255, 0) if origin == 'DoG_1' else (255, 0, 0) if origin == 'DoG_2' else (0, 0, 255)
    #     cv2.drawContours(img_plot, [box], 0, color, 3)

    # plt.figure(figsize=(10, 8))
    # plt.imshow(cv2.cvtColor(img_plot, cv2.COLOR_BGR2RGB))
    # plt.title(f'Final Filtered Detections: {len(segments)}')
    # plt.axis('off')
    # plt.show()

    return segments


####################################################################################################

# YOLO Bounding Box 
import os
import cv2
import pandas as pd

# --- Configuration ---
IMAGE_DIR = 'data/train'
LABEL_DIR = 'data/train_labels'
SEGMENTS_DIR = 'src/yolo_bounding_boxes/chocolate_segments'
DEBUG_BBOX_DIR = 'src/yolo_bounding_boxes/chocolate_debug_bboxes_padded_debugrun'
JPEG_QUALITY = 95
VISUALIZE_BOUNDING_BOXES = True
PADDING_PIXELS = 20

# Only run if segmentation output doesn't already exist
if not (os.path.exists(SEGMENTS_DIR) and os.path.exists(DEBUG_BBOX_DIR)):
    # --- Initialization ---
    os.makedirs(SEGMENTS_DIR, exist_ok=True)
    os.makedirs(DEBUG_BBOX_DIR, exist_ok=True)

    labels_list = []
    segment_counter = 0

    print("--- SEGMENTATION SCRIPT (WITH PADDING & VISUALIZATION) ---")
    print(f"Reading images from: {IMAGE_DIR}")
    print(f"Reading labels from: {LABEL_DIR}")
    print(f"Saving segments to: {SEGMENTS_DIR} (JPEG Quality: {JPEG_QUALITY})")
    print(f"Adding {PADDING_PIXELS}px padding to each side of the bounding box.")
    if VISUALIZE_BOUNDING_BOXES:
        print(f"Saving debug bounding box images to: {DEBUG_BBOX_DIR}")
    print("-" * 50)

    # --- Processing Loop ---
    for img_name in os.listdir(IMAGE_DIR):
        if not img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue

        base_name = os.path.splitext(img_name)[0]
        label_path = os.path.join(LABEL_DIR, base_name + '.txt')
        img_path = os.path.join(IMAGE_DIR, img_name)

        img = cv2.imread(img_path)
        if img is None:
            print(f"\n[Image: {img_name}] Error: Could not read image at {img_path}. Skipping.")
            continue

        h_img, w_img = img.shape[:2]
        print(f"\n[Image: {img_name}] Loaded. Dimensions: {w_img}x{h_img}")

        if VISUALIZE_BOUNDING_BOXES:
            img_with_boxes = img.copy()

        if not os.path.exists(label_path):
            print(f"  [Image: {img_name}] Label file not found: {label_path}. Skipping image.")
            continue

        with open(label_path, 'r') as f:
            lines = f.readlines()

        if not lines:
            print(f"  [Image: {img_name}] Label file {label_path} is empty. Skipping image.")
            continue

        print(f"  [Image: {img_name}] Found {len(lines)} lines in label file: {label_path}")
        found_valid_segment_in_image = False

        for i, line in enumerate(lines):
            print(f"    [L{i+1}] Processing line: '{line.strip()}'")
            parts = line.strip().split()

            if len(parts) != 5:
                print(f"      [L{i+1}] Warning: Malformed line. Expected 5 parts, got {len(parts)}. Skipping line.")
                continue
            try:
                class_id_str, x_center_str, y_center_str, box_w_str, box_h_str = parts
                class_id = int(class_id_str)
                x_center_norm = float(x_center_str)
                y_center_norm = float(y_center_str)
                box_w_norm = float(box_w_str)
                box_h_norm = float(box_h_str)
            except ValueError as e:
                print(f"      [L{i+1}] Warning: Error parsing numeric values from line. Error: {e}. Skipping line.")
                continue

            x_center_abs = x_center_norm * w_img
            y_center_abs = y_center_norm * h_img
            box_w_abs = box_w_norm * w_img
            box_h_abs = box_h_norm * h_img

            x1_tight = int(x_center_abs - box_w_abs / 2)
            y1_tight = int(y_center_abs - box_h_abs / 2)
            x2_tight = int(x_center_abs + box_w_abs / 2)
            y2_tight = int(y_center_abs + box_h_abs / 2)
            print(f"      [L{i+1}] Tight BBox (x1,y1,x2,y2 before padding/clip): {x1_tight}, {y1_tight}, {x2_tight}, {y2_tight}")

            x1_padded = x1_tight - PADDING_PIXELS
            y1_padded = y1_tight - PADDING_PIXELS
            x2_padded = x2_tight + PADDING_PIXELS
            y2_padded = y2_tight + PADDING_PIXELS
            print(f"      [L{i+1}] Padded BBox (x1p,y1p,x2p,y2p before clip): {x1_padded}, {y1_padded}, {x2_padded}, {y2_padded}")

            x1_clip = max(0, x1_padded)
            y1_clip = max(0, y1_padded)
            x2_clip = min(w_img, x2_padded)
            y2_clip = min(h_img, y2_padded)
            print(f"      [L{i+1}] Final Clipped BBox (x1c,y1c,x2c,y2c): {x1_clip}, {y1_clip}, {x2_clip}, {y2_clip}")

            current_crop_w = x2_clip - x1_clip
            current_crop_h = y2_clip - y1_clip
            if current_crop_w <= 0 or current_crop_h <= 0:
                print(f"      [L{i+1}] Warning: Invalid BBox after clipping. W={current_crop_w}, H={current_crop_h}. Skipping.")
                continue

            crop = img[y1_clip:y2_clip, x1_clip:x2_clip]
            if crop.size == 0:
                print(f"      [L{i+1}] Warning: Resulting crop is empty. Skipping.")
                continue

            actual_cropped_w = crop.shape[1]
            actual_cropped_h = crop.shape[0]
            print(f"      [L{i+1}] Actual Cropped Segment Dimensions: W={actual_cropped_w}, H={actual_cropped_h}")
            if actual_cropped_w < 50 or actual_cropped_h < 50:
                print(f"      [L{i+1}] ----> Note: This crop is relatively small (<50px in one dimension).")

            if VISUALIZE_BOUNDING_BOXES:
                cv2.rectangle(img_with_boxes, (x1_clip, y1_clip), (x2_clip, y2_clip), (0, 255, 0), 2)
                label_text = f"c{class_id}_s{segment_counter}"
                cv2.putText(img_with_boxes, label_text, (x1_clip, y1_clip - 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            segment_fname = f"{base_name}_segment_{segment_counter}.jpg"
            segment_save_path = os.path.join(SEGMENTS_DIR, segment_fname)

            try:
                cv2.imwrite(segment_save_path, crop, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                labels_list.append({'filename': segment_fname, 'label': class_id})
                segment_counter += 1
                found_valid_segment_in_image = True
            except Exception as e:
                print(f"      [L{i+1}] Error saving segment {segment_save_path}: {e}")

            print("-" * 20)

        if VISUALIZE_BOUNDING_BOXES and found_valid_segment_in_image:
            debug_img_path = os.path.join(DEBUG_BBOX_DIR, f"{base_name}_bboxes_padded_debug.jpg")
            try:
                cv2.imwrite(debug_img_path, img_with_boxes)
                print(f"  [Image: {img_name}] Saved debug image to: {debug_img_path}")
            except Exception as e:
                print(f"  [Image: {img_name}] Error saving debug image: {e}")
        elif VISUALIZE_BOUNDING_BOXES and not found_valid_segment_in_image:
            print(f"  [Image: {img_name}] No valid segments found. No debug image saved.")
        print("-" * 50)

    df_labels = pd.DataFrame(labels_list)
    if not df_labels.empty:
        csv_save_path = 'yolo_bounding_boxes/segment_labels.csv'
        df_labels.to_csv(csv_save_path, index=False)
        print(f"\n--- SCRIPT FINISHED ---")
        print(f"Saved {len(df_labels)} segments to '{SEGMENTS_DIR}'")
        print(f"Labels saved to '{csv_save_path}'.")
        if VISUALIZE_BOUNDING_BOXES:
            print(f"Debug images saved in '{DEBUG_BBOX_DIR}'.")
    else:
        print("\n--- SCRIPT FINISHED ---")
        print("No segments were processed or saved.")
else:
    print(f"Segmentation already exists in '{SEGMENTS_DIR}' and debug images in '{DEBUG_BBOX_DIR}'. Skipping script.")

####################################################################################################

# Classifier Helpers

# Parameters
RESIZE_DIM = (256, 256)
LBP_RADIUS = 1
LBP_N_POINTS = 8 * LBP_RADIUS
DEFAULT_BORDER_TYPE = cv2.BORDER_REFLECT_101 # Or cv2.BORDER_CONSTANT

# --- SCRIPT EXECUTION SETTINGS ---
ENABLE_AUGMENTATION = True 
RUN_GRID_SEARCH = True    # *** SET TO TRUE TO RUN HYPERPARAMETER TUNING ***
                           # Set to False to use default RF params for faster runs

# --- Utility Functions ---
def resize_and_pad(img, desired_size, border_type=DEFAULT_BORDER_TYPE, pad_color=(0, 0, 0)):
    old_h, old_w = img.shape[:2]
    desired_h, desired_w = desired_size
    ratio_w = float(desired_w) / old_w
    ratio_h = float(desired_h) / old_h
    ratio = min(ratio_w, ratio_h)
    new_w = int(round(old_w * ratio))
    new_h = int(round(old_h * ratio))
    new_w = max(1, new_w)
    new_h = max(1, new_h)
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    delta_w = desired_w - new_w
    delta_h = desired_h - new_h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    if border_type == cv2.BORDER_CONSTANT:
        padded_img = cv2.copyMakeBorder(img_resized, top, bottom, left, right,
                                        border_type, value=pad_color)
    else:
        padded_img = cv2.copyMakeBorder(img_resized, top, bottom, left, right,
                                        border_type)
    return padded_img

def augment_image(image):
    augmented_image = image.copy()
    if random.random() > 0.5: # Horizontal Flip
        augmented_image = cv2.flip(augmented_image, 1)

    # Brightness Adjustment
    hsv = cv2.cvtColor(augmented_image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    value_change = random.randint(-30, 30) # More conservative range
    v_new = cv2.add(v, value_change)
    v_new = np.clip(v_new, 0, 255)
    final_hsv = cv2.merge((h, s, v_new))
    augmented_image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

    # Slight Rotation (e.g., -10 to +10 degrees)
    if random.random() > 0.3: # Apply rotation 70% of the time for augmented images
        angle = random.uniform(-10, 10)
        (h_orig, w_orig) = augmented_image.shape[:2]
        center = (w_orig // 2, h_orig // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        # Use original border type for rotation padding
        border_val_for_rotation = (0,0,0) if DEFAULT_BORDER_TYPE == cv2.BORDER_CONSTANT else None
        if border_val_for_rotation:
             augmented_image = cv2.warpAffine(augmented_image, M, (w_orig, h_orig), borderMode=DEFAULT_BORDER_TYPE, borderValue=border_val_for_rotation)
        else:
             augmented_image = cv2.warpAffine(augmented_image, M, (w_orig, h_orig), borderMode=DEFAULT_BORDER_TYPE)


    # Add more augmentations here if needed:
    # - Contrast
    # - Small zooms/shifts (cv2.warpAffine with translation matrix)
    # - Gaussian blur (cv2.GaussianBlur)

    return augmented_image

# --- Feature Extraction Functions ---

# old one
def remove_background(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mask = cv2.adaptiveThreshold(gray, 255,
                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 51, 10)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask

def extract_fourier_features(img_masked_gray):
    if img_masked_gray.shape[0] < 2 or img_masked_gray.shape[1] < 2:
        return np.zeros(32*32)
    f = np.fft.fft2(img_masked_gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)
    magnitude_spectrum_norm = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX)
    magnitude_resized = cv2.resize(magnitude_spectrum_norm, (32, 32), interpolation=cv2.INTER_AREA)
    return magnitude_resized.flatten()

def extract_hog(img_bgr_masked, pixels_per_cell=(16,16), cells_per_block=(2,2)):
    gray = cv2.cvtColor(img_bgr_masked, cv2.COLOR_BGR2GRAY)
    min_height = cells_per_block[0] * pixels_per_cell[0]
    min_width = cells_per_block[1] * pixels_per_cell[1]
    expected_hog_len = 8100 # Based on (256,256) RESIZE_DIM, (16,16) ppc, (2,2) cpb, 9 orientations

    if gray.shape[0] < min_height or gray.shape[1] < min_width:
        # print(f"Warning: Image shape {gray.shape} too small for HOG. Returning zeros.")
        return np.zeros(expected_hog_len)

    # Get only the feature vector
    features = hog(gray,
                   orientations=9,
                   pixels_per_cell=pixels_per_cell,
                   cells_per_block=cells_per_block,
                   block_norm='L2-Hys',
                   visualize=False,       # Ensure visualize is False to get only features
                   feature_vector=True)   # Ensure feature_vector is True

    # Ensure consistent length (handle potential edge cases from skimage.hog)
    if features.shape[0] != expected_hog_len:
        # This case should be rare with feature_vector=True and sufficient image size
        # print(f"Warning: HOG feature length mismatch. Expected {expected_hog_len}, got {features.shape[0]}. Adjusting.")
        if features.shape[0] < expected_hog_len:
            features = np.pad(features, (0, expected_hog_len - features.shape[0]), 'constant')
        else:
            features = features[:expected_hog_len]
            
    return features

def extract_lbp(img_bgr_masked):
    gray = cv2.cvtColor(img_bgr_masked, cv2.COLOR_BGR2GRAY)
    if gray.shape[0] < (2 * LBP_RADIUS + 1) or gray.shape[1] < (2 * LBP_RADIUS + 1):
        return np.zeros(LBP_N_POINTS + 2)
    lbp = local_binary_pattern(gray, LBP_N_POINTS, LBP_RADIUS, method='uniform')
    fixed_n_bins = LBP_N_POINTS + 2
    hist, _ = np.histogram(lbp.ravel(),
                           bins=fixed_n_bins,
                           range=(0, fixed_n_bins),
                           density=True)
    return hist

def extract_color_features(img_bgr_masked):
    if img_bgr_masked.shape[0] == 0 or img_bgr_masked.shape[1] == 0:
        return np.zeros(16 + 8 + 8)
    hsv_img = cv2.cvtColor(img_bgr_masked, cv2.COLOR_BGR2HSV)
    gray_for_mask = cv2.cvtColor(img_bgr_masked, cv2.COLOR_BGR2GRAY)
    _, active_pixel_mask = cv2.threshold(gray_for_mask, 1, 255, cv2.THRESH_BINARY)
    if cv2.countNonZero(active_pixel_mask) == 0:
        return np.zeros(16 + 8 + 8)
    hist_h = cv2.calcHist([hsv_img], [0], active_pixel_mask, [16], [0, 180])
    hist_s = cv2.calcHist([hsv_img], [1], active_pixel_mask, [8], [0, 256])
    hist_v = cv2.calcHist([hsv_img], [2], active_pixel_mask, [8], [0, 256])
    cv2.normalize(hist_h, hist_h)
    cv2.normalize(hist_s, hist_s)
    cv2.normalize(hist_v, hist_v)
    return np.concatenate([hist_h.flatten(), hist_s.flatten(), hist_v.flatten()])

def extract_all_features(img_bgr_segment, hog_pixels_per_cell=(16,16), hog_cells_per_block=(2,2)):
    if img_bgr_segment is None or img_bgr_segment.shape[0] == 0 or img_bgr_segment.shape[1] == 0:
        # This case should be filtered out by load_segments_and_labels
        raise ValueError("Invalid segment passed to extract_all_features")

    img_resized_padded = resize_and_pad(img_bgr_segment, RESIZE_DIM)
    mask = remove_background(img_resized_padded)
    img_masked = cv2.bitwise_and(img_resized_padded, img_resized_padded, mask=mask)

    hog_feat = extract_hog(img_masked, pixels_per_cell=hog_pixels_per_cell, cells_per_block=hog_cells_per_block)
    lbp_feat = extract_lbp(img_masked)
    if len(img_masked.shape) == 3:
        gray_for_fourier = cv2.cvtColor(img_masked, cv2.COLOR_BGR2GRAY)
    else:
        gray_for_fourier = img_masked
    fourier_feat = extract_fourier_features(gray_for_fourier)
    color_feat = extract_color_features(img_masked)
    features = np.concatenate([hog_feat, lbp_feat, fourier_feat, color_feat])
    return features

####################################################################################################

# Train Classifier it doesnt exist 

import os

model_path = 'classification_model/chocolate_classifier_model_tuned.joblib'
scaler_path = 'classification_model/chocolate_feature_scaler_tuned.joblib'
training_script = 'py_scrips/classification_training.py'

if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
    print("One or both model components not found. Running training...")
    os.system(f'python {training_script}')
else:
    print("Both model and scaler found. Skipping training.")



####################################################################################################

# Reject unwanted colour/empty/small segments 

def reject_unwanted_colors(segment, color_thresh=0.5, show_plot=False, index=None):
    # Convert to HSV and compute masks
    hsv = cv2.cvtColor(segment, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    total_pixels = h.size

    # Define color masks
    # trial and error on train images 
    masks = {
        "yellow": ((h > 20) & (h < 40)) & (s > 100) & (v > 100),
        "black": (v < 50) ,
        "green": ((h > 35) & (h < 85)) & (s > 50),
        "blue": ((h > 85) & (h < 135)) & (s > 50),
        "brown": (((h >= 0) & (h < 30)) | ((h > 160) & (h < 175))) & (s > 45) & (v > 50) & (v < 200)
    }

    brown_ratio = np.sum(masks["brown"]) / total_pixels

    # 1. Reject if too small AND brown is insufficient
    # smallest we got are jellies and so just in case we check for brown so we dont get rid of them by acc
    # also we dont check for white because our model segments large chunks for white because of noise and etc.
    if (segment.shape[0] < 300 or segment.shape[1] < 300) and brown_ratio < 0.2:
        if show_plot:
            plt.figure(figsize=(2.5, 2.5))
            plt.imshow(cv2.cvtColor(segment, cv2.COLOR_BGR2RGB))
            plt.title(f"Rejected #{index}: too small, brown={brown_ratio:.1%}")
            plt.axis('off')
            plt.tight_layout()
            plt.show()
        return True, "too_small_low_brown"

    # 2. Reject blue segments only if brown is insufficient
    # because of peacock and bluebox bgs, we had to be careful not to reject chocolates on blue bgs
    # we mostly wanted to get rid of empty segments of bg and those fake objects
    for name in ["blue"]:
        mask = masks[name]
        proportion = np.sum(mask) / total_pixels
        if proportion > color_thresh and brown_ratio < 0.2:
            if show_plot:
                plt.figure(figsize=(2.5, 2.5))
                plt.imshow(cv2.cvtColor(segment, cv2.COLOR_BGR2RGB))
                plt.title(f"Rejected #{index}: {name}, brown={brown_ratio:.1%}")
                plt.axis('off')
                plt.tight_layout()
                plt.show()
            return True, name + "_low_brown"
    
    # 3. Reject black segments only if brown is insufficient
    # we wanna get rid of those fake chocolates that are just black boxes
    # but the dark chocos like jelly black and noir authentique would also get selected without the brown proportion check 
    # since even dark choco has more brown/warm undertones than just black 
    for name in ["black"]:
        mask = masks[name]
        proportion = np.sum(mask) / total_pixels
        if (proportion > 0.15 and brown_ratio < 0.02) or (proportion > 0.20 and brown_ratio < 0.1) :
            if show_plot:
                plt.figure(figsize=(2.5, 2.5))
                plt.imshow(cv2.cvtColor(segment, cv2.COLOR_BGR2RGB))
                plt.title(f"Rejected #{index}: {name}, prop={proportion:.1%},  brown={brown_ratio:.1%}")
                plt.axis('off')
                plt.tight_layout()
                plt.show()
            return True, name + "_low_brown"
    
    # 4. Reject yellow and green 
    # luckily we dont have any yellow or green backgrounds so we could go easier on the threshold 
    for name in ["yellow", "green"]:
        mask = masks[name]
        proportion = np.sum(mask) / total_pixels
        if proportion > 0.30:
            if show_plot:
                plt.figure(figsize=(2.5, 2.5))
                plt.imshow(cv2.cvtColor(segment, cv2.COLOR_BGR2RGB))
                plt.title(f"Rejected #{index}: {name}, yellow={proportion:.1%}")
                plt.axis('off')
                plt.tight_layout()
                plt.show()
            return True, name

    # 3. Reject plain/uniform segments based on low color variance
    std_dev = np.std(segment)
    if std_dev < 15: 
        # trial and error got us 15 so idk what stat you want :(
        if show_plot:
            plt.figure(figsize=(2.5, 2.5))
            plt.imshow(cv2.cvtColor(segment, cv2.COLOR_BGR2RGB))
            plt.title(f"Rejected #{index}: plain background (std={std_dev:.2f})")
            plt.axis('off')
            plt.tight_layout()
            plt.show()
        return True, "plain_background"

    return False, None

####################################################################################################

# Final run + CSV Creation 

# Load model and scaler
clf_loaded = joblib.load("classification_model/chocolate_classifier_model_tuned.joblib")
scaler_loaded = joblib.load("classification_model/chocolate_feature_scaler_tuned.joblib")

# Label mapping
label_mapping = {
    0: 'Amandina', 1: 'Arabia', 2: 'Comtesse', 3: 'Creme_brulee', 4: 'Jelly_black',
    5: 'Jelly_milk', 6: 'Jelly_white', 7: 'Noblesse', 8: 'Noir_authentique',
    9: 'Passion_au_lait', 10: 'Stracciatella', 11: 'Tentation_noir', 12: 'Triangolo'
}
ordered_labels = ['Jelly_white', 'Jelly_milk', 'Jelly_black', 'Amandina', 'Creme_brulee', 
                  'Triangolo', 'Tentation_noir', 'Comtesse', 'Noblesse', 
                  'Noir_authentique', 'Passion_au_lait', 'Arabia', 'Stracciatella']

# Final prediction storage
results = []
all_valid_segments = []
all_predicted_human_labels = []

# Process each test image
for fname in os.listdir("data/test"):
    if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    image_path = os.path.join("data/test", fname)
    segments = extract_chocolate_segments(image_path)
    
    predicted_labels = []
    valid_segments = []

    for i, seg in enumerate(segments):
        if seg is None or seg.shape[0] == 0 or seg.shape[1] == 0:
            continue

        reject, reason = reject_unwanted_colors(seg, color_thresh=0.75, show_plot=False, index=i)
        if reject:
            print(f"  Segment {i} in {fname} rejected due to {reason}.")
            continue

        try:
            features = extract_all_features(seg)
            scaled_features = scaler_loaded.transform([features])
            pred = clf_loaded.predict(scaled_features)[0]
            predicted_labels.append(pred)
            valid_segments.append(seg)
        except Exception as e:
            print(f"  Feature extraction or prediction failed for segment {i} in {fname}: {e}")
            continue

    # Count chocolate types
    label_counts = Counter(predicted_labels)
    result_row = {
        "id": os.path.splitext(fname)[0].replace("L", "")  # 'L1000781.JPG' → '1000781'
    }
    for label_name in ordered_labels:
        idx = list(label_mapping.keys())[list(label_mapping.values()).index(label_name)]
        result_row[label_name] = label_counts.get(idx, 0)
    results.append(result_row)

    # Save for visualization
    all_valid_segments.extend(valid_segments)
    all_predicted_human_labels.extend(
        [label_mapping.get(lbl, f"Unknown_{lbl}") for lbl in predicted_labels]
    )

# Save to CSV
df_results = pd.DataFrame(results)
df_results.to_csv("chocolate_test_predictions.csv", index=False)

# change header
import csv
file_path = 'chocolate_test_predictions.csv'
new_header = [
    "id", "Jelly White", "Jelly Milk", "Jelly Black", "Amandina", "Crème brulée",
    "Triangolo", "Tentation noir", "Comtesse", "Noblesse", "Noir authentique",
    "Passion au lait", "Arabia", "Stracciatella"
]
with open(file_path, 'r', newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    rows = list(reader)
rows[0] = new_header
with open(file_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerows(rows)

print("Predictions saved to chocolate_test_predictions.csv")
####################################################################################################
  