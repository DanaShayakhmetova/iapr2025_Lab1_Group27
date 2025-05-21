# THE ONE 

import cv2
import numpy as np
import os
import pandas as pd
import random
from skimage.feature import hog, local_binary_pattern
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import matplotlib.pyplot as plt

# Parameters
RESIZE_DIM = (256, 256)
LBP_RADIUS = 1
LBP_N_POINTS = 8 * LBP_RADIUS
DEFAULT_BORDER_TYPE = cv2.BORDER_REFLECT_101

# --- SCRIPT EXECUTION SETTINGS
ENABLE_AUGMENTATION = True
RUN_GRID_SEARCH = True     
ENABLE_BACKGROUND_REMOVAL = True 

# --- BACKGROUND REMOVAL PARAMETERS
BG_REMOVE_ADAPTIVE_BLOCK_SIZE = 71
BG_REMOVE_ADAPTIVE_C = 5          
BG_REMOVE_MORPH_KERNEL_SIZE = (9,9)

# --- HOG PARAMETERS 
HOG_PIXELS_PER_CELL = (24, 24) 
HOG_CELLS_PER_BLOCK = (2, 2)   
HOG_ORIENTATIONS = 9          

# --- PCA PARAMETER
N_PCA_COMPONENTS = 0.95 

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
    if random.random() > 0.5:
        augmented_image = cv2.flip(augmented_image, 1)
    hsv = cv2.cvtColor(augmented_image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    value_change = random.randint(-30, 30)
    v_new = cv2.add(v, value_change)
    v_new = np.clip(v_new, 0, 255)
    final_hsv = cv2.merge((h, s, v_new))
    augmented_image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    if random.random() > 0.3:
        angle = random.uniform(-15, 15)
        (h_orig, w_orig) = augmented_image.shape[:2]
        center = (w_orig // 2, h_orig // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        border_val_for_rotation = (0,0,0) if DEFAULT_BORDER_TYPE == cv2.BORDER_CONSTANT else None
        if border_val_for_rotation:
             augmented_image = cv2.warpAffine(augmented_image, M, (w_orig, h_orig), borderMode=DEFAULT_BORDER_TYPE, borderValue=border_val_for_rotation)
        else:
             augmented_image = cv2.warpAffine(augmented_image, M, (w_orig, h_orig), borderMode=DEFAULT_BORDER_TYPE)
    return augmented_image

# --- Feature Extraction Functions ---
def remove_background(img_bgr,
                      adaptive_block_size=BG_REMOVE_ADAPTIVE_BLOCK_SIZE,
                      adaptive_c=BG_REMOVE_ADAPTIVE_C,
                      morph_kernel_size=BG_REMOVE_MORPH_KERNEL_SIZE):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mask = cv2.adaptiveThreshold(gray, 255,
                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV,
                                 adaptive_block_size,
                                 adaptive_c)
    kernel = np.ones(morph_kernel_size, np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask

def extract_fourier_features(img_masked_gray):
    if img_masked_gray.shape[0] < 2 or img_masked_gray.shape[1] < 2:
        return np.zeros(32*32)
    if len(img_masked_gray.shape) == 3:
        img_masked_gray = cv2.cvtColor(img_masked_gray, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(img_masked_gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)
    magnitude_spectrum_norm = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX)
    magnitude_resized = cv2.resize(magnitude_spectrum_norm, (32, 32), interpolation=cv2.INTER_AREA)
    return magnitude_resized.flatten()

def extract_hog(img_bgr_masked, pixels_per_cell, cells_per_block, orientations):
    gray = cv2.cvtColor(img_bgr_masked, cv2.COLOR_BGR2GRAY)
    n_cells_y = gray.shape[0] // pixels_per_cell[0]
    n_cells_x = gray.shape[1] // pixels_per_cell[1]
    n_blocks_y = n_cells_y - cells_per_block[0] + 1
    n_blocks_x = n_cells_x - cells_per_block[1] + 1

    fallback_hog_len = 0
    try:
        default_n_cells_y = RESIZE_DIM[0] // HOG_PIXELS_PER_CELL[0]
        default_n_cells_x = RESIZE_DIM[1] // HOG_PIXELS_PER_CELL[1]
        default_n_blocks_y = default_n_cells_y - HOG_CELLS_PER_BLOCK[0] + 1
        default_n_blocks_x = default_n_cells_x - HOG_CELLS_PER_BLOCK[1] + 1
        fallback_hog_len = int(default_n_blocks_y * default_n_blocks_x * HOG_CELLS_PER_BLOCK[0] * HOG_CELLS_PER_BLOCK[1] * HOG_ORIENTATIONS)
    except ZeroDivisionError:
        print("Error: HOG_PIXELS_PER_CELL contains zero. Cannot calculate fallback HOG length.")

    if n_blocks_y <= 0 or n_blocks_x <= 0 or n_cells_y < cells_per_block[0] or n_cells_x < cells_per_block[1]:
        print(f"Warning: Image shape {gray.shape} too small for HOG with PPC={pixels_per_cell}, CPB={cells_per_block}. Returning zeros of length {fallback_hog_len}.")
        return np.zeros(fallback_hog_len if fallback_hog_len > 0 else 1000)

    expected_hog_len = int(n_blocks_y * n_blocks_x * cells_per_block[0] * cells_per_block[1] * orientations)
    min_height_required = cells_per_block[0] * pixels_per_cell[0]
    min_width_required = cells_per_block[1] * pixels_per_cell[1]

    if gray.shape[0] < min_height_required or gray.shape[1] < min_width_required:
        print(f"Warning: Image shape {gray.shape} too small for HOG. Min required: ({min_height_required},{min_width_required}). Returning zeros of length {expected_hog_len}.")
        return np.zeros(expected_hog_len)

    features = hog(gray,
                   orientations=orientations,
                   pixels_per_cell=pixels_per_cell,
                   cells_per_block=cells_per_block,
                   block_norm='L2-Hys',
                   visualize=False,
                   feature_vector=True)
    if features.shape[0] != expected_hog_len:
        print(f"Warning: HOG feature length mismatch. Expected {expected_hog_len}, got {features.shape[0]}. Adjusting.")
        if features.shape[0] < expected_hog_len: features = np.pad(features, (0, expected_hog_len - features.shape[0]), 'constant')
        else: features = features[:expected_hog_len]
    return features

def extract_lbp(img_bgr_masked):
    gray = cv2.cvtColor(img_bgr_masked, cv2.COLOR_BGR2GRAY)
    if gray.shape[0] < (2 * LBP_RADIUS + 1) or gray.shape[1] < (2 * LBP_RADIUS + 1): return np.zeros(LBP_N_POINTS + 2)
    lbp = local_binary_pattern(gray, LBP_N_POINTS, LBP_RADIUS, method='uniform')
    fixed_n_bins = LBP_N_POINTS + 2
    hist, _ = np.histogram(lbp.ravel(), bins=fixed_n_bins, range=(0, fixed_n_bins), density=True)
    return hist

def extract_color_features(img_bgr_masked):
    if img_bgr_masked.shape[0] == 0 or img_bgr_masked.shape[1] == 0: return np.zeros(16 + 8 + 8)
    hsv_img = cv2.cvtColor(img_bgr_masked, cv2.COLOR_BGR2HSV)
    gray_for_mask_check = cv2.cvtColor(img_bgr_masked, cv2.COLOR_BGR2GRAY)
    _, active_pixel_mask_for_hist = cv2.threshold(gray_for_mask_check, 1, 255, cv2.THRESH_BINARY)
    if cv2.countNonZero(active_pixel_mask_for_hist) == 0: return np.zeros(16 + 8 + 8)
    hist_h = cv2.calcHist([hsv_img], [0], active_pixel_mask_for_hist, [16], [0, 180])
    hist_s = cv2.calcHist([hsv_img], [1], active_pixel_mask_for_hist, [8], [0, 256])
    hist_v = cv2.calcHist([hsv_img], [2], active_pixel_mask_for_hist, [8], [0, 256])
    cv2.normalize(hist_h, hist_h); cv2.normalize(hist_s, hist_s); cv2.normalize(hist_v, hist_v)
    return np.concatenate([hist_h.flatten(), hist_s.flatten(), hist_v.flatten()])

def extract_all_features(img_bgr_segment, hog_pixels_per_cell, hog_cells_per_block, hog_orientations):
    if img_bgr_segment is None or img_bgr_segment.shape[0] == 0 or img_bgr_segment.shape[1] == 0:
        raise ValueError("Invalid segment passed to extract_all_features")
    img_resized_padded = resize_and_pad(img_bgr_segment, RESIZE_DIM)
    if ENABLE_BACKGROUND_REMOVAL:
        mask = remove_background(img_resized_padded)
        img_to_process = cv2.bitwise_and(img_resized_padded, img_resized_padded, mask=mask)
    else:
        img_to_process = img_resized_padded
    hog_feat = extract_hog(img_to_process, pixels_per_cell=hog_pixels_per_cell, cells_per_block=hog_cells_per_block, orientations=hog_orientations)
    lbp_feat = extract_lbp(img_to_process)
    if len(img_to_process.shape) == 3: gray_for_fourier = cv2.cvtColor(img_to_process, cv2.COLOR_BGR2GRAY)
    else:
        gray_for_fourier = img_to_process
        if gray_for_fourier.ndim == 2 and gray_for_fourier.shape[:2] != RESIZE_DIM: gray_for_fourier = cv2.resize(gray_for_fourier, RESIZE_DIM, interpolation=cv2.INTER_AREA)
        elif gray_for_fourier.ndim != 2:
             print(f"Warning: Unexpected image format for Fourier input. Shape: {gray_for_fourier.shape}")
             expected_other_len = len(hog_feat) + len(lbp_feat) + (16+8+8)
             # To ensure a fixed length vector is returned for concatenation in case of error
             fourier_feat = np.zeros(32*32)
             color_feat = np.zeros(16+8+8)
             return np.concatenate([hog_feat, lbp_feat, fourier_feat, color_feat])
    fourier_feat = extract_fourier_features(gray_for_fourier)
    color_feat = extract_color_features(img_to_process)
    if fourier_feat.shape[0] != 32*32: fourier_feat = np.zeros(32*32)
    features = np.concatenate([hog_feat, lbp_feat, fourier_feat, color_feat])
    return features

# --- Data Loading ---
def load_segments_and_labels(segment_folder, labels_csv, is_training=False,
                             hog_ppc=HOG_PIXELS_PER_CELL, hog_cpb=HOG_CELLS_PER_BLOCK, hog_orient=HOG_ORIENTATIONS):
    df = pd.read_csv(labels_csv)
    X_features = []
    y_labels = []
    processed_files_info = []
    print(f"Loading data. Augmentation {'ENABLED' if is_training and ENABLE_AUGMENTATION else 'disabled'}.")
    print(f"Padding method for resize_and_pad: {'BORDER_CONSTANT (Black)' if DEFAULT_BORDER_TYPE == cv2.BORDER_CONSTANT else 'BORDER_REFLECT_101 (or similar)'}")
    bg_removal_status = 'ENABLED' if ENABLE_BACKGROUND_REMOVAL else 'DISABLED'
    if ENABLE_BACKGROUND_REMOVAL: bg_removal_status += f" (BlockSize: {BG_REMOVE_ADAPTIVE_BLOCK_SIZE}, C: {BG_REMOVE_ADAPTIVE_C}, Kernel: {BG_REMOVE_MORPH_KERNEL_SIZE})"
    print(f"Background removal: {bg_removal_status}")
    print(f"HOG Parameters: PPC={hog_ppc}, CPB={hog_cpb}, Orientations={hog_orient}")

    for idx, row in df.iterrows():
        filename = row['filename']; label = row['label']
        img_path = os.path.join(segment_folder, filename)
        img_segment = cv2.imread(img_path)
        if img_segment is None: print(f"Warning: Could not read {img_path} (row {idx}), skipping."); continue
        if img_segment.shape[0] == 0 or img_segment.shape[1] == 0: print(f"Warning: Segment {filename} (row {idx}) has zero dimensions, skipping."); continue
        segments_to_process = [img_segment]; labels_for_sample = [label]; file_info_tags = [filename + "_orig"]
        if is_training and ENABLE_AUGMENTATION:
            for aug_idx in range(2):
                img_augmented = augment_image(img_segment)
                segments_to_process.append(img_augmented); labels_for_sample.append(label)
                file_info_tags.append(f"{filename}_aug{aug_idx+1}")
        for i, seg_to_proc in enumerate(segments_to_process):
            try:
                feats = extract_all_features(seg_to_proc, hog_pixels_per_cell=hog_ppc, hog_cells_per_block=hog_cpb, hog_orientations=hog_orient)
                X_features.append(feats); y_labels.append(labels_for_sample[i]); processed_files_info.append(f"{file_info_tags[i]} (row {idx})")
            except Exception as e: print(f"Error extracting features for {file_info_tags[i]} (row {idx}): {e}. Skipping this segment."); continue
    if not X_features: print("Critical Warning: No features were extracted."); return np.array([]), np.array([])
    if X_features:
        first_len = len(X_features[0])
        for i, f_vec in enumerate(X_features):
            if len(f_vec) != first_len: raise ValueError(f"FATAL: Inconsistent feature length for {processed_files_info[i]}! Expected {first_len}, got {len(f_vec)}.")
    return np.array(X_features), np.array(y_labels)

# --- Main Training Script ---
if __name__ == '__main__':
    train_segment_folder = "../src/yolo_bounding_boxes/chocolate_segments" 
    train_labels_csv = "../src/yolo_bounding_boxes/segment_labels.csv" 
    output_model_dir = "../classification_model"                       
    os.makedirs(output_model_dir, exist_ok=True)

    print("--- Starting Training Process ---")
    print(f"PCA Components Target: {N_PCA_COMPONENTS}")

    print("\n[PHASE 1: Loading Training Data]")
    X_train_raw, y_train = load_segments_and_labels(train_segment_folder, train_labels_csv, is_training=True,
                                                    hog_ppc=HOG_PIXELS_PER_CELL, hog_cpb=HOG_CELLS_PER_BLOCK, hog_orient=HOG_ORIENTATIONS)
    if X_train_raw.size == 0: print("No training data loaded. Exiting."); exit()
    print(f"Successfully processed {len(X_train_raw)} training samples (raw features).")
    print(f"Raw feature vector length: {X_train_raw.shape[1]}")
    print(f"Class distribution in raw training data: \n{pd.Series(y_train).value_counts(dropna=False)}")

    print("\n[PHASE 2: Train/Validation Split (Pre-Scaling/PCA)]")
    X_train_to_process, X_val_raw, y_train_final, y_val = train_test_split(
        X_train_raw, y_train, test_size=0.25, random_state=42, stratify=y_train
    )
    print(f"Raw Training samples for scaling/PCA: {len(X_train_to_process)}, Raw Validation samples: {len(X_val_raw)}")

    print("\n[PHASE 3: Feature Scaling & PCA]")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_to_process)
    X_val_scaled = scaler.transform(X_val_raw)
    print("Features scaled using StandardScaler (fitted on training split only).")

    pca = PCA(n_components=N_PCA_COMPONENTS, random_state=42)
    X_train_final_pca = pca.fit_transform(X_train_scaled)
    X_val_final_pca = pca.transform(X_val_scaled)
    print(f"PCA applied. Transformed features from {X_train_scaled.shape[1]} to {X_train_final_pca.shape[1]} components.")
    if hasattr(pca, 'explained_variance_ratio_'): 
      print(f"Explained variance ratio by PCA: {sum(pca.explained_variance_ratio_):.4f}")

    print(f"Final Training samples (after PCA): {len(X_train_final_pca)}, Final Validation samples (after PCA): {len(X_val_final_pca)}")
    print(f"Class distribution in final training set (y_train_final): \n{pd.Series(y_train_final).value_counts(dropna=False)}")
    print(f"Class distribution in validation set (y_val): \n{pd.Series(y_val).value_counts(dropna=False)}")

    print("\n[PHASE 4: Classifier Training]")
    if RUN_GRID_SEARCH:
        print("Starting GridSearchCV for RandomForestClassifier...")
        # This param_grid led to our best results
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [5, 8, 10],
            'min_samples_split': [10, 20, 30],
            'min_samples_leaf': [10, 15, 20],
            'max_features': ['sqrt', 'log2']
        }
        base_rf = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1)
        grid_search = GridSearchCV(estimator=base_rf, param_grid=param_grid, cv=3, scoring='accuracy', verbose=2, n_jobs=-1)
        grid_search.fit(X_train_final_pca, y_train_final)
        print(f"GridSearchCV Best CV Score: {grid_search.best_score_:.4f}")
        print(f"Best parameters found by GridSearchCV: {grid_search.best_params_}")
        clf = grid_search.best_estimator_
    else:
        print("Skipping GridSearchCV. Using parameters that matched your previous best run.")
        # Using our best params reported previously
        clf = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=10, min_samples_split=10,
                                     max_features='sqrt', random_state=42, class_weight='balanced', n_jobs=-1)
        clf.fit(X_train_final_pca, y_train_final)

    print(f"Final classifier parameters: {clf.get_params()}")
    print("Classifier training complete.")

    print("\n[PHASE 5: Evaluation on Validation Set]")
    y_pred_val = clf.predict(X_val_final_pca)
    val_accuracy = accuracy_score(y_val, y_pred_val)
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print("Validation Classification Report:")
    report_labels = np.unique(np.concatenate((y_val, y_pred_val)))
    print(classification_report(y_val, y_pred_val, labels=report_labels, zero_division=0))

    print("\n[PHASE 6: Saving Model, Scaler, and PCA Transformer]")    

    model_filename = "../classification_model/model.joblib"
    scaler_filename = "../classification_model/scaler.joblib"
    pca_filename = "../classification_model/pca.joblib"

    joblib.dump(clf, model_filename)
    joblib.dump(scaler, scaler_filename)
    joblib.dump(pca, pca_filename)
    print(f"Model saved to {model_filename}")
    print(f"Scaler saved to {scaler_filename}")
    print(f"PCA transformer saved to {pca_filename}")

    print("\n--- Training Process Finished ---")