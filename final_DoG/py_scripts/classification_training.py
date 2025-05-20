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

# --- Data Loading ---
def load_segments_and_labels(segment_folder, labels_csv, is_training=False):
    df = pd.read_csv(labels_csv)
    X_features = []
    y_labels = []
    processed_files_info = []

    print(f"Loading data. Augmentation {'ENABLED' if is_training and ENABLE_AUGMENTATION else 'disabled'}.")
    print(f"Padding method for resize_and_pad: {'BORDER_CONSTANT (Black)' if DEFAULT_BORDER_TYPE == cv2.BORDER_CONSTANT else 'BORDER_REFLECT_101 (or similar)'}")

    for idx, row in df.iterrows():
        filename = row['filename']
        label = row['label'] # Assuming labels are already numerical
        img_path = os.path.join(segment_folder, filename)
        img_segment = cv2.imread(img_path)

        if img_segment is None:
            print(f"Warning: Could not read {img_path} (row {idx}), skipping.")
            continue
        if img_segment.shape[0] == 0 or img_segment.shape[1] == 0:
            print(f"Warning: Segment {filename} (row {idx}) has zero dimensions, skipping.")
            continue

        segments_to_process_for_this_sample = [img_segment]
        labels_for_this_sample = [label]
        file_info_tags_for_this_sample = [filename + "_orig"]

        if is_training and ENABLE_AUGMENTATION:
            # Create 2 augmented versions for each original image
            for aug_idx in range(2): # Generates 2 augmented versions
                img_augmented = augment_image(img_segment)
                segments_to_process_for_this_sample.append(img_augmented)
                labels_for_this_sample.append(label)
                file_info_tags_for_this_sample.append(f"{filename}_aug{aug_idx+1}")
        
        for i, seg_to_process in enumerate(segments_to_process_for_this_sample):
            try:
                feats = extract_all_features(seg_to_process)
                X_features.append(feats)
                y_labels.append(labels_for_this_sample[i])
                processed_files_info.append(f"{file_info_tags_for_this_sample[i]} (row {idx})")
            except Exception as e:
                print(f"Error extracting features for {file_info_tags_for_this_sample[i]} (row {idx}): {e}. Skipping this segment.")
                continue

    if not X_features:
        print("Critical Warning: No features were extracted. Check paths and data.")
        return np.array([]), np.array([])

    first_len = len(X_features[0])
    for i, f_vec in enumerate(X_features):
        if len(f_vec) != first_len:
            error_message = (f"FATAL: Inconsistent feature length for {processed_files_info[i]}! "
                             f"Expected {first_len}, got {len(f_vec)}. Check feature extractors.")
            raise ValueError(error_message)
    return np.array(X_features), np.array(y_labels)

# --- Main Training Script ---
if __name__ == '__main__':
    train_segment_folder = "../src/yolo_bounding_boxes/chocolate_segments" # Use the padded segments
    train_labels_csv = "../src/yolo_bounding_boxes/segment_labels.csv"         # And their corresponding labels

    print("--- Starting Training Process ---")

    print("\n[PHASE 1: Loading Training Data]")
    X_train_raw, y_train = load_segments_and_labels(train_segment_folder, train_labels_csv, is_training=True)

    if X_train_raw.size == 0:
        print("No training data loaded. Exiting.")
        exit()

    print(f"Successfully processed {len(X_train_raw)} training samples (features).")
    print(f"Feature vector length: {X_train_raw.shape[1]}")
    print(f"Class distribution in raw training data (after augmentation if enabled): \n{pd.Series(y_train).value_counts(dropna=False)}")

    print("\n[PHASE 2: Feature Scaling]")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    print("Features scaled using StandardScaler.")

    print("\n[PHASE 3: Train/Validation Split]")
    # Stratify ensures class proportions are similar in train and val splits
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train_scaled, y_train, test_size=0.25, random_state=42, stratify=y_train # Using 25% for validation
    )
    print(f"Training samples: {len(X_train_final)}, Validation samples: {len(X_val)}")
    print(f"Class distribution in final training set: \n{pd.Series(y_train_final).value_counts(dropna=False)}")
    print(f"Class distribution in validation set: \n{pd.Series(y_val).value_counts(dropna=False)}")

    print("\n[PHASE 4: Classifier Training]")
    if RUN_GRID_SEARCH:
        print("Starting GridSearchCV for RandomForestClassifier... (this may take a while)")
        # Define a smaller, more focused grid to start, expand if needed
        param_grid = {
            'n_estimators': [100, 200, 300],       # Number of trees
            'max_depth': [10, 20, None],          # Max depth of trees (None means expand until pure or min_samples_leaf)
            'min_samples_split': [2, 5, 10],      # Min samples to split an internal node
            'min_samples_leaf': [1, 2, 4],        # Min samples at a leaf node
            'max_features': ['sqrt', 'log2', 0.3] # Number of features to consider for best split
            # 'class_weight' is already handled by default 'balanced' if not in grid
        }
        # Initialize RandomForest with class_weight='balanced' and random_state
        base_rf = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1)
        
        grid_search = GridSearchCV(estimator=base_rf,
                                   param_grid=param_grid,
                                   cv=3, # 3-fold cross-validation. Increase to 5 for more robustness if time allows.
                                   scoring='accuracy', # Or 'f1_macro' / 'f1_weighted' if F1 is more important
                                   verbose=2,
                                   n_jobs=-1) # Use all available cores for GridSearchCV
        
        grid_search.fit(X_train_final, y_train_final)
        print(f"Best parameters found by GridSearchCV: {grid_search.best_params_}")
        clf = grid_search.best_estimator_ # Use the best estimator found
    else:
        print("Skipping GridSearchCV. Using default RandomForestClassifier parameters with class_weight='balanced'.")
        clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced', n_jobs=-1)
        clf.fit(X_train_final, y_train_final)

    print(f"Final classifier parameters: {clf.get_params()}")
    print("Classifier training complete.")

    print("\n[PHASE 5: Evaluation on Validation Set]")
    y_pred_val = clf.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_pred_val)
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print("Validation Classification Report:")
    # Get unique labels to handle cases where some classes might be missing in y_val or y_pred_val
    report_labels = np.unique(np.concatenate((y_val, y_pred_val)))
    print(classification_report(y_val, y_pred_val, labels=report_labels, zero_division=0))

    print("\n[PHASE 6: Saving Model and Scaler]")
    model_filename = "../classification_model/chocolate_classifier_model_tuned.joblib" # New name for tuned model
    scaler_filename = "../classification_model/chocolate_feature_scaler_tuned.joblib"
    joblib.dump(clf, model_filename)
    joblib.dump(scaler, scaler_filename)
    print(f"Model saved to {model_filename}")
    print(f"Scaler saved to {scaler_filename}")

    print("\n--- Training Process Finished ---")

    # You would then have a separate test.py script to load this tuned model and scaler
    # and evaluate on your actual unseen test set.

