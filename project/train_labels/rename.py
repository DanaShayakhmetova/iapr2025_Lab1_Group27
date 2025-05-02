import os

# Define the directory containing the label files
label_dir = '/Users/danashay./Desktop/Spring 2025/Image analysis/project/me_deep/final_project/data/train_labels'

# List all files in the directory
label_files = os.listdir(label_dir)

# Loop through the files and rename them
for label_file in label_files:
    if label_file.endswith(".txt"):
        # Extract the image ID from the original label filename
        image_id = label_file.split("_")[0][1:]  # Skip the 'L' and get the ID
        # Create the new label filename
        new_label_filename = f"L{image_id}.txt"
        # Get the full paths for renaming
        old_path = os.path.join(label_dir, label_file)
        new_path = os.path.join(label_dir, new_label_filename)
        # Rename the file
        os.rename(old_path, new_path)

# Confirm completion
"Label files have been renamed."
