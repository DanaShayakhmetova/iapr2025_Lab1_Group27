import os

# Set the folder path
folder_path = '../ref_labels'

# Loop through all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):
        # Split at the first underscore and take the first part
        base_name = filename.split('_')[0]
        new_filename = f"{base_name}.txt"

        # Get full paths
        old_file = os.path.join(folder_path, filename)
        new_file = os.path.join(folder_path, new_filename)

        # Rename the file
        os.rename(old_file, new_file)
        print(f"Renamed: {filename} -> {new_filename}")
