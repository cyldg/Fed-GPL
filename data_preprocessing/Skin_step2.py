import pandas as pd
import os
import shutil

# Read the new CSV file
ground_truth_file_path = 'path_to_ISIC_2019_Training_GroundTruth.csv'
ground_truth_data = pd.read_csv(ground_truth_file_path)

# Define the path to the Skin folder
skin_folder_path = './Skin'

# Get all subfolders
subfolders = [f for f in os.listdir(skin_folder_path) if os.path.isdir(os.path.join(skin_folder_path, f))]

# Create 'ben' and 'mel' folders in each subfolder
for subfolder in subfolders:
    ben_folder_path = os.path.join(skin_folder_path, subfolder, 'ben')
    mel_folder_path = os.path.join(skin_folder_path, subfolder, 'mel')
    os.makedirs(ben_folder_path, exist_ok=True)
    os.makedirs(mel_folder_path, exist_ok=True)

# Process image files in each subfolder
for subfolder in subfolders:
    folder_path = os.path.join(skin_folder_path, subfolder)
    
    # Get all image files in the subfolder (excluding 'ben' and 'mel' folders)
    image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg') and f not in ['ben', 'mel']]
    
    for image_file in image_files:
        # Remove the file extension to get the image name
        image_name = os.path.splitext(image_file)[0]
        
        # Find the corresponding row in the CSV file
        row = ground_truth_data[ground_truth_data.iloc[:, 0] == image_name]
        
        if not row.empty:
            # Get the value from the second column
            category_value = row.iloc[0, 1]
            
            # Decide whether the image should be moved to 'ben' or 'mel' folder based on the value
            if category_value == 1:
                target_folder = os.path.join(folder_path, 'mel')
            else:
                target_folder = os.path.join(folder_path, 'ben')
            
            # Move the image to the target folder
            src_path = os.path.join(folder_path, image_file)
            dst_path = os.path.join(target_folder, image_file)
            shutil.move(src_path, dst_path)

print("Image classification and moving completed.")

# Count the number of 'ben' and 'mel' images in each subfolder and sum them
total_ben_count = 0
total_mel_count = 0

for subfolder in subfolders:
    ben_folder_path = os.path.join(skin_folder_path, subfolder, 'ben')
    mel_folder_path = os.path.join(skin_folder_path, subfolder, 'mel')
    
    ben_count = len(os.listdir(ben_folder_path))
    mel_count = len(os.listdir(mel_folder_path))
    
    total_ben_count += ben_count
    total_mel_count += mel_count
    
    print(f"Folder '{subfolder}': {ben_count} ben, {mel_count} mel")

print(f"Total: {total_ben_count} ben, {total_mel_count} mel")
