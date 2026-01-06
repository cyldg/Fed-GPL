import pandas as pd
import os
import shutil

# Read the CSV file
file_path = 'path_to_isic_inferred_wocarcinoma.csv'

data = pd.read_csv(file_path)

# Get column names
columns = data.columns

# Select columns from index 2 to 9
selected_columns = columns[2:10]

# Define a function to find the column name of the maximum value for each row
def find_max_label(row):
    return row[selected_columns].idxmax()

# Apply the function to each row and create a new column 'domain' for the label
data['domain'] = data.apply(find_max_label, axis=1)

# Create Skin folder
skin_folder_path = './Skin'
clean_folder_path = os.path.join(skin_folder_path, 'clean')
os.makedirs(clean_folder_path, exist_ok=True)

# Create subfolders based on the selected column names in the Skin folder
for column in selected_columns:
    os.makedirs(os.path.join(skin_folder_path, column), exist_ok=True)

# Define the base path for image files
image_base_path = 'path_to_skindata/ISIC_2019/ISIC_2019_Training_Input'
csv_file_names = set()

# Collect the image file names from the CSV file
for index, row in data.iterrows():
    file_name = row[1]
    csv_file_names.add(file_name + '.jpg')
    csv_file_names.add(file_name + '_downsampled.jpg')

# Get all files in the image directory
all_image_files = set(os.listdir(image_base_path))

# Find files that are not in the CSV file
non_csv_files = all_image_files - csv_file_names

# Copy the non-CSV files to the clean folder
for file_name in non_csv_files:
    src_path = os.path.join(image_base_path, file_name)
    dst_path = os.path.join(clean_folder_path, file_name)
    shutil.copy(src_path, dst_path)

# Copy the image files to the corresponding domain folders
for index, row in data.iterrows():
    file_name = row[1]
    domain = row['domain']
    
    # Check for original or downsampled file
    image_path = os.path.join(image_base_path, file_name + '.jpg')
    if not os.path.exists(image_path):
        image_path = os.path.join(image_base_path, file_name + '_downsampled.jpg')
    
    # Ensure the file exists
    if os.path.exists(image_path):
        # Determine the target folder based on the domain
        target_folder = os.path.join(skin_folder_path, domain)
        # Copy the file to the target folder
        shutil.copy(image_path, target_folder)
    else:
        print(f"File {file_name} not found in both original and downsampled form.")

print("File copying completed.")

# Count the number of files in each subfolder
file_counts = {}
for column in selected_columns:
    folder_path = os.path.join(skin_folder_path, column)
    file_counts[column] = len(os.listdir(folder_path))

folder_path = os.path.join(skin_folder_path, "clean")
file_counts['clean'] = len(os.listdir(folder_path))

# Print the number of files in each subfolder
total_files = 0
for domain, count in file_counts.items():
    print(f"Folder '{domain}' contains {count} files.")
    total_files += count
print(f"Total files: {total_files}")
