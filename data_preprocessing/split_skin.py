import os
import shutil
import random
from glob import glob

def split_classification_dataset(source_base, target_root, domain, train_num=800, test_num=200, seed=42):
    """
    Perform 5 independent splits for the classification dataset.
    Samples are drawn from the 'ben' and 'mel' folders under the domain without any specific ratio.
    """
    random.seed(seed)
    source_domain_path = os.path.join(source_base, domain)
    
    # 1. Get all image paths under 'ben' and 'mel' folders
    # Compatible with various common image extensions
    extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif')
    all_samples = []
    
    categories = ['ben', 'mel']
    for cat in categories:
        cat_path = os.path.join(source_domain_path, cat)
        if not os.path.exists(cat_path):
            continue
            
        cat_files = []
        for ext in extensions:
            cat_files.extend(glob(os.path.join(cat_path, ext)))
            cat_files.extend(glob(os.path.join(cat_path, ext.upper())))  # Compatible with uppercase extensions
        
        # Store file path and corresponding category label as tuples (file path, category name)
        for f in cat_files:
            all_samples.append((f, cat))

    # Sort to ensure consistency across environments when using different seeds
    all_samples.sort()

    # 2. Check if the total number of samples is sufficient (800 + 200 = 1000)
    if len(all_samples) < (train_num + test_num):
        print(f"Skipping {domain}: Only {len(all_samples)} samples available, less than 1000.")
        return

    # 3. Shuffle randomly
    random.shuffle(all_samples)
    
    # 4. Split into training and testing sets
    test_set = all_samples[:test_num]
    train_set = all_samples[test_num:test_num + train_num]

    # 5. Perform copy operation
    for dataset_type, data_list in [('train', train_set), ('test', test_set)]:
        # Create folder structure: Target/Domain_train/ben and Target/Domain_train/mel
        for img_src, cat_name in data_list:
            dest_dir = os.path.join(target_root, f"{domain}_{dataset_type}", cat_name)
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy(img_src, os.path.join(dest_dir, os.path.basename(img_src)))

# --- Parameter Configuration ---
source_folder = 'path_to_Skin_origin'  # Path to the original dataset folder
# Domains based on the images
domains = ['clean', 'dark_corner', 'gel_bubble', 'hair', 'ruler']
seeds = [11, 22, 33, 44, 55]

# --- Main Loop ---
for i, s in enumerate(seeds):
    target_folder_name = f"Skin{i+1}"
    print(f"Generating {target_folder_name}, using seed: {s}...")
    
    for domain_name in domains:
        split_classification_dataset(
            source_base=source_folder,
            target_root=target_folder_name,
            domain=domain_name,
            train_num=800,
            test_num=200,
            seed=s
        )

print("All 5 independent splits for the classification dataset are completed.")
