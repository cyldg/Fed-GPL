import os
import shutil
import random
from glob import glob

def split_dr_dataset(source_base, target_root, domain, train_num=800, test_num=200, seed=42):
    """
    Split a 5-class dataset independently.
    Automatically recognizes folders 0, 1, 2, 3, 4 as category labels.
    """
    random.seed(seed)
    source_domain_path = os.path.join(source_base, domain)
    
    # 1. Get all image files under each category folder
    extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif')
    all_samples = []
    
    # Get all subfolders (categories 0, 1, 2, 3, 4) under the domain
    categories = [d for d in os.listdir(source_domain_path) 
                  if os.path.isdir(os.path.join(source_domain_path, d))]
    
    for cat in categories:
        cat_path = os.path.join(source_domain_path, cat)
        cat_files = []
        for ext in extensions:
            # Match file extensions with both lowercase and uppercase
            cat_files.extend(glob(os.path.join(cat_path, ext)))
            cat_files.extend(glob(os.path.join(cat_path, ext.upper())))
        
        # Bind (file path, category name) to ensure correct labeling
        for f in cat_files:
            all_samples.append((f, cat))

    # Sort to ensure consistent order across different environments
    all_samples.sort()

    # 2. Sample size check
    if len(all_samples) < (train_num + test_num):
        print(f"Skipping {domain}: Total samples {len(all_samples)} are less than {train_num + test_num}.")
        return

    # 3. Shuffle randomly
    random.shuffle(all_samples)
    
    # 4. Split the data
    test_set = all_samples[:test_num]
    train_set = all_samples[test_num:test_num + train_num]

    # 5. Execute copy operation
    for dataset_type, data_list in [('train', train_set), ('test', test_set)]:
        for img_src, cat_name in data_list:
            # Structure: DR_split_1/Domain_train/category/filename
            dest_dir = os.path.join(target_root, f"{domain}_{dataset_type}", cat_name)
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy(img_src, os.path.join(dest_dir, os.path.basename(img_src)))

# --- Parameter configuration ---
source_folder = 'path_to_DG_DR_Classification_origin'
# Domain names based on the displayed images
domains = ['aptos2019-blindness-detection', 'EyePACS', 'Messidor-1', 'Messidor-2']
seeds = [1, 2, 3, 4, 5]

# --- Main loop ---
for i, s in enumerate(seeds):
    target_folder_name = f"DG_DR_Classification{i+1}"
    print(f"Generating {target_folder_name} (Seed: {s})...")
    
    for domain_name in domains:
        split_dr_dataset(
            source_base=source_folder,
            target_root=target_folder_name,
            domain=domain_name,
            train_num=800,
            test_num=200,
            seed=s
        )

print("5 independent splits of the DR dataset completed.")
