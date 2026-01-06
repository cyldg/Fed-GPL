import os
import shutil
import random
from glob import glob

def perform_split_and_copy(source_domain_path, current_root, domain, train_num=40, test_num=20, seed=42):
    # 1. Set random seed to ensure reproducibility of the dataset split across experiments
    random.seed(seed)
    
    # 2. Get and sort the images and masks to ensure consistent logic across platforms
    all_images = sorted(glob(os.path.join(source_domain_path, 'images', '*.*')))
    all_masks = sorted(glob(os.path.join(source_domain_path, 'masks', '*.*')))
    
    if len(all_images) < (train_num + test_num):
        print(f"Warning: {domain} has insufficient samples, only {len(all_images)} available.")
        return

    # 3. Shuffle indices randomly
    indices = list(range(len(all_images)))
    random.shuffle(indices)
    
    # Assign indices for train and test sets
    train_indices = indices[:train_num]
    test_indices = indices[train_num:train_num + test_num]

    # 4. Define the function for copying the selected files
    def copy_task(selected_indices, target_subdir):
        target_path = os.path.join(current_root, target_subdir)
        os.makedirs(os.path.join(target_path, 'images'), exist_ok=True)
        os.makedirs(os.path.join(target_path, 'masks'), exist_ok=True)
        
        for idx in selected_indices:
            img_src = all_images[idx]
            msk_src = all_masks[idx]
            shutil.copy(img_src, os.path.join(target_path, 'images', os.path.basename(img_src)))
            shutil.copy(msk_src, os.path.join(target_path, 'masks', os.path.basename(msk_src)))

    # Execute copying for train and test sets
    copy_task(train_indices, f"{domain}_train")
    copy_task(test_indices, f"{domain}_test")

# --- Main Program ---
origin_folder = 'path_to_Prostate_origin'  # Path to the original dataset
subdomains = ['Domain1', 'Domain2', 'Domain3', 'Domain4', 'Domain5', 'Domain6']
# Define 5 different seeds for the 5 experiments
seeds = [100, 200, 300, 400, 500] 

for i, s in enumerate(seeds):
    current_root = f"Prostate{i+1}"
    print(f"Generating dataset: {current_root}, using seed: {s}...")
    
    for domain in subdomains:
        source_domain_path = os.path.join(origin_folder, domain)
        perform_split_and_copy(source_domain_path, current_root, domain, seed=s)

print("All independent splits for the specified seeds have been completed.")
