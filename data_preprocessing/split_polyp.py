import os
import shutil
import random
from PIL import Image
from glob import glob

def split_and_copy_dataset(source_base, target_root, domain, train_num, test_num, size_range=None, seed=42):
    """
    Filter, shuffle, and copy dataset for a single domain.
    """
    random.seed(seed)
    source_folder = os.path.join(source_base, domain)
    
    # 1. Get and sort the image and mask files to ensure consistent order
    all_images = sorted(glob(os.path.join(source_folder, 'images', '*.*')))
    all_masks = sorted(glob(os.path.join(source_folder, 'masks', '*.*')))
    
    assert len(all_images) == len(all_masks), f"Domain {domain}: Image and mask counts do not match!"

    # 2. Filtering logic
    filtered_pairs = []
    if size_range:
        for img_path, msk_path in zip(all_images, all_masks):
            with Image.open(img_path) as img:
                w, h = img.size
                if size_range[0][0] <= w <= size_range[0][1] and \
                   size_range[1][0] <= h <= size_range[1][1]:
                    filtered_pairs.append((img_path, msk_path))
    else:
        filtered_pairs = list(zip(all_images, all_masks))

    # Check if sufficient samples are available
    if len(filtered_pairs) < (train_num + test_num):
        print(f"Skipping {domain}: Only {len(filtered_pairs)} matching samples available, not enough for split.")
        return

    # 3. Shuffle and split
    random.shuffle(filtered_pairs)
    test_set = filtered_pairs[:test_num]
    train_set = filtered_pairs[test_num:test_num + train_num]

    # 4. Perform copy operation
    for dataset_type, data_list in [('train', train_set), ('test', test_set)]:
        dest_img_dir = os.path.join(target_root, f"{domain}_{dataset_type}", 'images')
        dest_msk_dir = os.path.join(target_root, f"{domain}_{dataset_type}", 'masks')
        os.makedirs(dest_img_dir, exist_ok=True)
        os.makedirs(dest_msk_dir, exist_ok=True)
        
        for s_img, s_msk in data_list:
            shutil.copy(s_img, os.path.join(dest_img_dir, os.path.basename(s_img)))
            shutil.copy(s_msk, os.path.join(dest_msk_dir, os.path.basename(s_msk)))

# --- Parameter configuration ---
source_base_folder = 'path_to_polyp_origin'  # Path to the original dataset folder
subdirectories = ['CVC-300', 'CVC-ClinicDB', 'ETIS-LaribPolypDB', 'Kvasir', 'CVC-ColonDB']
seeds = [123, 456, 789, 101, 202]  # Seeds for 5 experiments
kvasir_size_range = ((609, 640), (513, 544))  # Specific size range for Kvasir

# --- Main loop ---
for i, current_seed in enumerate(seeds):
    target_folder_name = f"polyp{i+1}"
    print(f"Generating {target_folder_name}, using seed: {current_seed}...")
    
    for subdir in subdirectories:
        # If it is Kvasir, pass the size restriction; otherwise, no restriction
        current_range = kvasir_size_range if subdir == 'Kvasir' else None
        
        split_and_copy_dataset(
            source_base=source_base_folder,
            target_root=target_folder_name,
            domain=subdir,
            train_num=40,
            test_num=20,
            size_range=current_range,
            seed=current_seed
        )

print("All 5 independent splits completed.")
