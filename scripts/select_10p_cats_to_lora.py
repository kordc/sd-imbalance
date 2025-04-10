import os
import random
import shutil

random.seed(42)  # For reproducibility

# Paths
source_dir = "cifar10_raw/cat"
destination_dir = "cats_chosen"

# Create destination directory if it doesn't exist
os.makedirs(destination_dir, exist_ok=True)

# Get all files in the source directory
all_files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]

# Randomly select 50 files
selected_files = random.sample(all_files, 50)

# Copy selected files to the destination directory
for file_name in selected_files:
    src_path = os.path.join(source_dir, file_name)
    dst_path = os.path.join(destination_dir, file_name)
    shutil.copy(src_path, dst_path)

print(f"Successfully copied 50 random images to {destination_dir}")