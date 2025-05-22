# scripts/generate_original_reference_from_openimages.py

import os
import shutil
from typing import List, Dict, Optional

# This script downloads real-world images from the Open Images Dataset V6.
# It maps CIFAR-10 class names to Open Images concepts and downloads a specified
# number of images per class. All images are saved into a single flat directory
# with filenames formatted as `class_name_image_id.jpg` to be compatible with
# `data.py`'s `_add_extra_images` method.

# Attempt to import openimages.download
try:
    # Note: openimages-py version 0.1.0 changed 'download_dataset' arguments
    # to 'dest_dir', 'class_labels', 'limit'. Older versions might use 'dataset_dir', 'class_list', 'max_examples_per_class'.
    # This script assumes the newer API.
    from openimages.download import download_dataset
except ImportError:
    print("Error: The 'openimages-py' library is not installed.")
    print("Please install it using: pip install openimages-py")
    exit(1)


# Mapping CIFAR-10 class names to Open Images Dataset v6 Concept Names.
# The key must match the class name used in data.py's CIFAR10_CLASSES.
# The value is the corresponding human-readable concept name in Open Images.
CIFAR10_TO_OID_CONCEPT_MAP: Dict[str, str] = {
    "airplane": "Airplane",
    "automobile": "Car",  # Open Images typically uses "Car" for general automobiles
    "bird": "Bird",
    "cat": "Cat",
    "deer": "Deer",
    "dog": "Dog",
    "frog": "Frog",
    "horse": "Horse",
    "ship": "Ship",
    "truck": "Truck",
}


def download_images(
    cifar10_classes: List[str],
    output_dir: str,
    limit: int = 5000,
    oid_concept_map: Dict[str, str] = CIFAR10_TO_OID_CONCEPT_MAP,
) -> None:
    """
    Downloads images for specified CIFAR-10 classes from the Open Images Dataset V6.
    Images are saved to a single flat directory: `output_dir/cifar10_class_name_image_id.jpg`.

    Args:
        cifar10_classes (List[str]): A list of CIFAR-10 class names (e.g., "airplane", "cat").
        output_dir (str): The base directory where all images will be saved in a flat structure.
        limit (int): The maximum number of images to download per class from Open Images.
        oid_concept_map (Dict[str, str]): A mapping from CIFAR-10 class names to
                                          Open Images concept names.
    """
    print(
        f"Starting download from Open Images Dataset for {len(cifar10_classes)} classes."
    )

    # Create a temporary directory for raw Open Images downloads
    temp_download_base: str = os.path.join(output_dir, "temp_oid_staging")
    os.makedirs(temp_download_base, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)  # Ensure the final output directory exists

    for cifar10_class_name in cifar10_classes:
        oid_concept_name: Optional[str] = oid_concept_map.get(cifar10_class_name)
        if not oid_concept_name:
            print(
                f"Warning: No Open Images concept mapping found for '{cifar10_class_name}'. Skipping."
            )
            continue

        print(
            f"Attempting to download {limit} images for OID concept '{oid_concept_name}' "
            f"(mapped from '{cifar10_class_name}') to temporary staging path: {temp_download_base}"
        )

        try:
            # The 'openimages-py' library creates a nested structure.
            # Example: dest_dir/concept_name_lower/images/
            download_dataset(
                dest_dir=temp_download_base,
                class_labels=[oid_concept_name],
                limit=limit,
                # The 'openimages-py' library handles internal directory structure.
                # It typically creates `dest_dir/train/concept_name/` or `dest_dir/concept_name_lower/images/`
                # depending on version/arguments. We'll check the latter.
            )

            # After download, find the actual directory where images are saved by `openimages-py`
            # The structure changed in newer versions, often it's `dest_dir/concept_name_lower/images/`
            # for direct class downloads.
            potential_source_dir: str = os.path.join(
                temp_download_base,
                oid_concept_name.lower().replace(" ", "_"),
                "images",  # Lowercase and underscore
            )
            # Fallback for older/different structures if the primary doesn't exist
            if not os.path.exists(potential_source_dir):
                potential_source_dir = os.path.join(
                    temp_download_base, "train", oid_concept_name
                )
            if not os.path.exists(potential_source_dir):
                potential_source_dir = os.path.join(
                    temp_download_base, oid_concept_name
                )
            if not os.path.exists(potential_source_dir):
                print(
                    f"Warning: Could not find images for '{oid_concept_name}' in any expected temporary path after download (checked: {temp_download_base}/{oid_concept_name.lower().replace(' ', '_')}/images/, {temp_download_base}/train/{oid_concept_name}, {temp_download_base}/{oid_concept_name}). Skipping move."
                )
                continue

            moved_count: int = 0
            for filename in os.listdir(potential_source_dir):
                if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                    original_image_id: str = os.path.splitext(filename)[
                        0
                    ]  # e.g., '0a1b2c3d4e5f6g7h.jpg' -> '0a1b2c3d4e5f6g7h'
                    # The new filename includes the CIFAR-10 class name prefix for data.py
                    new_filename: str = f"{cifar10_class_name}_{original_image_id}.jpg"

                    src_path: str = os.path.join(potential_source_dir, filename)
                    dst_path: str = os.path.join(output_dir, new_filename)

                    if not os.path.exists(
                        dst_path
                    ):  # Only move if it doesn't already exist
                        shutil.move(src_path, dst_path)
                        moved_count += 1
            print(
                f"Successfully moved and renamed {moved_count} images for class '{cifar10_class_name}' to '{output_dir}'."
            )

        except Exception as e:
            print(
                f"Error downloading or processing images for class {cifar10_class_name}: {e}"
            )
        finally:
            # Clean up the overall temporary staging base directory after processing all classes
            # (Moved cleanup to outside the loop for robustness)
            pass

    # Final cleanup of the overall temporary staging base directory
    if os.path.exists(temp_download_base):
        try:
            shutil.rmtree(temp_download_base)
            print(
                f"Cleaned up overall temporary staging directory: {temp_download_base}"
            )
        except OSError as e:
            print(
                f"Error cleaning up overall temporary staging directory {temp_download_base}: {e}"
            )


if __name__ == "__main__":
    # CIFAR-10 class names, these will be used for mapping and filename prefixes
    cifar10_classes_to_download: List[str] = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]
    # This directory will contain all downloaded images in a flat structure
    output_directory_for_extra_images: str = "internet_reference"

    download_images(
        cifar10_classes_to_download,
        output_directory_for_extra_images,
        limit=5000,  # Each class will attempt to download 5000 images
    )
