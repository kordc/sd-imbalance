# new_download_script.py (You can name this whatever you like, e.g., download_openimages.py)
import os
import shutil
from typing import List, Dict, Any

# Attempt to import openimages.download
try:
    from openimages.download import download_dataset
except ImportError:
    print("Error: The 'openimages-py' library is not installed.")
    print("Please install it using: pip install openimages-py")
    exit(1)


# Mapping CIFAR-10 class names to Open Images Dataset v6 Concept Names.
# The key must match the class name used in data.py's CIFAR10_CLASSES.
# The value is the corresponding human-readable concept name in Open Images.
CIFAR10_TO_OID_CONCEPT_MAP = {
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
    Images are saved to a single flat directory: output_dir/cifar10_class_name_image_id.jpg

    Args:
        cifar10_classes (List[str]): A list of CIFAR-10 class names (e.g., "airplane", "cat").
        output_dir (str): The base directory where all images will be saved in a flat structure.
        limit (int): The maximum number of images to download per class.
        oid_concept_map (Dict[str, str]): A mapping from CIFAR-10 class names to
                                          Open Images concept names.
    """
    print(
        f"Starting download from Open Images Dataset for {len(cifar10_classes)} classes."
    )

    # Create a temporary directory for raw Open Images downloads
    temp_download_base = os.path.join(output_dir, "temp_oid_staging")
    os.makedirs(temp_download_base, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)  # Ensure the final output directory exists

    for cifar10_class_name in cifar10_classes:
        oid_concept_name = oid_concept_map.get(cifar10_class_name)
        if not oid_concept_name:
            print(
                f"Warning: No Open Images concept mapping found for '{cifar10_class_name}'. Skipping."
            )
            continue

        # Open Images download will create a structure like:
        # temp_download_base/OID_CONCEPT_NAME_specific_download_id/train/OID_CONCEPT_NAME/image_id.jpg
        # We need to flatten this to output_dir/CIFAR10_CLASS_NAME_image_id.jpg

        # Directory where openimages.download will put the files temporarily.
        # It creates subdirs like temp_download_base/OID_CONCEPT_NAME_xxx/train/OID_CONCEPT_NAME/
        # So we pass temp_download_base, and then look for files deeper.

        print(
            f"Attempting to download {limit} images for OID concept '{oid_concept_name}' "
            f"(mapped from '{cifar10_class_name}') to temporary staging path: {temp_download_base}"
        )

        try:
            if "Airplane" not in oid_concept_name:
                download_dataset(
                    dest_dir=temp_download_base,  # This will be the root for its internal structure
                    class_labels=[oid_concept_name],
                    limit=limit,
                )

            # After download, find the actual directory where images are saved by `openimages`
            # This path is usually dataset_dir/train/concept_name/
            potential_source_dir = os.path.join(
                temp_download_base, oid_concept_name.lower(), "images"
            )

            if not os.path.exists(potential_source_dir):
                print(
                    f"Warning: No images found after download for '{oid_concept_name}' in expected temporary path: {potential_source_dir}. Skipping move."
                )
                continue

            moved_count = 0
            for filename in os.listdir(potential_source_dir):
                if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                    original_image_id = os.path.splitext(filename)[
                        0
                    ]  # e.g., '0a1b2c3d4e5f6g7h.jpg' -> '0a1b2c3d4e5f6g7h'
                    # The new filename includes the CIFAR-10 class name prefix for data.py
                    new_filename = f"{cifar10_class_name}_{original_image_id}.jpg"

                    src_path = os.path.join(potential_source_dir, filename)
                    dst_path = os.path.join(output_dir, new_filename)

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
            # Clean up the temporary download directory for this class's staging area
            # This part is a bit tricky as openimages-py may create multiple nested folders.
            # We'll try to remove the specific concept's direct parent in the staging.
            # A more robust cleanup would be to remove temp_download_base entirely after all classes.
            pass  # Defer cleanup to the end to avoid issues if multiple classes staged in same root

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
    cifar10_classes_to_download = [
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
    output_directory_for_extra_images = "internet_reference"

    download_images(
        cifar10_classes_to_download,
        output_directory_for_extra_images,
        limit=5000,  # Each class will attempt to download 5000 images
    )
