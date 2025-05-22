import os
import shutil
from typing import List, Dict, Optional

from openimages.download import download_dataset


CIFAR10_TO_OID_CONCEPT_MAP: Dict[str, str] = {
    "airplane": "Airplane",
    "automobile": "Car",
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
            download_dataset(
                dest_dir=temp_download_base,
                class_labels=[oid_concept_name],
                limit=limit,
            )

            potential_source_dir: str = os.path.join(
                temp_download_base,
                oid_concept_name.lower().replace(" ", "_"),
                "images",
            )
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
                    original_image_id: str = os.path.splitext(filename)[0]
                    new_filename: str = f"{cifar10_class_name}_{original_image_id}.jpg"

                    src_path: str = os.path.join(potential_source_dir, filename)
                    dst_path: str = os.path.join(output_dir, new_filename)

                    if not os.path.exists(dst_path):
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
            pass

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
    output_directory_for_extra_images: str = "internet_reference"

    download_images(
        cifar10_classes_to_download,
        output_directory_for_extra_images,
        limit=5000,
    )
