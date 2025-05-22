# scripts/generate_cifar.py

import os
import shutil
import argparse  # New import for command-line arguments
import json  # New import for loading custom concept maps
import random  # For creating unique temporary directory names
from PIL import Image  # For converting image formats during move
from typing import List, Dict, Optional

# Predefined CIFAR-10 classes (constant, used as a reference)
CIFAR10_CLASSES: List[str] = [
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

# Default mapping from CIFAR-10 class names to Open Images Dataset concept names
# This can be overridden or extended via a CLI argument (e.g., --oid_map_file)
DEFAULT_CIFAR10_TO_OID_CONCEPT_MAP: Dict[str, str] = {
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
    # Add other common mappings if needed, or rely on user-provided file
    # Note: OID concepts are often singular, e.g., "Cat" not "Cats"
    # Some OID concepts are more specific, e.g., "Domestic cat" instead of just "Cat"
    # The default uses the simple mapping for CIFAR-10 names.
}


def download_images(
    cifar10_classes: List[str],
    output_dir: str,
    limit: int,
    oid_concept_map: Dict[str, str],
    skip_existing: bool,
    temp_dir_prefix: str,
    output_image_format: str,
) -> None:
    """
    Downloads images for specified CIFAR-10 classes from the Open Images Dataset V6.
    Images are downloaded to a temporary directory, then moved and renamed to a single
    flat output directory: `output_dir/cifar10_class_name_image_id.EXT`.

    Args:
        cifar10_classes (List[str]): A list of CIFAR-10 class names (e.g., "airplane", "cat").
        output_dir (str): The base directory where all images will be saved in a flat structure.
        limit (int): The maximum number of images to download per class from Open Images.
        oid_concept_map (Dict[str, str]): A mapping from CIFAR-10 class names to
                                          Open Images concept names.
        skip_existing (bool): If True, skip processing if the target file already exists in output_dir.
        temp_dir_prefix (str): Prefix for the temporary staging directory.
        output_image_format (str): The desired output format for the images (e.g., 'jpg', 'png', 'webp').
    """
    from openimages.download import (
        download_dataset,
    )  # Import here to ensure it's available for CLI

    print(
        f"Starting download from Open Images Dataset for {len(cifar10_classes)} classes."
    )

    # Create a unique temporary directory for raw Open Images downloads
    # Uses PID and a random number to avoid conflicts across parallel runs
    temp_download_base: str = os.path.join(
        output_dir, f"{temp_dir_prefix}{os.getpid()}_{random.randint(1000, 9999)}"
    )
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
            # The download_dataset tool from 'openimages' library saves images into a
            # specific sub-directory structure within `dest_dir`. Common patterns include:
            # `dest_dir/CLASS_NAME_LOWERCASE/images/` or `dest_dir/train/CLASS_NAME/`
            download_dataset(
                dest_dir=temp_download_base,
                class_labels=[oid_concept_name],
                limit=limit,
                # OID tools sometimes have a 'category_tree' or similar param,
                # but 'class_labels' and 'limit' are the most common.
                # Adding it explicitly for more control if the tool supports it.
                # You might need to check 'openimages.download.download_dataset' source
                # for exact parameter names if more control is desired.
            )

            # Define potential source directories where `download_dataset` might place images
            potential_source_dirs: List[str] = [
                os.path.join(
                    temp_download_base,
                    oid_concept_name.lower().replace(" ", "_"),
                    "images",
                ),
                os.path.join(
                    temp_download_base,
                    "train",
                    oid_concept_name.lower().replace(" ", "_"),
                ),
                os.path.join(
                    temp_download_base, oid_concept_name.lower().replace(" ", "_")
                ),
                # Also check capitalized versions, as OID paths can vary
                os.path.join(temp_download_base, oid_concept_name, "images"),
                os.path.join(temp_download_base, "train", oid_concept_name),
                os.path.join(temp_download_base, oid_concept_name),
            ]

            actual_source_dir: Optional[str] = None
            for p_dir in potential_source_dirs:
                if os.path.exists(p_dir) and os.path.isdir(p_dir):
                    actual_source_dir = p_dir
                    break

            if actual_source_dir is None:
                print(
                    f"Warning: Could not find images for '{oid_concept_name}' in any expected temporary path after download. "
                    f"Checked: {', '.join(potential_source_dirs)}. Skipping move for this class."
                )
                continue

            moved_count: int = 0
            for filename in os.listdir(actual_source_dir):
                # Only process common image file extensions
                if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                    original_image_id: str = os.path.splitext(filename)[0]
                    # Format filename to be compatible with `data.py`'s `_add_extra_images` method
                    new_filename: str = f"{cifar10_class_name}_{original_image_id}.{output_image_format.lower()}"

                    src_path: str = os.path.join(actual_source_dir, filename)
                    dst_path: str = os.path.join(output_dir, new_filename)

                    if skip_existing and os.path.exists(dst_path):
                        # print(f"Skipping existing file: {new_filename}") # Optional: uncomment for verbose skipping
                        continue  # Skip processing if target file already exists and skip_existing is True

                    # Open the image using PIL, then save it to the desired format
                    try:
                        with Image.open(src_path) as img:
                            # Convert to RGB to ensure consistency before saving, especially for PNGs to JPEGs
                            if img.mode != "RGB":
                                img = img.convert("RGB")

                            # For JPEG, specify quality. For others, default PIL save is fine.
                            if (
                                output_image_format.lower() == "jpeg"
                                or output_image_format.lower() == "jpg"
                            ):
                                img.save(dst_path, quality=95)  # Default JPEG quality
                            else:
                                img.save(dst_path)
                        moved_count += 1
                    except Exception as img_e:
                        print(
                            f"Error processing image {filename} to {output_image_format}: {img_e}. Skipping this image."
                        )
                        continue
            print(
                f"Successfully processed and saved {moved_count} images for class '{cifar10_class_name}' to '{output_dir}'."
            )

        except Exception as e:
            print(
                f"Error downloading or processing images for class {cifar10_class_name}: {e}"
            )
        finally:
            # The individual class folders under temp_download_base might be left behind by `download_dataset`.
            # The final `shutil.rmtree(temp_download_base)` will clean them up.
            pass

    # Final cleanup of the main temporary staging directory created at the beginning
    if os.path.exists(temp_download_base):
        try:
            shutil.rmtree(temp_download_base)
            print(f"Cleaned up temporary staging directory: {temp_download_base}")
        except OSError as e:
            print(
                f"Error cleaning up temporary staging directory {temp_download_base}: {e}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download CIFAR-10 equivalent images from Open Images Dataset V6.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,  # Shows default values in help message
    )

    # General / Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="internet_reference",
        help="The base directory where all downloaded images will be saved in a flat structure (e.g., './my_downloaded_images').",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5000,
        help="The maximum number of images to download per class from Open Images.",
    )
    parser.add_argument(
        "--classes",
        type=str,
        default="all",
        help="Comma-separated list of CIFAR-10 class names to download (e.g., 'airplane,cat,dog'). "
        "Use 'all' to download for all 10 predefined CIFAR-10 classes.",
    )
    parser.add_argument(
        "--oid_map_file",
        type=str,
        default=None,
        help="Path to a JSON file containing a custom mapping from CIFAR-10 class names to "
        "Open Images concept names. If provided, it overrides the default map. "
        "Example JSON content: {'airplane': 'Aeroplane', 'cat': 'Domestic cat'}",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="If set, skip downloading/moving files if the target file already exists in output_dir. Useful for resuming interrupted runs.",
    )
    parser.add_argument(
        "--temp_dir_prefix",
        type=str,
        default="oid_staging_",
        help="Prefix for the temporary directory created to stage Open Images downloads before moving to final destination. "
        "A unique ID will be appended to this prefix.",
    )
    parser.add_argument(
        "--output_image_format",
        type=str,
        default="jpg",
        choices=["jpg", "png", "webp"],  # Common web formats
        help="Desired output image format for the saved files. All images will be converted to this format.",
    )

    args = parser.parse_args()

    # Determine classes to download based on the --classes argument
    classes_to_process: List[str] = []
    if args.classes.lower() == "all":
        classes_to_process = CIFAR10_CLASSES
    else:
        # Split the comma-separated string and strip whitespace
        requested_classes = [c.strip() for c in args.classes.split(",")]
        for cls in requested_classes:
            if cls not in CIFAR10_CLASSES:
                print(
                    f"Warning: '{cls}' is not a recognized CIFAR-10 class. Skipping it."
                )
            else:
                classes_to_process.append(cls)

        if not classes_to_process:
            print("Error: No valid classes specified for download. Exiting.")
            exit(1)

    # Load custom OID concept map if provided, otherwise use default
    current_oid_concept_map: Dict[str, str] = DEFAULT_CIFAR10_TO_OID_CONCEPT_MAP.copy()
    if args.oid_map_file:
        try:
            with open(args.oid_map_file, "r") as f:
                custom_map = json.load(f)
                # Update the default map with entries from the custom map
                current_oid_concept_map.update(custom_map)
            print(
                f"Loaded custom OID concept map from '{args.oid_map_file}'. Merged with defaults."
            )
        except FileNotFoundError:
            print(
                f"Error: OID map file not found at '{args.oid_map_file}'. Using default map only."
            )
        except json.JSONDecodeError as e:
            print(
                f"Error parsing OID map file '{args.oid_map_file}': {e}. Please ensure it's valid JSON. Using default map only."
            )
        except Exception as e:
            print(
                f"An unexpected error occurred reading OID map file '{args.oid_map_file}': {e}. Using default map only."
            )

    # Call the download function with parsed arguments
    download_images(
        cifar10_classes=classes_to_process,
        output_dir=args.output_dir,
        limit=args.limit,
        oid_concept_map=current_oid_concept_map,
        skip_existing=args.skip_existing,
        temp_dir_prefix=args.temp_dir_prefix,
        output_image_format=args.output_image_format,
    )

    print("\nScript finished.")
