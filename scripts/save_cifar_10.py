import os
import torchvision
import torchvision.transforms as transforms
from typing import List
from PIL import Image
import argparse  # New import for command-line arguments

"""
This script downloads the original CIFAR-10 dataset (training or test split)
and saves each image as a separate file into a structured directory.
The structure is `output_folder/class_name/image_index.EXT`.
This is typically used for initial data exploration or to create a
local copy of the raw dataset in a human-readable format.
"""


def main() -> None:
    """
    Main function to parse arguments, download the CIFAR-10 dataset,
    and save its images into a structured folder system.
    """
    parser = argparse.ArgumentParser(
        description="Download and save CIFAR-10 images to a structured directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,  # Show default values in help
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="cifar10_raw",
        help="The root directory where structured image folders (class_name/image_index.EXT) will be created.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="./data",
        help="The root directory where the raw CIFAR-10 dataset files (e.g., 'cifar-10-batches-py') will be downloaded by torchvision.",
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Which dataset split to download and process: 'train' (50,000 images) or 'test' (10,000 images).",
    )
    parser.add_argument(
        "--no_download",
        action="store_true",
        help="If set, torchvision will not attempt to download the dataset. It assumes the dataset files already exist at --data_root.",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        default="png",
        choices=["png", "jpeg", "webp"],
        help="The desired image format for the saved files.",
    )
    parser.add_argument(
        "--image_id_padding",
        type=int,
        default=5,
        help="The number of digits to use for zero-padding the image index in the filename (e.g., 5 for 00001.png).",
    )

    args = parser.parse_args()

    output_folder: str = args.output_dir
    data_download_root: str = args.data_root
    is_train_split: bool = args.dataset_split == "train"
    should_download: bool = not args.no_download
    output_image_format: str = args.output_format.lower()
    image_id_padding: int = args.image_id_padding

    print(
        f"Downloading CIFAR-10 {args.dataset_split} dataset to '{data_download_root}' (download={should_download})..."
    )
    transform = (
        transforms.ToTensor()
    )  # Standard transform to get image into tensor format

    try:
        dataset = torchvision.datasets.CIFAR10(
            root=data_download_root,
            train=is_train_split,
            download=should_download,
            transform=transform,
        )
        print("CIFAR-10 dataset loaded.")
    except Exception as e:
        print(f"Error loading CIFAR-10 dataset: {e}")
        if should_download:
            print(
                "Please check your internet connection or try running without --no_download if you believe the dataset is already present."
            )
        else:
            print(
                f"Dataset files not found at '{data_download_root}'. Please ensure they are present or remove --no_download flag to attempt download."
            )
        return

    os.makedirs(output_folder, exist_ok=True)
    print(f"Base output directory '{output_folder}' ensured.")

    class_names: List[str] = dataset.classes

    # Create class-specific subfolders
    for class_name in class_names:
        class_folder: str = os.path.join(output_folder, class_name)
        os.makedirs(class_folder, exist_ok=True)
        print(f"Ensured class folder: {class_folder}")

    print(
        f"Saving {len(dataset)} images to {output_folder} (format: {output_image_format})..."
    )
    for idx, (image_tensor, label) in enumerate(dataset):
        class_name: str = class_names[label]
        class_folder: str = os.path.join(output_folder, class_name)

        # Format the image index with zero-padding
        image_filename: str = f"{idx:0{image_id_padding}d}.{output_image_format}"
        image_path: str = os.path.join(class_folder, image_filename)

        # Convert the PyTorch tensor image back to PIL Image for saving
        image_PIL: Image.Image = transforms.ToPILImage()(image_tensor)

        # Save the image in the specified format
        if output_image_format == "jpeg" or output_image_format == "jpg":
            # For JPEG, save with a quality setting
            image_PIL.save(
                image_path, quality=95
            )  # Default quality is 75, 95 is higher
        else:
            image_PIL.save(image_path)

        if (idx + 1) % 1000 == 0:
            print(f"Saved {idx + 1}/{len(dataset)} images...")

    print(f"All {len(dataset)} CIFAR-10 images saved to {output_folder}.")


if __name__ == "__main__":
    main()
