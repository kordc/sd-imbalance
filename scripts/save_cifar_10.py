import os
import torchvision
import torchvision.transforms as transforms
from typing import List
from PIL import Image

"""
This script downloads the original CIFAR-10 training dataset and saves
each image as a separate PNG file into a structured directory.
The structure is `output_folder/class_name/image_index.png`.
This is typically used for initial data exploration or to create a
local copy of the raw dataset, NOT for images intended to be added
as 'extra_images' by `data.py` (which expects a flat structure with
class names in filenames).
"""


def main() -> None:
    """
    Downloads the CIFAR-10 training dataset and saves its images
    into a structured folder system (class_name/image_index.png).
    """
    output_folder: str = "cifar10_raw"

    print("Downloading CIFAR-10 training dataset to ./data if not exists...")
    transform = transforms.ToTensor()
    dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    print("CIFAR-10 dataset downloaded.")

    os.makedirs(output_folder, exist_ok=True)
    print(f"Output directory '{output_folder}' ensured.")

    class_names: List[str] = dataset.classes

    for class_name in class_names:
        class_folder: str = os.path.join(output_folder, class_name)
        os.makedirs(class_folder, exist_ok=True)
        print(f"Created class folder: {class_folder}")

    print(f"Saving images to {output_folder}...")
    for idx, (image_tensor, label) in enumerate(dataset):
        class_name: str = class_names[label]
        class_folder: str = os.path.join(output_folder, class_name)

        image_path: str = os.path.join(class_folder, f"{idx:05d}.png")

        image_PIL: Image.Image = transforms.ToPILImage()(image_tensor)
        image_PIL.save(image_path)

        if (idx + 1) % 1000 == 0:
            print(f"Saved {idx + 1} images...")

    print(f"All CIFAR-10 images saved to {output_folder}.")


if __name__ == "__main__":
    main()
