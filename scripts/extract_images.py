import os
from data import DownsampledCIFAR10
from utils import (
    CIFAR10_CLASSES,
)  # Assumes CIFAR10_CLASSES maps "cat" to its label (e.g., 3)
from PIL import Image


def main():
    # Configure dataset parameters.
    # For CIFAR10, there are 5,000 cat images in the training set.
    # To keep 50 cat images, we use a downsample ratio of 50/5000 = 0.01.
    dataset = DownsampledCIFAR10(
        root="./data",
        train=True,
        transform=None,  # No transform needed for extraction.
        download=True,
        downsample_class="cat",
        downsample_ratio=0.01,  # This will keep approximately 50 cat images.
        naive_oversample=False,
        naive_undersample=False,
        smote=False,
        adasyn=False,
        random_state=42,
        add_extra_images=False,
        extra_images_dir="",
        max_extra_images=None,
    )

    # After downsampling, the dataset still contains non-cat images too.
    # We filter out only those images that belong to the cat class.
    cat_label = CIFAR10_CLASSES["cat"]  # For example, 3.
    cat_indices = [i for i, label in enumerate(dataset.targets) if label == cat_label]

    print(f"Found {len(cat_indices)} cat images in the downsampled dataset.")

    # Create output directory for cat images.
    output_dir = "cats"
    os.makedirs(output_dir, exist_ok=True)

    # Save each cat image as a JPEG file after resizing to 512x512 pixels.
    for count, idx in enumerate(cat_indices):
        # dataset.data is a numpy array of shape (N, H, W, C) (e.g., (32, 32, 3))
        img_array = dataset.data[idx]
        img = Image.fromarray(img_array)

        # Resize image to 512x512 pixels using high-quality resampling.
        img_resized = img.resize((512, 512), Image.LANCZOS)

        save_path = os.path.join(output_dir, f"cat_{count + 1}.jpg")
        img_resized.save(save_path, format="JPEG")
        print(f"Saved {save_path}")


if __name__ == "__main__":
    main()
