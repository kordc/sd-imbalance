import os
import torchvision

import torchvision.transforms as transforms

# Define the root folder to save CIFAR-10 images
output_folder = "cifar10_raw"

# Download CIFAR-10 dataset
transform = transforms.ToTensor()
dataset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Class names in CIFAR-10
class_names = dataset.classes

# Create subfolders for each class
for class_name in class_names:
    class_folder = os.path.join(output_folder, class_name)
    os.makedirs(class_folder, exist_ok=True)

# Save images to corresponding class folders
for idx, (image, label) in enumerate(dataset):
    class_name = class_names[label]
    class_folder = os.path.join(output_folder, class_name)
    image_path = os.path.join(class_folder, f"{idx}.png")
    image_PIL = transforms.ToPILImage()(image)
    image_PIL.save(image_path)

print(f"Images saved to {output_folder}")
