import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score
import random
import torchvision.transforms.functional as TF
from PIL import Image
import torch.nn.functional as F
import os
import lightning as L

CIFAR10_CLASSES = {
    "airplane": 0,
    "automobile": 1,
    "bird": 2,
    "cat": 3,
    "deer": 4,
    "dog": 5,
    "frog": 6,
    "horse": 7,
    "ship": 8,
    "truck": 9,
}
CIFAR10_CLASSES_REVERSE = {v: k for k, v in CIFAR10_CLASSES.items()}


def set_reproducibility(cfg):
    L.seed_everything(cfg.seed, workers=True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def visualize_feature_maps(model, data_module, return_image=False):
    cat_label = CIFAR10_CLASSES["cat"]
    # Access the underlying dataset (random_split creates a Subset)
    if hasattr(data_module.train_dataset, "dataset"):
        dataset = data_module.train_dataset.dataset
    else:
        dataset = data_module.train_dataset

    # Find indices for samples with the "cat" label.
    cat_indices = [i for i, target in enumerate(dataset.targets) if target == cat_label]
    if not cat_indices:
        pass
    else:
        chosen_index = random.choice(cat_indices)
        sample, label = dataset[chosen_index]

        # Save the original sample image.
        # If sample is a tensor, convert to PIL image.
        if torch.is_tensor(sample):
            sample = sample.cpu()  # Ensure it's on CPU.
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]

            def unnormalize(tensor, mean, std):
                # Create tensors for mean and std and apply unnormalization.
                mean_tensor = torch.tensor(mean, device=tensor.device).view(
                    -1,
                    1,
                    1,
                )
                std_tensor = torch.tensor(std, device=tensor.device).view(-1, 1, 1)
                return tensor * std_tensor + mean_tensor

            sample = unnormalize(sample, mean, std)
            sample = sample.clamp(0, 1)
            # Optionally convert to 8-bit values explicitly:
            sample_img = TF.to_pil_image(sample)
            sample_img_resized = sample_img.resize(
                (512, 512),
                resample=Image.BICUBIC,
            )
        elif isinstance(sample, Image.Image):
            sample_img = sample
        else:
            msg = "Unsupported image type for saving the sample."
            raise TypeError(msg)

        sample_save_path = "sample_image.png"
        sample_img_resized.save(sample_save_path)

        sample = dataset[chosen_index][0].unsqueeze(0).to(model.device)
        # Get feature maps from a designated layer (see method below)
        feature_maps = model.visualize_feature_maps(sample)

        # Plot and save a few feature maps (e.g. first 8 channels)

        if return_image:
            return feature_maps[0, 0].cpu().numpy(), sample_img_resized
        else:
            num_maps = min(feature_maps.shape[1], 8)
            fig, axs = plt.subplots(1, num_maps, figsize=(num_maps * 2, 2))
            for i in range(num_maps):
                fm = feature_maps[0, i].cpu().numpy()
                axs[i].imshow(fm, cmap="viridis")
                axs[i].axis("off")
            feature_maps_save_path = "feature_maps.png"
            plt.savefig(feature_maps_save_path, bbox_inches="tight")
            plt.close()


def visualize_filters(model, return_image=False):
    """Visualize the filters of the first convolutional layer in the model."""
    # Get the first convolutional layer
    first_conv_layer = None
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            first_conv_layer = module
            break

    if first_conv_layer is None:
        print("No convolutional layer found in the model.")
        return

    # Get the weights of the first conv layer
    weights = first_conv_layer.weight.data.cpu()

    # Normalize the weights for better visualization
    min_val = weights.min()
    max_val = weights.max()
    weights = (weights - min_val) / (max_val - min_val)

    # Plot the filters
    n_filters = min(weights.size(0), 64)  # Limit to 64 filters
    grid_size = int(np.ceil(np.sqrt(n_filters)))

    fig, axes = plt.subplots(
        grid_size, grid_size, figsize=(grid_size * 2, grid_size * 2)
    )
    axes = axes.flatten()

    for i in range(grid_size * grid_size):
        if i < n_filters:
            # For RGB filters, convert to displayable image
            filter_img = weights[i]
            if filter_img.size(0) == 3:  # RGB
                img = filter_img.permute(1, 2, 0)  # Convert to HWC format
            else:  # Grayscale
                img = filter_img[0]

            axes[i].imshow(img)
        axes[i].axis("off")

    if return_image:
        return fig
    else:
        plt.suptitle("First Layer Conv Filters")
        plt.tight_layout()
        plt.savefig("conv_filters.png")
        plt.close()
        print("Filter visualization saved to conv_filters.png")
