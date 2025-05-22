# utils:
import matplotlib.pyplot as plt
import numpy as np
import torch
import random
import torchvision.transforms.functional as TF
from PIL import Image
import os
import lightning as L
from omegaconf import DictConfig
from typing import Tuple, Optional

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


def set_reproducibility(cfg: DictConfig) -> None:
    """
    Sets the random seed for reproducibility across different libraries and environments.

    Args:
        cfg (DictConfig): A Hydra configuration object containing the 'seed' value.
    """
    L.seed_everything(cfg.seed, workers=True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def visualize_feature_maps(
    model: L.LightningModule,
    data_module: L.LightningDataModule,
    return_image: bool = False,
) -> Optional[Tuple[np.ndarray, Image.Image]]:
    """
    Visualizes feature maps for a randomly selected "cat" image from the dataset.

    It fetches a "cat" image, saves its resized version, passes it through
    the model's feature extraction layers, and optionally returns the feature map
    and the resized image as numpy arrays. If `return_image` is False,
    it saves the feature maps to a file.

    Args:
        model (L.LightningModule): The trained Lightning model with a
                                   `visualize_feature_maps` method.
        data_module (L.LightningDataModule): The data module containing the dataset.
        return_image (bool): If True, returns the feature map and resized image.
                             If False, saves them to files. Defaults to False.

    Returns:
        Optional[Tuple[np.ndarray, Image.Image]]: A tuple containing the numpy
        array of the first feature map and the PIL Image of the resized sample,
        if `return_image` is True. Returns None otherwise.
    """
    cat_label = CIFAR10_CLASSES["cat"]

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

            def unnormalize(
                tensor: torch.Tensor, mean: list[float], std: list[float]
            ) -> torch.Tensor:
                """Unnormalizes a tensor image."""
                mean_tensor = torch.tensor(mean, device=tensor.device).view(
                    -1,
                    1,
                    1,
                )
                std_tensor = torch.tensor(std, device=tensor.device).view(-1, 1, 1)
                return tensor * std_tensor + mean_tensor

            sample = unnormalize(sample, mean, std)
            sample = sample.clamp(0, 1)

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
            _, axs = plt.subplots(1, num_maps, figsize=(num_maps * 2, 2))
            for i in range(num_maps):
                fm = feature_maps[0, i].cpu().numpy()
                axs[i].imshow(fm, cmap="viridis")
                axs[i].axis("off")
            feature_maps_save_path = "feature_maps.png"
            plt.savefig(feature_maps_save_path, bbox_inches="tight")
            plt.close()
    return None


def visualize_filters(
    model: L.LightningModule, return_image: bool = False
) -> Optional[plt.Figure]:
    """
    Visualizes the filters of the first convolutional layer in the model.

    Args:
        model (L.LightningModule): The Lightning model containing convolutional layers.
        return_image (bool): If True, returns the Matplotlib Figure object.
                             If False, saves the figure to a file. Defaults to False.

    Returns:
        Optional[plt.Figure]: The Matplotlib Figure object if `return_image` is True.
                              Returns None otherwise.
    """
    # Get the first convolutional layer
    first_conv_layer = None
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            first_conv_layer = module
            break

    if first_conv_layer is None:
        print("No convolutional layer found in the model.")
        return None

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
    return None


def prepare_fine_tune(cfg: DictConfig) -> None:
    """
    Resets values needed for fine-tuning

    Args:
        cfg (DictConfig): A Hydra configuration object
    """
    L.seed_everything(cfg.seed, workers=True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    cfg.naive_oversample = False
    cfg.naive_undersample = False
    cfg.smote = False
    cfg.adasyn = False
    cfg.label_smoothing = False
    cfg.class_weighting = False
    cfg.epochs = 100
    cfg.add_extra_images = False
    for class_name in CIFAR10_CLASSES:
        cfg.downsample_classes[class_name] = 0.1
        cfg.extra_images_per_class[class_name] = 0
    cfg.dynamic_upsample = False
    cfg.cutmix_or_mixup = False
    cfg.name += "_fine_tuned"
    cfg.naive_undersample = True
