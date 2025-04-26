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


def evaluate_model(model, dataloader, device):
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    accuracy = (
        100 * (np.array(all_preds) == np.array(all_labels)).sum() / len(all_labels)
    )
    balanced_acc = balanced_accuracy_score(all_labels, all_preds) * 100
    return accuracy, balanced_acc


def plot_metrics(log_dir) -> None:
    # Load logged metrics from CSV
    metrics_file = f"{log_dir}/metrics.csv"
    metrics = pd.read_csv(metrics_file)

    plt.figure(figsize=(12, 6))

    if "train_accuracy_epoch" in metrics.columns and "val_accuracy" in metrics.columns:
        plt.subplot(1, 2, 1)
        plt.plot(
            metrics["epoch"],
            metrics["train_accuracy_epoch"],
            label="Train Accuracy",
            marker="o",
            linestyle="-",
        )
        plt.plot(
            metrics["epoch"],
            metrics["val_accuracy"],
            label="Validation Accuracy",
            marker="o",
            linestyle="-",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Train vs Validation Accuracy")
        plt.legend()
        plt.grid()

    if "train_loss_epoch" in metrics.columns and "val_loss" in metrics.columns:
        plt.subplot(1, 2, 2)
        plt.plot(
            metrics["epoch"],
            metrics["train_loss_epoch"],
            label="Train Loss",
            marker="o",
            linestyle="-",
        )
        plt.plot(
            metrics["epoch"],
            metrics["val_loss"],
            label="Validation Loss",
            marker="o",
            linestyle="-",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Train vs Validation Loss")
        plt.legend()
        plt.grid()

    plt.tight_layout()
    plt.savefig("train_plot.png")
    plt.close()


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


def apply_gradcam(
    model, data_module, target_layer=None, num_samples=5, return_image=False
):
    """
    Apply GradCAM to visualize which parts of input images the model focuses on.

    Args:
        model: The trained model
        data_module: Data module containing the dataset
        target_layer: The layer to use for GradCAM. If None, use the last conv layer.
        num_samples: Number of samples to visualize
        return_image: If True, return the figure instead of saving it

    Returns:
        If return_image is True, returns the matplotlib figure; otherwise None.
    """
    # Move model to evaluation mode
    model.eval()

    # Find target layer if not specified
    if target_layer is None:
        # Find the last convolutional layer
        for module in reversed(list(model.modules())):
            if isinstance(module, torch.nn.Conv2d):
                target_layer = module
                break

    if target_layer is None:
        print("No convolutional layer found for GradCAM.")
        return None

    # Get samples from test set
    test_loader = data_module.test_dataloader()
    batch = next(iter(test_loader))
    images, labels = batch

    # Limit to num_samples
    images = images[:num_samples]
    labels = labels[:num_samples]

    # Register hooks to get activations and gradients
    activations = {}
    gradients = {}

    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()

        return hook

    def get_gradient(name):
        def hook(grad):
            gradients[name] = grad.detach()

        return hook

    # Register hooks
    handle_act = target_layer.register_forward_hook(get_activation("target"))

    # Create figure for results
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))

    for i in range(num_samples):
        img = images[i : i + 1].to(model.device)
        label = labels[i].item()
        class_name = CIFAR10_CLASSES_REVERSE[label]

        # Forward pass
        output = model(img)
        pred_score, pred_class = torch.max(output, 1)
        pred_class_name = CIFAR10_CLASSES_REVERSE[pred_class.item()]

        # Register backward hook
        activations["target"].register_hook(get_gradient("target"))

        # Backprop
        model.zero_grad()
        output[0, pred_class].backward()

        # Get activations and gradients
        act = activations["target"]
        grad = gradients["target"]

        # Global average pooling of gradients
        weights = torch.mean(grad, dim=(2, 3), keepdim=True)

        # Weighted sum of activation maps
        gradcam = torch.sum(weights * act, dim=1, keepdim=True)

        # ReLU and normalize
        gradcam = torch.relu(gradcam)
        gradcam = F.interpolate(
            gradcam,
            size=(32, 32),  # CIFAR10 image size
            mode="bilinear",
            align_corners=False,
        )

        # Normalize between 0 and 1
        gradcam_min, gradcam_max = gradcam.min(), gradcam.max()
        gradcam = (gradcam - gradcam_min) / (gradcam_max - gradcam_min + 1e-8)

        # Convert to numpy for visualization
        img_np = img[0].permute(1, 2, 0).cpu().numpy()
        img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array(
            [0.485, 0.456, 0.406]
        )
        img_np = np.clip(img_np, 0, 1)

        gradcam_np = gradcam[0, 0].cpu().numpy()

        # Create heatmap
        heatmap = plt.cm.jet(gradcam_np)[:, :, :3]  # RGBA -> RGB
        superimposed = 0.6 * heatmap + 0.4 * img_np

        # Plot original image
        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title(f"Original: {class_name}")
        axes[i, 0].axis("off")

        # Plot GradCAM
        axes[i, 1].imshow(heatmap)
        axes[i, 1].set_title("GradCAM")
        axes[i, 1].axis("off")

        # Plot superimposed
        axes[i, 2].imshow(superimposed)
        axes[i, 2].set_title(f"Prediction: {pred_class_name}")
        axes[i, 2].axis("off")

    plt.tight_layout()

    if return_image:
        # Clean up hooks before returning
        handle_act.remove()
        return fig
    else:
        # Save and close
        plt.savefig("gradcam_results.png")
        plt.close()
        print("GradCAM results saved to gradcam_results.png")
        # Clean up hooks
        handle_act.remove()
        return None
