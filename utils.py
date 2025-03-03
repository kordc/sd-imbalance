import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score
import random
import torchvision.transforms.functional as TF
from PIL import Image

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


def visualize_feature_maps(model, data_module):
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
        num_maps = min(feature_maps.shape[1], 8)
        fig, axs = plt.subplots(1, num_maps, figsize=(num_maps * 2, 2))
        for i in range(num_maps):
            fm = feature_maps[0, i].cpu().numpy()
            axs[i].imshow(fm, cmap="viridis")
            axs[i].axis("off")
        feature_maps_save_path = "feature_maps.png"
        plt.savefig(feature_maps_save_path, bbox_inches="tight")
        plt.close()
