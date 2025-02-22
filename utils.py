import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score

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
