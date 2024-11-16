import torch
import matplotlib.pyplot as plt


def calculate_metrics(outputs, labels):
    _, predicted = torch.max(outputs.data, 1)
    total = labels.size(0)
    correct = (predicted == labels).sum().item()
    accuracy = correct / total
    return accuracy


def save_checkpoint(state, filename='checkpoint.pth'):
    torch.save(state, filename)


def save_metrics_report(metrics, filename='metrics_report.txt'):
    with open(filename, 'w') as f:
        f.write("Epoch,Train Loss,Train Acc,Val Loss,Val Acc\n")
        for i in range(len(metrics["train_loss"])):
            val_loss = metrics["val_loss"][i] if i < len(
                metrics["val_loss"]) else "N/A"
            val_acc = metrics["val_acc"][i] if i < len(
                metrics["val_acc"]) else "N/A"
            f.write(
                f"{i+1},{metrics['train_loss'][i]:.4f},{metrics['train_acc'][i]:.4f},{val_loss},{val_acc}\n")


def plot_metrics(metrics, model_name):
    epochs = range(1, len(metrics["train_loss"]) + 1)

    plt.figure(figsize=(12, 4))

    # Plot training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, metrics["train_loss"], label='Train Loss')
    plt.plot(epochs, metrics["val_loss"], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Plot training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, metrics["train_acc"], label='Train Accuracy')
    plt.plot(epochs, metrics["val_acc"], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.suptitle(f'Metrics for {model_name}')
    plt.savefig(f'{model_name}_metrics.png')
    plt.show()
