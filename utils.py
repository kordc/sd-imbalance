import torch


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
