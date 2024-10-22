import torch
import torch.optim as optim
import torch.nn as nn
from omegaconf import DictConfig
import hydra
import os
from data import get_dataloader
from model import ResNetModel
from utils import calculate_metrics, save_checkpoint, save_metrics_report


@hydra.main(config_path="configs", config_name="default")
def train(config: DictConfig):
    # Set seeds for reproducibility
    torch.manual_seed(config.train.seed)

    # Initialize data and model
    train_loader = get_dataloader(config, split='train')
    val_loader = get_dataloader(config, split='val')  # Validation DataLoader
    model = ResNetModel(config)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = getattr(optim, config.train.optimizer)(
        model.parameters(), lr=config.train.lr)

    # Variables to store metrics
    metrics = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": []
    }

    # Training loop
    for epoch in range(config.train.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Track loss and accuracy
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        metrics["train_loss"].append(train_loss)
        metrics["train_acc"].append(train_acc)

        print(
            f"Epoch {epoch+1}/{config.train.epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

        # Validation phase every 5 epochs
        if (epoch + 1) % 5 == 0:
            val_loss, val_acc = validate(model, val_loader, criterion)
            metrics["val_loss"].append(val_loss)
            metrics["val_acc"].append(val_acc)
            print(
                f"Validation - Epoch {epoch+1}/{config.train.epochs}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # Final evaluation on validation set
    final_val_loss, final_val_acc = validate(model, val_loader, criterion)
    print(
        f"Final Validation Loss: {final_val_loss:.4f}, Final Validation Accuracy: {final_val_acc:.4f}")

    # Test data
    test_loader = get_dataloader(config, split='test')
    test_loss, test_acc = validate(model, test_loader, criterion)
    metrics["test_loss"] = test_loss
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    # Save model
    save_checkpoint(model.state_dict(),
                    filename=f'{config.model.name}_final.pth')

    # Save metrics report
    save_metrics_report(
        metrics, filename=f'{config.model.name}_metrics_report.txt')


def validate(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss = running_loss / len(dataloader)
    val_acc = correct / total
    return val_loss, val_acc


if __name__ == "__main__":
    train()
