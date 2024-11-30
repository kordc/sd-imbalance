import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.data import random_split
from utils import CIFAR10_CLASSES, evaluate_model
from data import DownsampledCIFAR10


def train_resnet18(downsample_class=None, downsample_ratio=1.0, epochs=10, batch_size=64, learning_rate=0.01):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    full_train_dataset = DownsampledCIFAR10(
        root='./data',
        train=True,
        transform=transform,
        download=True,
        downsample_class=downsample_class,
        downsample_ratio=downsample_ratio
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, transform=transform, download=True)

    val_size = int(0.2 * len(full_train_dataset))
    train_size = len(full_train_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_train_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    model = torchvision.models.resnet18(num_classes=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        print(f"Epoch {epoch+1}/{epochs}:")
        for inputs, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        val_accuracy, val_balanced_acc = evaluate_model(
            model, val_loader, device)
        print(
            f"Train Loss: {train_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%, Validation Balanced Accuracy: {val_balanced_acc:.2f}%")

    test_accuracy, test_balanced_acc = evaluate_model(
        model, test_loader, device)
    print(
        f"Test Accuracy: {test_accuracy:.2f}%, Test Balanced Accuracy: {test_balanced_acc:.2f}%")


def main():
    train_resnet18(
        epochs=2, downsample_class=CIFAR10_CLASSES['cat'], downsample_ratio=0.1)


if __name__ == "__main__":
    main()
