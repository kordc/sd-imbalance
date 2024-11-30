import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

# Modify CIFAR-10 Dataset to allow downsampling
class DownsampledCIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, download=False, downsample_class=None, downsample_ratio=1.0):
        super().__init__(root=root, train=train, transform=transform, download=download)
        self.downsample_class = downsample_class
        self.downsample_ratio = downsample_ratio
        if downsample_class is not None:
            self._downsample()

    def _downsample(self):
        # Convert targets to a NumPy array for easier manipulation
        targets = np.array(self.targets)
        selected_idx = np.arange(len(targets))
        
        # Check if the specified class exists in the dataset
        if self.downsample_class in targets:
            # Get indices of all samples of the target class
            class_idx = selected_idx[targets == self.downsample_class]
            # Number of samples to keep for the target class
            keep_size = int(len(class_idx) * self.downsample_ratio)
            # Randomly sample indices to keep
            keep_idx = np.random.choice(class_idx, size=keep_size, replace=False)
            # Combine with indices of all other classes
            non_class_idx = selected_idx[targets != self.downsample_class]
            final_idx = np.concatenate([non_class_idx, keep_idx])
            # Update the dataset
            self.data = self.data[final_idx]
            self.targets = list(targets[final_idx])
        else:
            print(f"Class {self.downsample_class} not found in the dataset. Skipping downsampling.")

# Function to compute accuracy and balanced accuracy
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

    # Compute accuracy
    accuracy = 100 * (np.array(all_preds) == np.array(all_labels)).sum() / len(all_labels)
    # Compute balanced accuracy
    balanced_acc = balanced_accuracy_score(all_labels, all_preds) * 100
    return accuracy, balanced_acc

# Define the ResNet18 Training Pipeline
def train_resnet18(downsample_class=None, downsample_ratio=1.0, epochs=10, batch_size=64, learning_rate=0.01):
    # Data transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load the dataset
    full_train_dataset = DownsampledCIFAR10(
        root='./data',
        train=True,
        transform=transform,
        download=True,
        downsample_class=downsample_class,
        downsample_ratio=downsample_ratio
    )
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)

    # Split into training and validation sets (80/20 split)
    val_size = int(0.2 * len(full_train_dataset))
    train_size = len(full_train_dataset) - val_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Model, Loss, Optimizer
    model = torchvision.models.resnet18(num_classes=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # Training loop
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
        val_accuracy, val_balanced_acc = evaluate_model(model, val_loader, device)
        print(f"Train Loss: {train_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%, Validation Balanced Accuracy: {val_balanced_acc:.2f}%")

    # Final evaluation on test set
    test_accuracy, test_balanced_acc = evaluate_model(model, test_loader, device)
    print(f"Test Accuracy: {test_accuracy:.2f}%, Test Balanced Accuracy: {test_balanced_acc:.2f}%")

if __name__ == "__main__":
    reversed_cifar_class_mapping = {
        'airplane': 0,
        'automobile': 1,
        'bird': 2,
        'cat': 3,
        'deer': 4,
        'dog': 5,
        'frog': 6,
        'horse': 7,
        'ship': 8,
        'truck': 9
    }

    train_resnet18(epochs=2, downsample_class=reversed_cifar_class_mapping['cat'], downsample_ratio=0.1)

    train_resnet18(epochs=2)