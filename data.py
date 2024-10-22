import random  # Add random module for shuffling
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torchvision import datasets, transforms
import hydra
from omegaconf import DictConfig


class CustomDataset:
    def __init__(self, config: DictConfig):
        self.config = config

        # Set the random seed for reproducibility
        random.seed(self.config.train.seed)

        # Define transformations
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        # Load dataset (CIFAR-10 as example)
        if self.config.dataset.name == 'cifar10':
            self.dataset = datasets.CIFAR10(
                root=self.config.dataset.path, train=True, download=True, transform=transform)
        else:
            raise ValueError(f"Unsupported dataset {self.config.dataset.name}")

        # Apply undersampling
        if self.config.dataset.undersample:
            self.dataset = self.undersample_classes(
                self.dataset, self.config.dataset.undersample)

        # Split into train and validation sets (80/20 split as default)
        train_size = int(0.8 * len(self.dataset))
        val_size = len(self.dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(
            self.dataset, [train_size, val_size])

        self.test_set = datasets.CIFAR10(
            root=self.config.dataset.path, train=False, download=True, transform=transform)

    def undersample_classes(self, dataset, undersample_config):
        class_indices = {i: [] for i in range(self.config.dataset.num_classes)}

        # Collect indices for each class
        for idx, (_, label) in enumerate(dataset):
            class_indices[label].append(idx)

        undersampled_indices = []
        for cls in undersample_config.classes:
            # Shuffle class indices to ensure randomness
            random.shuffle(class_indices[cls])

            # Select a subset of samples according to the percentage
            n_samples = int(
                len(class_indices[cls]) * (undersample_config.percentage / 100))
            undersampled_indices.extend(class_indices[cls][:n_samples])

        return Subset(dataset, undersampled_indices)

    def get_dataloader(self, split='train'):
        if split == 'train':
            return DataLoader(self.train_dataset, batch_size=self.config.train.batch_size, shuffle=True)
        elif split == 'val':
            return DataLoader(self.val_dataset, batch_size=self.config.train.batch_size, shuffle=False)
        elif split == 'test':
            return DataLoader(self.test_set, batch_size=self.config.train.batch_size, shuffle=False)
        else:
            raise ValueError(f"Unsupported split {split}")


@hydra.main(config_path="configs", config_name="default")
def get_dataloader(config: DictConfig, split='train'):
    dataset = CustomDataset(config)
    return dataset.get_dataloader(split=split)
