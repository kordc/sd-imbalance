import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
import hydra
from omegaconf import DictConfig

class CustomDataset:
    def __init__(self, config: DictConfig):
        self.config = config

        # Define transformations (can add more)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        # Load dataset (CIFAR-10 as example)
        if self.config.dataset.name == 'cifar10':
            self.dataset = datasets.CIFAR10(root=self.config.dataset.path, train=True, download=True, transform=transform)
        else:
            raise ValueError(f"Unsupported dataset {self.config.dataset.name}")

        # Apply undersampling
        if self.config.dataset.undersample:
            self.dataset = self.undersample_classes(self.dataset, self.config.dataset.undersample)

    def undersample_classes(self, dataset, undersample_config):
        class_indices = {i: [] for i in range(self.config.dataset.num_classes)}

        for idx, (_, label) in enumerate(dataset):
            class_indices[label].append(idx)

        undersampled_indices = []
        for cls in undersample_config.classes:
            n_samples = int(len(class_indices[cls]) * (undersample_config.percentage / 100))
            undersampled_indices.extend(class_indices[cls][:n_samples])

        return Subset(dataset, undersampled_indices)

    def get_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.config.train.batch_size, shuffle=True)

@hydra.main(config_path="configs", config_name="default")
def get_dataloader(config: DictConfig):
    dataset = CustomDataset(config)
    return dataset.get_dataloader()
