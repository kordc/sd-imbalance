import numpy as np
import torchvision
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
import lightning as L
from omegaconf import DictConfig


class DownsampledCIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, download=False, downsample_class=None, downsample_ratio=1.0):
        super().__init__(root=root, train=train, transform=transform, download=download)
        self.downsample_class = downsample_class
        self.downsample_ratio = downsample_ratio
        if downsample_class is not None:
            self._downsample()

    def _downsample(self):
        targets = np.array(self.targets)
        selected_idx = np.arange(len(targets))

        # Check if the specified class exists in the dataset
        if self.downsample_class in targets:
            # Get indices of all samples of the target class
            class_idx = selected_idx[targets == self.downsample_class]
            # Number of samples to keep for the target class
            keep_size = int(len(class_idx) * self.downsample_ratio)
            # Randomly sample indices to keep
            keep_idx = np.random.choice(
                class_idx, size=keep_size, replace=False)
            # Combine with indices of all other classes
            non_class_idx = selected_idx[targets != self.downsample_class]
            final_idx = np.concatenate([non_class_idx, keep_idx])
            # Update the dataset
            self.data = self.data[final_idx]
            self.targets = list(targets[final_idx])
        else:
            print(
                f"Class {self.downsample_class} not found in the dataset. Skipping downsampling.")

class CIFAR10DataModule(L.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.transform = self.get_augmentations(cfg.augmentations)
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def get_augmentations(self, augmentations_cfg):
        transform_list = []
        for aug in augmentations_cfg:
            aug_name = aug['name']
            params = aug.get('params', {})
            transform_list.append(getattr(transforms, aug_name)(**params))
        return transforms.Compose(transform_list)

    def prepare_data(self):
        DownsampledCIFAR10(root="./data", train=True, download=True)
        torchvision.datasets.CIFAR10(root="./data", train=False, download=True)

    def setup(self, stage=None):
        full_train_dataset = DownsampledCIFAR10(
            root="./data",
            train=True,
            transform=self.transform,
            downsample_class=self.cfg.downsample_class,
            downsample_ratio=self.cfg.downsample_ratio,
        )
        self.test_dataset = torchvision.datasets.CIFAR10(
            root="./data", train=False, transform=self.transform
        )

        val_size = int(self.cfg.val_size * len(full_train_dataset))
        train_size = len(full_train_dataset) - val_size
        self.train_dataset, self.val_dataset = random_split(
            full_train_dataset, [train_size, val_size]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.cfg.batch_size, 
            shuffle=True, 
            num_workers=self.cfg.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.cfg.batch_size, 
            shuffle=False, 
            num_workers=self.cfg.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=self.cfg.batch_size, 
            shuffle=False, 
            num_workers=self.cfg.num_workers
        )