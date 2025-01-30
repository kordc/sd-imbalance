import os

import lightning as L
import numpy as np
import torch
import torchvision
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from omegaconf import DictConfig
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from utils import CIFAR10_CLASSES


class DownsampledCIFAR10(torchvision.datasets.CIFAR10):
    def __init__(
        self,
        root,
        train=True,
        transform=None,
        download=True,
        downsample_class=None,
        downsample_ratio=1.0,
        naive_oversample=False,
        naive_undersample=False,
        smote=False,
        adasyn=False,
        random_state=42,
    ):
        super().__init__(root=root, train=train, transform=transform, download=download)
        self.downsample_class = downsample_class
        self.downsample_ratio = downsample_ratio
        self.naive_oversample = naive_oversample
        self.naive_undersample = naive_undersample
        self.smote = smote
        self.adasyn = adasyn
        self.random_state = random_state
        if downsample_class is not None:
            self._downsample()

    def _downsample(self):
        targets = np.array(self.targets)
        selected_idx = np.arange(len(targets))
        if self.downsample_class is not None:
            self.downsample_class = CIFAR10_CLASSES[self.downsample_class]

        if self.downsample_class in targets:
            print(f"Downsampling class {self.downsample_class}")
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

            if (
                self.naive_oversample
                + self.naive_undersample
                + self.smote
                + self.adasyn
                > 1
            ):
                raise ValueError(
                    "Only one of naive_oversample, naive_undersample, smote, or adasyn can be True at a time.",
                )

            if self.naive_oversample:
                print(
                    f"Naive oversampling class {self.downsample_class} to original size using RandomOverSampler.",
                )
                ros = RandomOverSampler(sampling_strategy="auto", random_state=42)
                # Reshape data and targets for use with imbalanced-learn
                data_reshaped = self.data.reshape(
                    self.data.shape[0],
                    -1,
                )  # Flatten images
                oversampled_data, oversampled_targets = ros.fit_resample(
                    data_reshaped,
                    self.targets,
                )

                # Reshape oversampled data back to image dimensions (e.g., 32x32x3)
                oversampled_data = oversampled_data.reshape(-1, 32, 32, 3)
                self.data = oversampled_data
                self.targets = list(oversampled_targets)

            elif self.naive_undersample:
                print(
                    f"Naive undersampling all classes to match the size of downsampled class {self.downsample_class}.",
                )

                # Calculate the target number of samples (size of downsampled class)
                target_size = keep_size  # Size of downsampled class

                rus = RandomUnderSampler(
                    sampling_strategy={i: target_size for i in range(10)},
                    random_state=self.random_state,
                )
                # Reshape data and targets for use with imbalanced-learn
                data_reshaped = self.data.reshape(
                    self.data.shape[0],
                    -1,
                )  # Flatten images
                undersampled_data, undersampled_targets = rus.fit_resample(
                    data_reshaped,
                    self.targets,
                )

                # Reshape undersampled data back to image dimensions (e.g., 32x32x3)
                undersampled_data = undersampled_data.reshape(-1, 32, 32, 3)
                self.data = undersampled_data
                self.targets = list(undersampled_targets)

            elif self.smote:
                print(f"Applying SMOTE for oversampling class {self.downsample_class}.")
                smote = SMOTE(sampling_strategy="auto", random_state=self.random_state)
                data_reshaped = self.data.reshape(
                    self.data.shape[0],
                    -1,
                )  # Flatten images
                smote_data, smote_targets = smote.fit_resample(
                    data_reshaped,
                    self.targets,
                )

                # Reshape SMOTE data back to image dimensions (e.g., 32x32x3)
                smote_data = smote_data.reshape(-1, 32, 32, 3)
                self.data = smote_data
                self.targets = list(smote_targets)

            elif self.adasyn:
                print(
                    f"Applying ADASYN for oversampling class {self.downsample_class}.",
                )
                adasyn = ADASYN(
                    sampling_strategy="auto",
                    random_state=self.random_state,
                )
                data_reshaped = self.data.reshape(
                    self.data.shape[0],
                    -1,
                )  # Flatten images
                adasyn_data, adasyn_targets = adasyn.fit_resample(
                    data_reshaped,
                    self.targets,
                )

                # Reshape ADASYN data back to image dimensions (e.g., 32x32x3)
                adasyn_data = adasyn_data.reshape(-1, 32, 32, 3)
                self.data = adasyn_data
                self.targets = list(adasyn_targets)
        else:
            print(
                f"Class {self.downsample_class} not found in the dataset. Skipping downsampling.",
            )


class CIFAR10DataModule(L.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.transform = self.get_augmentations(cfg.augmentations)
        self.test_transform = self.get_augmentations(cfg.test_augmentations)
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def get_augmentations(self, augmentations_cfg):
        transform_list = []
        for aug in augmentations_cfg:
            aug_name = aug["name"]
            params = aug.get("params", {})
            transform_list.append(getattr(transforms, aug_name)(**params))
        return transforms.Compose(transform_list)

    def prepare_data(self):
        cifar10_train_path = os.path.join("./data", "cifar-10-batches-py")
        download_flag = not os.path.exists(cifar10_train_path)
        full_train_dataset = DownsampledCIFAR10(
            root="./data",
            train=True,
            transform=self.transform,
            downsample_class=self.cfg.downsample_class,
            downsample_ratio=self.cfg.downsample_ratio,
            naive_oversample=self.cfg.naive_oversample,
            naive_undersample=self.cfg.naive_undersample,
            smote=self.cfg.smote,
            adasyn=self.cfg.adasyn,
            random_state=self.cfg.seed,
            download=download_flag,
        )
        self.test_dataset = torchvision.datasets.CIFAR10(
            root="./data",
            train=False,
            transform=self.test_transform,
            download=download_flag,
        )

        val_size = int(self.cfg.val_size * len(full_train_dataset))
        train_size = len(full_train_dataset) - val_size
        self.train_dataset, self.val_dataset = random_split(
            full_train_dataset,
            [train_size, val_size],
        )

        train_targets = torch.tensor(
            [full_train_dataset.targets[i] for i in self.train_dataset.indices],
        )

        # Calculate class counts and class weights based on the train dataset
        class_counts = torch.bincount(train_targets)
        class_weights = 1.0 / class_counts.float()
        self.class_weights = class_weights / class_weights.sum()

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            persistent_workers=True,
        )
