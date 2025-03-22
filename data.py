import os
import sys
from glob import glob

import lightning as L
import numpy as np
import torch
import torchvision
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from omegaconf import DictConfig
from PIL import Image
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

# from scripts.flux_redux_augment import FluxReduxAugment
from utils import CIFAR10_CLASSES


class DownsampledCIFAR10(torchvision.datasets.CIFAR10):
    def __init__(
        self,
        root,
        train=True,
        transform=None,
        download=True,
        downsample_class=None,  # Single class or None
        downsample_ratio=1.0,   # Single ratio
        downsample_classes=None,  # Dict of class_name:ratio pairs
        naive_oversample=False,
        naive_undersample=False,
        smote=False,
        adasyn=False,
        random_state=42,
        add_extra_images=False,
        extra_images_dir="extra-images",
        max_extra_images=None,  # Single limit
        extra_images_per_class=None,  # Dict of class_name:count pairs
        keep_only_cat=False,
    ) -> None:
        super().__init__(root=root, train=train, transform=transform, download=download)
        
        # Backward compatibility
        self.downsample_class = downsample_class
        self.downsample_ratio = downsample_ratio
        
        # New multi-class configuration
        self.downsample_classes = downsample_classes or {}
        
        # If single downsample_class and ratio are provided, convert to downsample_classes dict
        if self.downsample_class is not None and not self.downsample_classes:
            self.downsample_classes = {self.downsample_class: self.downsample_ratio}
            
        self.naive_oversample = naive_oversample
        self.naive_undersample = naive_undersample
        self.smote = smote
        self.adasyn = adasyn
        self.random_state = random_state

        # Image addition parameters
        self.add_extra_images = add_extra_images
        self.extra_images_dir = extra_images_dir
        self.max_extra_images = max_extra_images
        self.extra_images_per_class = extra_images_per_class or {}
        
        self.keep_only_cat = keep_only_cat
        self._extra_images_added = False  # Flag to avoid adding twice

        # Apply transformations
        if self.downsample_classes:
            self._downsample_multiple()
        
        if self.add_extra_images and not self._extra_images_added:
            self._add_extra_images()
            self._extra_images_added = True

        # After modifying self.data, update normalization in the transform:
        self.update_normalization()

    def update_normalization(self):
        """Update the normalization parameters for the transforms after data changes."""
        if self.transform is None:
            return
            
        # Calculate new mean and std based on the modified dataset
        data = self.data.astype(np.float32) / 255.0
        mean = np.mean(data, axis=(0, 1, 2))
        std = np.std(data, axis=(0, 1, 2))
        
        # Look for Normalize transform in the composed transforms
        if isinstance(self.transform, transforms.Compose):
            for i, transform in enumerate(self.transform.transforms):
                if isinstance(transform, transforms.Normalize):
                    # Replace with updated normalization values
                    self.transform.transforms[i] = transforms.Normalize(
                        mean=mean.tolist(),
                        std=std.tolist(),
                    )
                    break

    def get_new_std_mean(self):
        """Return the new mean and std of the dataset after modifications."""
        data = self.data.astype(np.float32) / 255.0
        mean = np.mean(data, axis=(0, 1, 2))
        std = np.std(data, axis=(0, 1, 2))
        return mean.tolist(), std.tolist()

    def _downsample_multiple(self) -> None:
        """Downsample multiple classes according to their specified ratios."""
        targets = np.array(self.targets)
        selected_idx = np.arange(len(targets))
        keep_indices = []
        
        # Convert class names to IDs if needed
        class_id_ratios = {}
        for class_name, ratio in self.downsample_classes.items():
            if isinstance(class_name, str):
                class_id = CIFAR10_CLASSES.get(class_name)
                if class_id is not None:
                    class_id_ratios[class_id] = ratio
            else:
                class_id_ratios[class_name] = ratio
                
        # For backward compatibility
        if self.downsample_class is not None:
            if isinstance(self.downsample_class, str):
                self.downsample_class = CIFAR10_CLASSES.get(self.downsample_class)
        
        # Process each class to downsample
        for class_id, ratio in class_id_ratios.items():
            if class_id in np.unique(targets):
                # Get indices of all samples of this class
                class_indices = selected_idx[targets == class_id]
                # Number of samples to keep
                keep_size = int(len(class_indices) * ratio)
                # Randomly sample indices to keep
                keep_idx = np.random.choice(class_indices, size=keep_size, replace=False)
                keep_indices.append(keep_idx)
        
        # Get indices of classes not being downsampled
        downsampled_classes = list(class_id_ratios.keys())
        non_downsampled_indices = selected_idx[~np.isin(targets, downsampled_classes)]
        
        # Combine all indices
        if keep_indices:
            all_keep_indices = np.concatenate([non_downsampled_indices] + keep_indices)
            
            # Update the dataset
            self.data = self.data[all_keep_indices]
            self.targets = list(targets[all_keep_indices])
            
            # Apply resampling methods if selected
            self._apply_resampling()
        
        # For backward compatibility with keep_only_cat
        if self.keep_only_cat and self.downsample_class is not None:
            updated_targets = np.array(self.targets)
            mask = updated_targets == self.downsample_class
            self.data = self.data[mask]
            self.targets = [self.downsample_class] * len(self.data)

    def _apply_resampling(self):
        """Apply resampling methods (SMOTE, ADASYN, etc.) after downsampling."""
        if (self.naive_oversample + self.naive_undersample + self.smote + self.adasyn > 1):
            raise ValueError("Only one of naive_oversample, naive_undersample, smote, or adasyn can be True at a time.")
        
        if not any([self.naive_oversample, self.naive_undersample, self.smote, self.adasyn]):
            return
            
        # Reshape for imbalanced-learn
        data_reshaped = self.data.reshape(self.data.shape[0], -1)
        
        if self.naive_oversample:
            ros = RandomOverSampler(sampling_strategy="auto", random_state=self.random_state)
            resampled_data, resampled_targets = ros.fit_resample(data_reshaped, self.targets)
        elif self.naive_undersample:
            # If we have multiple classes with different ratios, calculate the target size
            # based on the smallest downsampled class
            if self.downsample_classes:
                targets = np.array(self.targets)
                target_sizes = {}
                for class_id in np.unique(targets):
                    target_sizes[class_id] = len(targets[targets == class_id])
                rus = RandomUnderSampler(
                    sampling_strategy=target_sizes,
                    random_state=self.random_state
                )
            else:
                rus = RandomUnderSampler(sampling_strategy="auto", random_state=self.random_state)
            resampled_data, resampled_targets = rus.fit_resample(data_reshaped, self.targets)
        elif self.smote:
            smote = SMOTE(sampling_strategy="auto", random_state=self.random_state)
            resampled_data, resampled_targets = smote.fit_resample(data_reshaped, self.targets)
        elif self.adasyn:
            adasyn = ADASYN(sampling_strategy="auto", random_state=self.random_state)
            resampled_data, resampled_targets = adasyn.fit_resample(data_reshaped, self.targets)
        
        # Reshape back to images
        resampled_data = resampled_data.reshape(-1, 32, 32, 3)
        self.data = resampled_data
        self.targets = list(resampled_targets)

    def _add_extra_images(self) -> None:
        """
        Adds extra images from a directory into the training data.
        Images should be named as CLASS_idx.png/jpg (e.g., cat_1.png, airplane_42.jpg).
        Can be configured to add specific numbers of images per class.
        """
        image_files = glob(os.path.join(self.extra_images_dir, "*.*"))
        if not image_files:
            print(f"No images found in {self.extra_images_dir}")
            return

        # Group files by class
        class_to_files = {}
        for fpath in image_files:
            filename = os.path.basename(fpath)
            try:
                class_name = filename.split('_')[0]
                if class_name in CIFAR10_CLASSES:
                    if class_name not in class_to_files:
                        class_to_files[class_name] = []
                    class_to_files[class_name].append(fpath)
            except:
                print(f"Skipping file with unexpected format: {filename}")
                
        # Process each class according to configuration
        new_images_list = []
        target_classes = []
        
        for class_name, files in class_to_files.items():
            class_idx = CIFAR10_CLASSES[class_name]
            
            # Skip if this class should be filtered out (for backward compatibility)
            if (self.downsample_class is not None and 
                class_idx != self.downsample_class and 
                not self.extra_images_per_class):
                continue
            
            # Determine how many images to use for this class
            num_to_use = None
            
            # First priority: extra_images_per_class if specified for this class
            if self.extra_images_per_class and class_name in self.extra_images_per_class:
                num_to_use = min(self.extra_images_per_class[class_name], len(files))
                print(f"Adding {num_to_use} images for class '{class_name}' (per-class config)")
                
            # Second priority: max_extra_images if specified and we're only using one class
            elif self.max_extra_images is not None and self.downsample_class is not None:
                num_to_use = min(self.max_extra_images, len(files))
                print(f"Adding {num_to_use} images for class '{class_name}' (global limit)")
                
            # Otherwise use all files
            else:
                num_to_use = len(files)
                print(f"Adding all {num_to_use} images for class '{class_name}'")
                
            # Shuffle and select files
            selected_files = files
            if num_to_use < len(files):
                np.random.shuffle(files)
                selected_files = files[:num_to_use]
                
            # Load and process images
            for fpath in selected_files:
                try:
                    with Image.open(fpath) as img:
                        img = img.convert("RGB")
                        img = img.resize((32, 32))
                        arr = np.array(img)
                        new_images_list.append(arr)
                        target_classes.append(class_idx)
                except Exception as e:
                    print(f"Error processing {fpath}: {e}")
                    
        if not new_images_list:
            print("No valid images found to add")
            return
            
        # Stack them into shape (N, 32, 32, 3)
        new_images = np.stack(new_images_list, axis=0)

        # Append to existing dataset data
        self.data = np.concatenate([self.data, new_images], axis=0)

        # Extend targets with appropriate class indices
        self.targets.extend(target_classes)
        
        print(f"Added a total of {len(new_images_list)} new images to the dataset")

    # Keep original methods
    def _downsample(self) -> None:
        # Redirect to new method for backward compatibility
        self._downsample_multiple()


class CIFAR10DataModule(L.LightningDataModule):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.transform = self.get_augmentations(cfg.augmentations)
        self.test_transform = self.get_augmentations(cfg.test_augmentations)
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def get_augmentations(self, augmentations_cfg):
        transform_list = []
        # Mapping for custom transforms.
        custom_transforms = {
            "FluxReduxAugment": FluxReduxAugment
            if "FluxReduxAugment" in augmentations_cfg
            else None
        }  # noqa: F821
        for aug in augmentations_cfg:
            aug_name = aug["name"]
            params = aug.get("params", {})
            if aug_name in custom_transforms:
                transform_list.append(custom_transforms[aug_name](**params))
            elif "resize" in aug_name.lower():
                interpolation_dict = {
                    "nearest": Image.NEAREST,
                    "bilinear": Image.BILINEAR,
                    "bicubic": Image.BICUBIC,
                    "lanczos": Image.LANCZOS,
                }
                params["interpolation"] = interpolation_dict[params["interpolation"]]
                transform_list.append(getattr(transforms, aug_name)(**params))
            else:
                transform_list.append(getattr(transforms, aug_name)(**params))
        return transforms.Compose(transform_list)

    def setup(self, stage):
        if stage == "fit" or stage is None:
            return
    
    def prepare_data(self) -> None:
        cifar10_train_path = os.path.join("./data", "cifar-10-batches-py")
        download_flag = not os.path.exists(cifar10_train_path)
        
        # Get multi-class downsampling settings if available
        downsample_classes = getattr(self.cfg, "downsample_classes", {})
        extra_images_per_class = getattr(self.cfg, "extra_images_per_class", {})
        
        full_train_dataset = DownsampledCIFAR10(
            root="./data",
            train=True,
            transform=self.transform,
            # Original parameters
            downsample_class=self.cfg.downsample_class,
            downsample_ratio=self.cfg.downsample_ratio,
            # New multi-class parameters
            downsample_classes=downsample_classes,
            naive_oversample=self.cfg.naive_oversample,
            naive_undersample=self.cfg.naive_undersample,
            smote=self.cfg.smote,
            adasyn=self.cfg.adasyn,
            random_state=self.cfg.seed,
            add_extra_images=self.cfg.add_extra_images,
            extra_images_dir=self.cfg.extra_images_dir,
            max_extra_images=self.cfg.max_extra_images,
            extra_images_per_class=extra_images_per_class,
            keep_only_cat=self.cfg.keep_only_cat,
            download=download_flag,
        )

        new_mean, new_std = full_train_dataset.get_new_std_mean()

        self.test_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=new_mean, std=new_std)]
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
        val = DataLoader(
            self.val_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            persistent_workers=True,
        )
        test = DataLoader(
            self.test_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            persistent_workers=True,
        )
        return [val, test]

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            persistent_workers=True,
        )
