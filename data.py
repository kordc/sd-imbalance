import os
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
from torchvision.transforms import v2 as transforms

# from scripts.flux_redux_augment import FluxReduxAugment
from utils import CIFAR10_CLASSES


class DownsampledCIFAR10(torchvision.datasets.CIFAR10):
    def __init__(
        self,
        root,
        train=True,
        transform=None,
        download=True,
        downsample_classes=None,  # Dict of class_name:ratio pairs
        naive_oversample=False,
        naive_undersample=False,
        smote=False,
        adasyn=False,
        random_state=42,
        add_extra_images=False,
        extra_images_dir="extra-images",
        extra_images_per_class=None,  # Dict of class_name:count pairs
        normalize_synthetic=None,  # None, 'mean_std', or 'clahe'
        similarity_filter=None,  # None, 'original', or 'synthetic'
        similarity_threshold=0.7,  # Threshold for similarity filtering (0.0-1.0)
        reference_sample_size=50,  # Number of reference images to use
    ) -> None:
        super().__init__(root=root, train=train, transform=transform, download=download)

        orig_data = self.data.astype(np.float32) / 255.0
        self.original_mean = np.mean(orig_data, axis=(0, 1, 2))
        self.original_std = np.std(orig_data, axis=(0, 1, 2))
        print(
            f"Original dataset stats stored - Mean: {self.original_mean}, Std: {self.original_std}"
        )

        self.normalize_synthetic = normalize_synthetic
        self.similarity_filter = similarity_filter
        self.similarity_threshold = similarity_threshold
        self.reference_sample_size = reference_sample_size

        # New multi-class configuration
        self.downsample_classes = downsample_classes or {}

        self.naive_oversample = naive_oversample
        self.naive_undersample = naive_undersample
        self.smote = smote
        self.adasyn = adasyn
        self.random_state = random_state

        # Image addition parameters
        self.add_extra_images = add_extra_images
        self.extra_images_dir = extra_images_dir
        self.extra_images_per_class = extra_images_per_class or {}

        self._extra_images_added = False  # Flag to avoid adding twice

        if self.downsample_classes:
            self._downsample_multiple()

        if self.add_extra_images and not self._extra_images_added:
            self._add_extra_images()
            self._extra_images_added = True

        self._apply_resampling()
        self.update_normalization()

    def update_normalization(self):
        """Update the normalization parameters for the transforms after data changes."""
        if self.transform is None:
            return

        data = self.data.astype(np.float32) / 255.0
        mean = np.mean(data, axis=(0, 1, 2))
        std = np.std(data, axis=(0, 1, 2))

        if isinstance(self.transform, transforms.Compose):
            found = False
            for i, t in enumerate(self.transform.transforms):
                if isinstance(t, transforms.Normalize):
                    self.transform.transforms[i] = transforms.Normalize(
                        mean=mean.tolist(),
                        std=std.tolist(),
                    )
                    found = True
                    break
            if not found:
                # Append Normalize transform if not present
                self.transform.transforms.append(
                    transforms.Normalize(mean=mean.tolist(), std=std.tolist())
                )

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

        # Process each class to downsample
        for class_id, ratio in class_id_ratios.items():
            if class_id in np.unique(targets):
                class_indices = selected_idx[targets == class_id]
                keep_size = int(len(class_indices) * ratio)
                keep_idx = np.random.choice(
                    class_indices, size=keep_size, replace=False
                )
                keep_indices.append(keep_idx)

        # Get indices of classes not being downsampled
        downsampled_classes = list(class_id_ratios.keys())
        non_downsampled_indices = selected_idx[~np.isin(targets, downsampled_classes)]

        if keep_indices:
            all_keep_indices = np.concatenate([non_downsampled_indices] + keep_indices)
            self.data = self.data[all_keep_indices]
            self.targets = list(targets[all_keep_indices])

    def _apply_resampling(self):
        """Apply resampling methods (SMOTE, ADASYN, etc.) after downsampling."""
        num_resampling = sum(
            [self.naive_oversample, self.naive_undersample, self.smote, self.adasyn]
        )
        if num_resampling > 1:
            raise ValueError(
                "Only one of naive_oversample, naive_undersample, smote, or adasyn can be True at a time."
            )

        if not any(
            [self.naive_oversample, self.naive_undersample, self.smote, self.adasyn]
        ):
            return

        data_reshaped = self.data.reshape(self.data.shape[0], -1)

        if self.naive_oversample:
            ros = RandomOverSampler(
                sampling_strategy="auto", random_state=self.random_state
            )
            resampled_data, resampled_targets = ros.fit_resample(
                data_reshaped, self.targets
            )
        elif self.naive_undersample:
            if self.downsample_classes:
                targets = np.array(self.targets)
                target_sizes = {}
                for class_id in np.unique(targets):
                    target_sizes[class_id] = len(targets[targets == class_id])
                rus = RandomUnderSampler(
                    sampling_strategy=target_sizes, random_state=self.random_state
                )
            else:
                rus = RandomUnderSampler(
                    sampling_strategy="auto", random_state=self.random_state
                )
            resampled_data, resampled_targets = rus.fit_resample(
                data_reshaped, self.targets
            )
        elif self.smote:
            smote = SMOTE(sampling_strategy="auto", random_state=self.random_state)
            resampled_data, resampled_targets = smote.fit_resample(
                data_reshaped, self.targets
            )
        elif self.adasyn:
            adasyn = ADASYN(sampling_strategy="auto", random_state=self.random_state)
            resampled_data, resampled_targets = adasyn.fit_resample(
                data_reshaped, self.targets
            )

        resampled_data = resampled_data.reshape(-1, 32, 32, 3)
        self.data = resampled_data
        self.targets = list(resampled_targets)

    def _add_extra_images(self) -> None:
        """
        Adds extra images from a directory into the training data.
        Images are filtered by CLIP similarity and normalized using selected method.
        Images should be named as CLASS_idx.png/jpg (e.g., cat_1.png, airplane_42.jpg).
        """
        import cv2
        import clip
        import torch
        from torchvision import transforms

        image_files = glob(os.path.join(self.extra_images_dir, "*.*"))
        if not image_files:
            print(f"No images found in {self.extra_images_dir}")
            exit(1)

        class_to_files = {}
        for fpath in image_files:
            filename = os.path.basename(fpath)
            try:
                class_name = filename.split("_")[0]
                if class_name in CIFAR10_CLASSES:
                    class_to_files.setdefault(class_name, []).append(fpath)
            except Exception as e:
                print(f"Skipping file with unexpected format: {filename} ({e})")

        orig_mean = self.original_mean
        orig_std = self.original_std
        print(
            f"Using original dataset statistics for normalization - Mean: {orig_mean}, Std: {orig_std}"
        )

        clip_model = None
        clip_preprocess = None
        if self.similarity_filter:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Loading CLIP model for similarity filtering (device: {device})")
            clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

        new_images_list = []
        target_classes = []

        for class_name, files in class_to_files.items():
            class_idx = CIFAR10_CLASSES[class_name]

            if not self.extra_images_per_class:
                continue

            if (
                self.extra_images_per_class
                and class_name in self.extra_images_per_class
            ):
                num_to_use = self.extra_images_per_class[class_name]
                if num_to_use == 0:
                    print(
                        f"Skipping class '{class_name}' as per-class config is set to 0"
                    )
                    continue
                else:
                    print(
                        f"Adding {num_to_use} images for class '{class_name}' (per-class config)"
                    )

            else:
                num_to_use = len(files)
                print(f"Adding all filtered images for class '{class_name}'")

            class_images = []
            file_paths = []
            for fpath in files:
                try:
                    with Image.open(fpath) as img:
                        img = img.convert("RGB")
                        img = img.resize((32, 32))
                        arr = np.array(img)
                        class_images.append(arr)
                        file_paths.append(fpath)
                except Exception as e:
                    print(f"Error processing {fpath}: {e}")

            if not class_images:
                print(f"No valid images found for class {class_name}")
                continue

            if self.similarity_filter is not None and clip_model is not None:
                print(
                    f"Filtering {len(class_images)} images for class {class_name} using {self.similarity_filter} reference..."
                )

                synth_embeddings = []
                valid_indices = []
                for i, img_arr in enumerate(class_images):
                    try:
                        img_pil = Image.fromarray(img_arr)
                        image_input = clip_preprocess(img_pil).unsqueeze(0).to(device)
                        with torch.no_grad():
                            embedding = clip_model.encode_image(image_input)
                            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
                            synth_embeddings.append(embedding.cpu().numpy()[0])
                            valid_indices.append(i)
                    except Exception as e:
                        print(f"Error computing embedding: {e}")

                if not synth_embeddings:
                    print(f"No valid embeddings for class {class_name}")
                    continue

                synth_embeddings = np.array(synth_embeddings)

                ref_embeddings = None
                if self.similarity_filter == "original":
                    original_indices = [
                        i
                        for i, target in enumerate(self.targets)
                        if target == class_idx
                    ]
                    if len(original_indices) == 0:
                        print(
                            f"No original images found for class {class_name}. Using synthetic reference instead."
                        )
                        self.similarity_filter = "synthetic"
                    else:
                        num_refs = min(
                            self.reference_sample_size, len(original_indices)
                        )
                        ref_indices = np.random.choice(
                            original_indices, size=num_refs, replace=False
                        )
                        to_pil = transforms.ToPILImage()
                        ref_embeddings = []
                        for idx in ref_indices:
                            try:
                                img_tensor = transforms.ToTensor()(self.data[idx])
                                img_pil = to_pil(img_tensor)
                                image_input = (
                                    clip_preprocess(img_pil).unsqueeze(0).to(device)
                                )
                                with torch.no_grad():
                                    embedding = clip_model.encode_image(image_input)
                                    embedding = embedding / embedding.norm(
                                        dim=-1, keepdim=True
                                    )
                                    ref_embeddings.append(embedding.cpu().numpy()[0])
                            except Exception as e:
                                print(f"Error processing reference image: {e}")

                        if not ref_embeddings:
                            print(
                                "Failed to create original reference embeddings. Using synthetic reference."
                            )
                            self.similarity_filter = "synthetic"
                        else:
                            ref_embeddings = np.array(ref_embeddings)

                if self.similarity_filter == "synthetic":
                    num_refs = min(self.reference_sample_size, len(synth_embeddings))
                    ref_indices = np.random.choice(
                        len(synth_embeddings), size=num_refs, replace=False
                    )
                    ref_embeddings = synth_embeddings[ref_indices]

                similarity_scores = np.dot(synth_embeddings, ref_embeddings.T).mean(
                    axis=1
                )
                filtered_indices = [
                    valid_indices[i]
                    for i, score in enumerate(similarity_scores)
                    if score >= self.similarity_threshold
                ]
                print(
                    f"Filtered {len(class_images)} to {len(filtered_indices)} images with similarity >= {self.similarity_threshold}"
                )
                class_images = [class_images[i] for i in filtered_indices]
                file_paths = [file_paths[i] for i in filtered_indices]

            if num_to_use < len(class_images):
                indices = np.random.choice(
                    len(class_images), size=num_to_use, replace=False
                )
                class_images = [class_images[i] for i in indices]

            if not class_images:
                print(f"No images left after filtering for class {class_name}")
                continue

            if (
                self.normalize_synthetic
                and orig_mean is not None
                and orig_std is not None
            ):
                class_images_array = np.stack(class_images)

                if self.normalize_synthetic == "mean_std":
                    synth_data = class_images_array.astype(np.float32) / 255.0
                    synth_mean = np.mean(synth_data, axis=(0, 1, 2))
                    synth_std = np.std(synth_data, axis=(0, 1, 2))
                    print(
                        f"Class {class_name} synthetic stats - Mean: {synth_mean}, Std: {synth_std}"
                    )

                    normalized = (
                        synth_data - synth_mean[None, None, None, :]
                    ) / synth_std[None, None, None, :]
                    normalized = (
                        normalized * orig_std[None, None, None, :]
                        + orig_mean[None, None, None, :]
                    )
                    normalized = np.clip(normalized * 255, 0, 255).astype(np.uint8)
                    class_images = list(normalized)

                elif self.normalize_synthetic == "clahe":
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    normalized = []
                    for img in class_images_array:
                        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
                        l, a, b = cv2.split(lab)  # noqa: E741
                        l_clahe = clahe.apply(l)
                        lab_clahe = cv2.merge((l_clahe, a, b))
                        rgb_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
                        rgb_float = rgb_clahe.astype(np.float32) / 255.0
                        img_mean = np.mean(rgb_float, axis=(0, 1))
                        img_std = np.std(rgb_float, axis=(0, 1))
                        rgb_adjusted = (rgb_float - img_mean[None, None, :]) / (
                            img_std[None, None, :] + 1e-6
                        )
                        rgb_adjusted = (
                            rgb_adjusted * orig_std[None, None, :]
                            + orig_mean[None, None, :]
                        )
                        rgb_adjusted = np.clip(rgb_adjusted * 255, 0, 255).astype(
                            np.uint8
                        )
                        normalized.append(rgb_adjusted)
                    class_images = normalized

            new_images_list.extend(class_images)
            target_classes.extend([class_idx] * len(class_images))

        if not new_images_list:
            print("No valid images found to add")
            exit(1)

        new_images = np.stack(new_images_list, axis=0)
        print(f"Normalization method: {self.normalize_synthetic}")
        print(f"New images shape: {new_images.shape}")
        print(f"New images range: [{new_images.min()}, {new_images.max()}]")

        self.data = np.concatenate([self.data, new_images], axis=0)
        self.targets.extend(target_classes)
        print(f"Added a total of {len(new_images_list)} new images to the dataset")

    def _downsample(self) -> None:
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

        self.data_prepared = False

    def get_augmentations(self, augmentations_cfg):
        transform_list = []
        if "FluxReduxAugment" in augmentations_cfg:
            from scripts.flux_redux_augment import FluxReduxAugment

            custom_transforms = {"FluxReduxAugment": FluxReduxAugment}
        else:
            custom_transforms = {}
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
                    0: Image.NEAREST,
                    1: Image.BILINEAR,
                    2: Image.BICUBIC,
                    3: Image.LANCZOS,
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
        if self.data_prepared:
            return
        print("Preparing data...")
        cifar10_train_path = os.path.join("./data", "cifar-10-batches-py")
        download_flag = not os.path.exists(cifar10_train_path)

        downsample_classes = getattr(self.cfg, "downsample_classes", {})
        extra_images_per_class = getattr(self.cfg, "extra_images_per_class", {})

        full_train_dataset = DownsampledCIFAR10(
            root="./data",
            train=True,
            transform=self.transform,
            downsample_classes=downsample_classes,
            naive_oversample=self.cfg.naive_oversample,
            naive_undersample=self.cfg.naive_undersample,
            smote=self.cfg.smote,
            adasyn=self.cfg.adasyn,
            random_state=self.cfg.seed,
            add_extra_images=self.cfg.add_extra_images,
            extra_images_dir=self.cfg.extra_images_dir,
            extra_images_per_class=extra_images_per_class,
            download=download_flag,
            normalize_synthetic=self.cfg.normalize_synthetic,
            similarity_filter=self.cfg.similarity_filter,
            similarity_threshold=self.cfg.similarity_threshold,
            reference_sample_size=self.cfg.reference_sample_size,
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

        class_counts = torch.bincount(train_targets)
        class_weights = 1.0 / class_counts.float()
        self.class_weights = class_weights / class_weights.sum()

        self.data_prepared = True

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
