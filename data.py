# data.py
import os
from glob import glob

import lightning as L
import numpy as np
import torch
import torchvision
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import v2 as transforms_v2  # Renamed to avoid conflict
from torch.utils.data import Dataset
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2

# Removed 'import clip' as we'll use transformers' CLIPProcessor
from transformers import CLIPProcessor  # Added for CLIP model

from utils import CIFAR10_CLASSES


class DownsampledCIFAR10(torchvision.datasets.CIFAR10):
    """
    (Docstring largely the same)
    """

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Any] = None,  # Changed type to Any for flexibility
        download: bool = True,
        downsample_classes: Optional[Dict[str, float]] = None,
        naive_oversample: bool = False,
        naive_undersample: bool = False,
        smote: bool = False,
        adasyn: bool = False,
        random_state: int = 42,
        add_extra_images: bool = False,
        extra_images_dir: str = "extra-images",
        extra_images_per_class: Optional[Dict[str, int]] = None,
        normalize_synthetic: Optional[str] = None,
        similarity_filter: Optional[str] = None,
        similarity_threshold: float = 0.7,
        reference_sample_size: int = 50,
        # Added cfg to access model_type for normalization updates
        cfg: Optional[DictConfig] = None,
    ) -> None:
        super().__init__(root=root, train=train, transform=transform, download=download)
        self.cfg = cfg  # Store cfg

        orig_data_for_stats = self.data.astype(np.float32) / 255.0
        self.original_mean = np.mean(orig_data_for_stats, axis=(0, 1, 2))
        self.original_std = np.std(orig_data_for_stats, axis=(0, 1, 2))
        print(
            f"Original dataset stats stored - Mean: {self.original_mean}, Std: {self.original_std}"
        )

        self.normalize_synthetic = normalize_synthetic
        self.similarity_filter = similarity_filter
        self.similarity_threshold = similarity_threshold
        self.reference_sample_size = reference_sample_size
        self.downsample_classes = downsample_classes or {}
        self.naive_oversample = naive_oversample
        self.naive_undersample = naive_undersample
        self.smote = smote
        self.adasyn = adasyn
        self.random_state = random_state
        self.add_extra_images = add_extra_images
        self.extra_images_dir = extra_images_dir
        self.extra_images_per_class = extra_images_per_class or {}
        self._extra_images_added = False

        if self.downsample_classes:
            self._downsample_multiple()

        if self.add_extra_images and not self._extra_images_added:
            self._add_extra_images()  # This calls CLIP if needed internally
            self._extra_images_added = True

        self._apply_resampling()
        # Normalization update is now conditional and potentially handled by DataModule for CLIP
        if self.cfg is None or self.cfg.model_type == "resnet18":
            self.update_normalization()  # Only for ResNet style

    def update_normalization(self) -> None:
        """
        Updates the normalization parameters in the active transform (if Compose and Normalize exists)
        based on the current mean and standard deviation of the dataset.
        This is primarily for ResNet-style models. CLIP uses fixed normalization.
        """
        if self.transform is None or not isinstance(
            self.transform, transforms_v2.Compose
        ):
            return

        # This logic is for when self.transform is a Compose of torchvision.transforms
        # If self.transform is a CLIPProcessor, this won't apply.
        current_data = self.data.astype(np.float32) / 255.0
        mean = np.mean(current_data, axis=(0, 1, 2))
        std = np.std(current_data, axis=(0, 1, 2))

        found = False
        for i, t in enumerate(self.transform.transforms):
            if isinstance(t, transforms_v2.Normalize):
                self.transform.transforms[i] = transforms_v2.Normalize(
                    mean=mean.tolist(), std=std.tolist()
                )
                found = True
                print(
                    f"Updated Normalize transform with mean: {mean.tolist()}, std: {std.tolist()}"
                )
                break
        if not found:
            # If Normalize is not found, it might be that the initial transform didn't include it,
            # or it's handled differently (e.g. by CLIPProcessor).
            # For ResNet, it's typically added.
            self.transform.transforms.append(
                transforms_v2.Normalize(mean=mean.tolist(), std=std.tolist())
            )
            print(
                f"Appended Normalize transform with mean: {mean.tolist()}, std: {std.tolist()}"
            )

    def get_new_std_mean(self) -> Tuple[List[float], List[float]]:
        current_data = self.data.astype(np.float32) / 255.0
        mean = np.mean(current_data, axis=(0, 1, 2))
        std = np.std(current_data, axis=(0, 1, 2))
        return mean.tolist(), std.tolist()

    def _downsample_multiple(self) -> None:
        # (No changes needed here)
        targets = np.array(self.targets)
        selected_idx = np.arange(len(targets))
        keep_indices: List[np.ndarray] = []

        class_id_ratios: Dict[int, float] = {}
        for class_name, ratio in self.downsample_classes.items():
            if isinstance(class_name, str):
                class_id = CIFAR10_CLASSES.get(class_name)
                if class_id is not None:
                    class_id_ratios[class_id] = ratio
            else:
                class_id_ratios[class_name] = ratio  # type: ignore

        for class_id, ratio in class_id_ratios.items():
            if class_id in np.unique(targets):
                class_indices = selected_idx[targets == class_id]
                keep_size = int(len(class_indices) * ratio)
                keep_idx = np.random.choice(
                    class_indices, size=keep_size, replace=False
                )
                keep_indices.append(keep_idx)

        downsampled_classes_keys = list(class_id_ratios.keys())
        non_downsampled_indices = selected_idx[
            ~np.isin(targets, downsampled_classes_keys)
        ]

        if keep_indices:
            all_keep_indices = np.concatenate([non_downsampled_indices] + keep_indices)
            self.data = self.data[all_keep_indices]
            self.targets = [
                targets[i] for i in all_keep_indices
            ]  # Corrected way to update targets

    def _apply_resampling(self) -> None:
        # (No changes needed here)
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

        resampled_data: np.ndarray
        resampled_targets: Union[List[int], np.ndarray]  # Adjusted type

        if self.naive_oversample:
            ros = RandomOverSampler(
                sampling_strategy="auto", random_state=self.random_state
            )
            resampled_data, resampled_targets = ros.fit_resample(
                data_reshaped, self.targets
            )
        elif self.naive_undersample:
            # ... (rest of the undersampling logic)
            targets_np = np.array(self.targets)
            target_sizes: Dict[int, int] = {}
            for class_id_val in np.unique(targets_np):  # Ensure class_id is int
                class_id: int = int(class_id_val)
                target_sizes[class_id] = len(targets_np[targets_np == class_id])

            rus = RandomUnderSampler(
                sampling_strategy=target_sizes if self.downsample_classes else "auto",  # type: ignore
                random_state=self.random_state,
            )
            resampled_data, resampled_targets = rus.fit_resample(
                data_reshaped, self.targets
            )
        elif self.smote:
            smote_sampler = SMOTE(
                sampling_strategy="auto", random_state=self.random_state
            )  # Renamed variable
            resampled_data, resampled_targets = smote_sampler.fit_resample(
                data_reshaped, self.targets
            )
        elif self.adasyn:
            adasyn_sampler = ADASYN(
                sampling_strategy="auto", random_state=self.random_state
            )  # Renamed variable
            resampled_data, resampled_targets = adasyn_sampler.fit_resample(
                data_reshaped, self.targets
            )
        else:
            return

        resampled_data = resampled_data.reshape(-1, 32, 32, 3)
        self.data = resampled_data.astype(np.uint8)  # Ensure data type is consistent
        self.targets = list(resampled_targets)

    def _add_extra_images(self) -> None:
        # This method uses `clip.load` directly. We should adapt if we want to use transformers.CLIPProcessor here
        # For now, keeping it as is, but noting that `transformers.CLIPModel` and `transformers.CLIPProcessor`
        # are used in `ClipClassifier`. If `similarity_filter` is active, this might lead to loading CLIP twice.
        # A refactor could pass the CLIP model/processor if already loaded.
        # However, `DownsampledCIFAR10` is a generic dataset, might be okay for it to have its own CLIP instance
        # if `similarity_filter` is used.
        # IMPORTING `clip` library HERE if similarity_filter is active
        _clip_model = None
        _clip_preprocess = None
        if self.similarity_filter:
            try:
                import clip as openai_clip  # Use a distinct name

                device = "cuda" if torch.cuda.is_available() else "cpu"
                print(
                    f"Loading OpenAI CLIP model for similarity filtering (device: {device})"
                )
                _clip_model, _clip_preprocess = openai_clip.load(
                    "ViT-B/32", device=device
                )
            except ImportError:
                print(
                    "OpenAI 'clip' library not found. Similarity filtering will be skipped."
                )
                self.similarity_filter = None  # Disable if not found

        # (Rest of _add_extra_images remains the same, using _clip_model and _clip_preprocess)
        # ... (previous code for _add_extra_images)
        image_files = glob(os.path.join(self.extra_images_dir, "*.*"))
        if not image_files:
            print(f"No images found in {self.extra_images_dir}")
            return

        class_to_files: Dict[str, List[str]] = {}
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

        new_images_list: List[np.ndarray] = []
        target_classes: List[int] = []

        for class_name, files in class_to_files.items():
            class_idx = CIFAR10_CLASSES[class_name]
            if not self.extra_images_per_class:
                continue
            num_to_use = self.extra_images_per_class.get(class_name, len(files))
            if num_to_use == 0:
                continue

            class_images_pil: List[Image.Image] = []  # Store PIL images for CLIP
            class_images_arrays: List[np.ndarray] = []  # Store numpy arrays for dataset

            for fpath in files:
                try:
                    with Image.open(fpath) as img:
                        img_rgb = img.convert("RGB")
                        img_resized_pil = img_rgb.resize(
                            (32, 32)
                        )  # For storage & ResNet
                        class_images_pil.append(img_rgb)  # Original for CLIP processing
                        class_images_arrays.append(np.array(img_resized_pil))
                except Exception as e:
                    print(f"Error processing {fpath}: {e}")

            if not class_images_arrays:
                continue

            if self.similarity_filter and _clip_model and _clip_preprocess:
                # Similarity filtering logic (using _clip_model, _clip_preprocess)
                device = "cuda" if torch.cuda.is_available() else "cpu"
                synth_embeddings_list: List[np.ndarray] = []
                valid_img_indices_for_filter: List[int] = []

                for i, pil_img in enumerate(
                    class_images_pil
                ):  # Use original PIL images for CLIP
                    try:
                        image_input = _clip_preprocess(pil_img).unsqueeze(0).to(device)
                        with torch.no_grad():
                            embedding = _clip_model.encode_image(image_input)
                            embedding_norm = embedding / embedding.norm(
                                dim=-1, keepdim=True
                            )
                            synth_embeddings_list.append(
                                embedding_norm.cpu().numpy()[0]
                            )
                            valid_img_indices_for_filter.append(i)
                    except Exception as e:
                        print(f"Error computing embedding for {files[i]}: {e}")

                if not synth_embeddings_list:
                    continue
                synth_embeddings = np.array(synth_embeddings_list)

                # Reference embeddings logic (original or synthetic)
                ref_embeddings_np: Optional[np.ndarray] = None
                if self.similarity_filter == "original":
                    # ... (get original_indices, sample ref_indices)
                    original_indices = [
                        i
                        for i, target in enumerate(self.targets)
                        if target == class_idx
                    ]
                    if not original_indices:
                        print(
                            f"No original images for class {class_name}, switching to synthetic reference."
                        )
                        self.similarity_filter = "synthetic"  # Fallback
                    else:
                        num_refs = min(
                            self.reference_sample_size, len(original_indices)
                        )
                        ref_indices_orig = np.random.choice(
                            original_indices, size=num_refs, replace=False
                        )

                        temp_ref_embeddings = []
                        for idx in ref_indices_orig:
                            # self.data[idx] is (H,W,C) numpy uint8
                            pil_ref_img = Image.fromarray(self.data[idx])
                            try:
                                image_input = (
                                    _clip_preprocess(pil_ref_img)
                                    .unsqueeze(0)
                                    .to(device)
                                )  # Use original PIL
                                with torch.no_grad():
                                    embedding = _clip_model.encode_image(image_input)
                                    embedding_norm = embedding / embedding.norm(
                                        dim=-1, keepdim=True
                                    )
                                    temp_ref_embeddings.append(
                                        embedding_norm.cpu().numpy()[0]
                                    )
                            except Exception as e:
                                print(f"Error embedding original ref image: {e}")
                        if temp_ref_embeddings:
                            ref_embeddings_np = np.array(temp_ref_embeddings)
                        else:
                            self.similarity_filter = "synthetic"  # Fallback

                if self.similarity_filter == "synthetic":  # Handles fallback too
                    num_refs = min(self.reference_sample_size, len(synth_embeddings))
                    if num_refs > 0:
                        ref_indices_synth = np.random.choice(
                            len(synth_embeddings), size=num_refs, replace=False
                        )
                        ref_embeddings_np = synth_embeddings[ref_indices_synth]

                if ref_embeddings_np is not None and len(ref_embeddings_np) > 0:
                    similarity_scores = np.dot(
                        synth_embeddings, ref_embeddings_np.T
                    ).mean(axis=1)

                    final_filtered_indices_original = [
                        valid_img_indices_for_filter[i]
                        for i, score in enumerate(similarity_scores)
                        if score >= self.similarity_threshold
                    ]
                    class_images_arrays = [
                        class_images_arrays[i] for i in final_filtered_indices_original
                    ]
                    # class_images_pil is not strictly needed beyond this point for filtering
                    print(
                        f"Class {class_name}: Filtered to {len(class_images_arrays)} images by similarity."
                    )
                else:
                    print(
                        f"Class {class_name}: No ref embeddings, skipping similarity filter."
                    )

            if not class_images_arrays:
                continue

            # Take num_to_use from the (potentially filtered) images
            if num_to_use < len(class_images_arrays):
                indices_to_select = np.random.choice(
                    len(class_images_arrays), size=num_to_use, replace=False
                )
                class_images_arrays = [
                    class_images_arrays[i] for i in indices_to_select
                ]

            # Normalization logic (applied to class_images_arrays)
            if (
                self.normalize_synthetic
                and orig_mean is not None
                and orig_std is not None
                and class_images_arrays
            ):
                # ... (Normalization like 'mean_std' or 'clahe' on class_images_arrays)
                # This part remains as it was, operating on the 32x32 numpy arrays
                class_images_array_np = np.stack(class_images_arrays)
                if self.normalize_synthetic == "mean_std":
                    synth_data = class_images_array_np.astype(np.float32) / 255.0
                    synth_mean = np.mean(synth_data, axis=(0, 1, 2))
                    synth_std = np.std(synth_data, axis=(0, 1, 2))
                    normalized_synth = (
                        synth_data - synth_mean[None, None, None, :]
                    ) / (synth_std[None, None, None, :] + 1e-6)
                    normalized_synth = (
                        normalized_synth * orig_std[None, None, None, :]
                        + orig_mean[None, None, None, :]
                    )
                    normalized_synth = np.clip(normalized_synth * 255, 0, 255).astype(
                        np.uint8
                    )
                    class_images_arrays = list(normalized_synth)
                elif self.normalize_synthetic == "clahe":
                    # ... CLAHE logic ...
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    normalized_clahe_list: List[np.ndarray] = []
                    for img_arr_clahe in class_images_array_np:
                        lab = cv2.cvtColor(img_arr_clahe, cv2.COLOR_RGB2LAB)
                        l_channel, a_channel, b_channel = cv2.split(lab)
                        l_clahe = clahe.apply(l_channel)
                        lab_clahe = cv2.merge((l_clahe, a_channel, b_channel))
                        rgb_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
                        # Further match to original mean/std
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
                        rgb_adjusted_uint8 = np.clip(rgb_adjusted * 255, 0, 255).astype(
                            np.uint8
                        )
                        normalized_clahe_list.append(rgb_adjusted_uint8)
                    class_images_arrays = normalized_clahe_list

            new_images_list.extend(class_images_arrays)
            target_classes.extend([class_idx] * len(class_images_arrays))

        if not new_images_list:
            print("No valid images found to add after all processing.")
            return

        new_images_np = np.stack(new_images_list, axis=0)
        self.data = np.concatenate([self.data, new_images_np], axis=0)
        self.targets.extend(target_classes)
        print(f"Added a total of {len(new_images_list)} new images to the dataset.")

    def _downsample(self) -> None:
        self._downsample_multiple()


class CIFAR10DataModule(L.LightningDataModule):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.train_transform: Optional[Any] = None  # PIL -> Tensor
        self.val_test_transform: Optional[Any] = None  # PIL -> Tensor

        self.train_dataset: Optional[
            Union[torch.utils.data.Subset, DownsampledCIFAR10]
        ] = None
        self.val_dataset: Optional[torch.utils.data.Subset] = None
        self.test_dataset: Optional[
            Union[torchvision.datasets.CIFAR10, RealImagesTestDataset]
        ] = None

        self.data_prepared = False
        self.class_weights: Optional[torch.Tensor] = None
        self.clip_processor = None  # Store CLIP processor if used

        # Initialize transforms based on model_type
        if self.cfg.model_type == "clip":
            print(
                f"Initializing CLIP-specific transforms using {self.cfg.clip_model_name}."
            )
            self.clip_processor = CLIPProcessor.from_pretrained(
                self.cfg.clip_model_name
            )
            # The processor itself can be used as a transform if it handles PIL/numpy to tensor
            # CLIPProcessor output for images is pixel_values (Tensor)
            self.train_transform = lambda pil_img: self.clip_processor(
                images=pil_img, return_tensors="pt", padding=True
            ).pixel_values.squeeze(0)
            self.val_test_transform = lambda pil_img: self.clip_processor(
                images=pil_img, return_tensors="pt", padding=True
            ).pixel_values.squeeze(0)
        elif self.cfg.model_type == "resnet18":
            print("Initializing ResNet18-specific transforms.")
            self.train_transform = self.get_resnet_augmentations(self.cfg.augmentations)
            # Test transform for ResNet will be set in prepare_data after deriving mean/std
        else:
            raise ValueError(
                f"Unsupported model_type in CIFAR10DataModule: {self.cfg.model_type}"
            )

    def get_resnet_augmentations(
        self, augmentations_cfg: List[Dict[str, Any]]
    ) -> transforms_v2.Compose:
        transform_list: List[Any] = []
        # ... (original get_augmentations logic, but renamed and using transforms_v2)
        custom_transforms: Dict[str, Any] = {}
        if "FluxReduxAugment" in [aug["name"] for aug in augmentations_cfg]:
            from scripts.flux_redux_augment import FluxReduxAugment

            custom_transforms = {"FluxReduxAugment": FluxReduxAugment}

        for aug in augmentations_cfg:
            aug_name = aug["name"]
            params = aug.get("params", {})
            if aug_name in custom_transforms:
                transform_list.append(custom_transforms[aug_name](**params))
            elif (
                "resize" in aug_name.lower()
            ):  # Specific handling for torchvision.transforms.Resize
                interpolation_mode = params.get("interpolation", "bilinear")
                interpolation_map = {
                    "nearest": transforms_v2.InterpolationMode.NEAREST,
                    "bilinear": transforms_v2.InterpolationMode.BILINEAR,
                    "bicubic": transforms_v2.InterpolationMode.BICUBIC,
                    # Add others if needed, like LANCZOS
                }
                params["interpolation"] = interpolation_map.get(
                    interpolation_mode, transforms_v2.InterpolationMode.BILINEAR
                )
                transform_list.append(getattr(transforms_v2, aug_name)(**params))
            elif aug_name == "ToTensor":  # Ensure ToTensor is from v2 if others are
                transform_list.append(transforms_v2.ToTensor())
            elif (
                aug_name == "Normalize"
            ):  # Normalize will be added/updated later for ResNet
                continue  # Skip adding Normalize here, will be handled by update_normalization or in prepare_data
            else:
                try:
                    transform_list.append(getattr(transforms_v2, aug_name)(**params))
                except (
                    AttributeError
                ):  # Fallback to torchvision.transforms if not in v2
                    import torchvision.transforms as tv_transforms

                    transform_list.append(getattr(tv_transforms, aug_name)(**params))

        # Ensure ToTensor is present if not already for ResNet
        if not any(
            isinstance(t, (transforms_v2.ToTensor, torchvision.transforms.ToTensor))
            for t in transform_list
        ):
            transform_list.insert(
                0, transforms_v2.ToTensor()
            )  # PIL to Tensor first for many v2 transforms

        # For ResNet, Normalize will be added/updated in prepare_data or by DownsampledCIFAR10's update_normalization
        return transforms_v2.Compose(transform_list)

    def setup(self, stage: Optional[str]) -> None:
        # (No changes needed here)
        if stage == "fit" or stage is None:
            # print(f"CIFAR10DataModule setup called for stage: {stage}. Data preparation is handled in prepare_data.")
            return
        # print(f"CIFAR10DataModule setup called for stage: {stage}. Data preparation is handled in prepare_data.")

    def prepare_data(self) -> None:
        if self.data_prepared:
            return
        print("Preparing data...")
        cifar10_train_path = os.path.join("./data", "cifar-10-batches-py")
        download_flag = not os.path.exists(cifar10_train_path)

        downsample_classes_cfg: Dict[str, float] = OmegaConf.to_container(
            self.cfg.get("downsample_classes", {}), resolve=True
        )  # type: ignore
        extra_images_per_class_cfg: Dict[str, int] = OmegaConf.to_container(
            self.cfg.get("extra_images_per_class", {}), resolve=True
        )  # type: ignore

        # Pass self.cfg to DownsampledCIFAR10 so it knows model_type for normalization
        full_train_dataset = DownsampledCIFAR10(
            root="./data",
            train=True,
            transform=self.train_transform,  # Pass the already configured train_transform
            downsample_classes=downsample_classes_cfg,
            naive_oversample=self.cfg.naive_oversample,
            naive_undersample=self.cfg.naive_undersample,
            smote=self.cfg.smote,
            adasyn=self.cfg.adasyn,
            random_state=self.cfg.seed,
            add_extra_images=self.cfg.add_extra_images,
            extra_images_dir=self.cfg.extra_images_dir,
            extra_images_per_class=extra_images_per_class_cfg,
            download=download_flag,
            normalize_synthetic=self.cfg.normalize_synthetic,
            similarity_filter=self.cfg.similarity_filter,
            similarity_threshold=self.cfg.similarity_threshold,
            reference_sample_size=self.cfg.reference_sample_size,
            cfg=self.cfg,  # Pass the main config
        )

        # For ResNet, set test_transform based on derived mean/std.
        # For CLIP, val_test_transform is already set.
        if self.cfg.model_type == "resnet18":
            new_mean, new_std = full_train_dataset.get_new_std_mean()
            print(
                f"Derived ResNet normalization for test/val - Mean: {new_mean}, Std: {new_std}"
            )
            # Ensure ToTensor is part of test_transform for ResNet
            base_test_transforms = [transforms_v2.ToTensor()]
            if self.cfg.get(
                "test_augmentations"
            ):  # Apply test_augmentations if specified
                for aug_spec in self.cfg.test_augmentations:
                    if aug_spec["name"] == "ToTensor":
                        continue
                    if aug_spec["name"] == "Normalize":
                        continue  # Normalize added below
                    aug_name = aug_spec["name"]
                    params = aug_spec.get("params", {})
                    if "resize" in aug_name.lower():
                        interpolation_mode = params.get("interpolation", "bilinear")
                        interpolation_map = {
                            "nearest": transforms_v2.InterpolationMode.NEAREST,
                            "bilinear": transforms_v2.InterpolationMode.BILINEAR,
                            "bicubic": transforms_v2.InterpolationMode.BICUBIC,
                        }
                        params["interpolation"] = interpolation_map.get(
                            interpolation_mode, transforms_v2.InterpolationMode.BILINEAR
                        )
                    base_test_transforms.append(
                        getattr(transforms_v2, aug_name)(**params)
                    )

            base_test_transforms.append(
                transforms_v2.Normalize(mean=new_mean, std=new_std)
            )
            self.val_test_transform = transforms_v2.Compose(base_test_transforms)

        if self.cfg.get("test_on_real", False):
            print(
                f"Configuring test set to use real images from: {self.cfg.test_images_dir}"
            )
            self.test_dataset = RealImagesTestDataset(
                extra_images_dir=self.cfg.test_images_dir,
                transform=self.val_test_transform,  # Use the common val/test transform
            )
            # ... (warning for empty dataset)
        else:
            print("Configuring test set to use standard CIFAR-10 test data.")
            self.test_dataset = torchvision.datasets.CIFAR10(
                root="./data",
                train=False,
                transform=self.val_test_transform,  # Use the common val/test transform
                download=download_flag,
            )

        # Split full_train_dataset into train_dataset and val_dataset
        val_size_ratio = self.cfg.val_size
        num_train_samples = len(full_train_dataset)

        if num_train_samples == 0:
            print(
                "Error: full_train_dataset is empty after initial setup. Cannot proceed."
            )
            self.train_dataset = torch.utils.data.Subset(
                full_train_dataset, []
            )  # Empty
            self.val_dataset = torch.utils.data.Subset(full_train_dataset, [])  # Empty
        else:
            val_count = int(val_size_ratio * num_train_samples)
            train_count = num_train_samples - val_count

            if train_count <= 0 or val_count < 0:  # val_count can be 0
                print(
                    f"Warning: Train count ({train_count}) or val count ({val_count}) is not appropriate. Adjusting split."
                )
                if (
                    num_train_samples > 0
                ):  # If we have some data, assign all to train, val becomes empty or small
                    self.train_dataset = full_train_dataset
                    self.val_dataset = torch.utils.data.Subset(
                        full_train_dataset, []
                    )  # Empty validation
                # If num_train_samples is 0, handled above.
            else:
                self.train_dataset, self.val_dataset = random_split(
                    full_train_dataset,
                    [train_count, val_count],
                    generator=torch.Generator().manual_seed(self.cfg.seed),
                )

        # Calculate class weights (no changes needed here, uses self.train_dataset)
        # ... (original class weight calculation logic)
        if self.train_dataset and len(self.train_dataset) > 0:
            targets_for_weights: List[int]
            if isinstance(self.train_dataset, torch.utils.data.Subset):
                # Access targets from the underlying DownsampledCIFAR10 dataset
                original_dataset = self.train_dataset.dataset
                if hasattr(original_dataset, "targets") and isinstance(
                    original_dataset.targets, list
                ):
                    targets_for_weights = [
                        original_dataset.targets[i] for i in self.train_dataset.indices
                    ]
                else:
                    print(
                        "Warning: Underlying dataset for subset has no 'targets' or it's not a list."
                    )
                    targets_for_weights = []
            elif isinstance(
                self.train_dataset, DownsampledCIFAR10
            ):  # If it's the full dataset
                if hasattr(self.train_dataset, "targets") and isinstance(
                    self.train_dataset.targets, list
                ):
                    targets_for_weights = self.train_dataset.targets
                else:
                    targets_for_weights = []
            else:  # Should not happen
                targets_for_weights = []

            if targets_for_weights:
                train_targets_tensor = torch.tensor(targets_for_weights)
                class_counts = torch.bincount(
                    train_targets_tensor, minlength=len(CIFAR10_CLASSES)
                )
                # Avoid division by zero for classes not present in the training split
                valid_counts = class_counts.float()
                valid_counts[valid_counts == 0] = (
                    1  # Effectively high weight, but avoids NaN/inf
                )

                class_weights_val = 1.0 / valid_counts
                # Normalize weights so they sum to 1 or num_classes, or use as is
                self.class_weights = (
                    class_weights_val / class_weights_val.sum()
                )  # Normalize to sum to 1
                print(f"Calculated class weights: {self.class_weights.tolist()}")
            else:
                print(
                    "Warning: No targets found in training dataset for class weight calculation."
                )
                self.class_weights = None
        else:
            print(
                "Warning: Training dataset is empty or not set, cannot calculate class weights."
            )
            self.class_weights = None

        self.data_prepared = True

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None or len(self.train_dataset) == 0:
            print("Warning: Train dataset is not prepared or is empty.")
            # Return an empty DataLoader or raise error, for now, let it proceed
            # For safety, if it's None, raise error
            if self.train_dataset is None:
                raise RuntimeError(
                    "Train dataset not prepared. Call prepare_data() first."
                )
            # If empty, Lightning will handle it.
        return DataLoader(
            self.train_dataset,  # type: ignore
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            persistent_workers=True if self.cfg.num_workers > 0 else False,
        )

    def val_dataloader(self) -> List[DataLoader]:
        if self.val_dataset is None or self.test_dataset is None:
            raise RuntimeError(
                "Validation or Test dataset not prepared. Call prepare_data() first."
            )

        # Handle empty val_dataset case gracefully
        val_dl_dataset = self.val_dataset
        if len(self.val_dataset) == 0:
            print(
                "Warning: Validation dataset is empty. Using a dummy placeholder for DataLoader."
            )
            # Create a dummy dataset if val_dataset is empty to avoid DataLoader error with empty list
            # This is a workaround. Ideally, val_size should ensure val_dataset is not empty if validation is expected.
            # For now, if it's an empty Subset, DataLoader should handle it.
            pass  # DataLoader can handle empty Subset

        val = DataLoader(
            val_dl_dataset,  # type: ignore
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            persistent_workers=True if self.cfg.num_workers > 0 else False,
        )
        test = DataLoader(
            self.test_dataset,  # type: ignore
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            persistent_workers=True if self.cfg.num_workers > 0 else False,
        )
        return [val, test]

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None or len(self.test_dataset) == 0:
            print("Warning: Test dataset is not prepared or is empty.")
            if self.test_dataset is None:
                raise RuntimeError(
                    "Test dataset not prepared. Call prepare_data() first."
                )
        return DataLoader(
            self.test_dataset,  # type: ignore
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            persistent_workers=True if self.cfg.num_workers > 0 else False,
        )


class RealImagesTestDataset(Dataset):  # type: ignore
    def __init__(
        self,
        extra_images_dir: str,
        transform: Optional[Any] = None,  # Changed type
        cifar10_classes: Dict[str, int] = CIFAR10_CLASSES,
    ):
        self.extra_images_dir = extra_images_dir
        self.transform = transform
        self.cifar10_classes = cifar10_classes
        self.image_paths: List[str] = []  # Store paths, load on-the-fly
        self.targets: List[int] = []
        self._load_image_paths()

    def _load_image_paths(self) -> None:
        # (Logic similar to before, but stores paths and targets)
        image_files_glob = glob(os.path.join(self.extra_images_dir, "*.*"))
        if not image_files_glob:
            print(
                f"Warning: No images found in {self.extra_images_dir} for RealImagesTestDataset."
            )
            return

        loaded_count = 0
        skipped_count = 0
        for fpath in image_files_glob:
            filename = os.path.basename(fpath)
            class_name_candidate = filename.split("_")[0]  # Simple assumption
            if class_name_candidate in self.cifar10_classes:
                class_idx = self.cifar10_classes[class_name_candidate]
                self.image_paths.append(fpath)
                self.targets.append(class_idx)
                loaded_count += 1
            else:
                skipped_count += 1

        if not self.image_paths:
            print(
                f"No valid images loaded from {self.extra_images_dir} for RealImagesTestDataset."
            )
        else:
            print(
                f"Found {loaded_count} images for RealImagesTestDataset. Skipped {skipped_count}."
            )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[Any, int]:
        img_path, target = self.image_paths[idx], self.targets[idx]
        try:
            img = Image.open(img_path).convert("RGB")  # Load as PIL
        except Exception as e:
            print(
                f"Error loading image {img_path} in RealImagesTestDataset: {e}. Returning dummy data."
            )
            # Return a dummy tensor and target to avoid crashing the batch.
            # This should ideally be handled by filtering out bad images during _load_image_paths.
            dummy_tensor = (
                torch.zeros((3, 32, 32))
                if self.transform is None
                else torch.zeros((3, 224, 224))
            )  # adapt size
            return dummy_tensor, target  # Or a specific error target like -1

        if self.transform:
            img = self.transform(
                img
            )  # Transform (PIL -> Tensor for CLIP, or PIL -> PIL -> Tensor for ResNet)
        return img, target
