import glob
import os

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from omegaconf import DictConfig
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torchmetrics import Accuracy, F1Score
from torchvision.transforms import v2
import timm
from typing import Any, Dict, List, Tuple, Optional
from PIL import Image  # Added for dynamic upsampling in CLIP

from data import DownsampledCIFAR10
from utils import CIFAR10_CLASSES_REVERSE, CIFAR10_CLASSES

# Imports for CLIP
from transformers import CLIPProcessor, CLIPModel

# Text templates for CLIP zero-shot classification
CIFAR10_PROMPT_TEMPLATES = [
    "a photo of a {}."
    # Can add more templates for robustness, e.g., "an image of a {}", "the photo of a {}"
]


class ResNet18Model(L.LightningModule):
    """
    A LightningModule for a ResNet18 model, including custom training logic,
    metrics, and dynamic data handling features.

    This class encapsulates the ResNet18 architecture, loss function,
    optimizers, and the training/validation/test steps, along with
    custom features like CutMix/MixUp, dynamic upsampling, and detailed
    metric logging.
    """

    def __init__(
        self, cfg: DictConfig, class_weights: Optional[torch.Tensor] = None
    ) -> None:
        """
        Initializes the ResNet18Model.

        Args:
            cfg (DictConfig): Configuration object containing model, training,
                              and data parameters.
            class_weights (Optional[torch.Tensor]): Optional tensor of class weights
                                                     to be used in the loss function.
        """
        super().__init__()
        self.cfg = cfg
        # self.model = torchvision.models.resnet18(num_classes=10)
        self.model = timm.create_model(
            "resnet18", num_classes=10, pretrained=self.cfg.pretrained
        )
        # Assuming the original intent was to override the first conv layer
        # with a 3x3 kernel and no maxpool as is common for CIFAR-10 ResNets.
        # This part might need careful review depending on exact desired ResNet-18 variant for CIFAR-10.
        self.model.conv1 = nn.Conv2d(
            3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
        )
        self.model.maxpool = nn.Identity()
        self.class_weights = class_weights

        # Setup the loss: use class weights, label smoothing, or default cross entropy.
        if self.class_weights is not None:
            self.criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        elif cfg.label_smoothing:
            self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.3)
        else:
            self.criterion = torch.nn.CrossEntropyLoss()

        # Define accuracy metrics.
        self.train_accuracy = Accuracy(num_classes=10, task="multiclass")
        self.val_accuracy = Accuracy(num_classes=10, task="multiclass")
        self.test_accuracy = Accuracy(num_classes=10, task="multiclass")

        # Add F1 score metrics
        self.train_f1 = F1Score(num_classes=10, task="multiclass", average="macro")
        self.val_f1 = F1Score(num_classes=10, task="multiclass", average="macro")
        self.test_f1 = F1Score(num_classes=10, task="multiclass", average="macro")

        self.val_confusion_matrices: Dict[int, np.ndarray] = {}
        self.test_confusion_matrix: Optional[np.ndarray] = None

        # Track support for each class in different splits
        self.val_support: Dict[int, np.ndarray] = {}
        self.test_support: Optional[np.ndarray] = None

        # To track all predictions and targets for more detailed metrics at epoch end
        self.val_preds: Dict[int, List[torch.Tensor]] = {}
        self.val_targets: Dict[int, List[torch.Tensor]] = {}
        self.test_preds: List[torch.Tensor] = []
        self.test_targets: List[torch.Tensor] = []

        self.cutmix = v2.CutMix(num_classes=10)
        self.mixup = v2.MixUp(num_classes=10)
        self.cutmix_or_mixup = v2.RandomChoice([self.cutmix, self.mixup])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor (logits).
        """
        return self.model(x)

    def custom_cutmix_cat(
        self, inputs: torch.Tensor, labels: torch.Tensor, beta: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies CutMix only for images belonging to the cat class (index 3).
        For each cat image in the batch, a different cat image is selected (if available)
        and a random patch from it is pasted over the original image.
        Since both images are cats, the label remains unchanged.

        Args:
            inputs (torch.Tensor): The input batch of images.
            labels (torch.Tensor): The corresponding labels for the input images.
            beta (float): The beta parameter for the Beta distribution used to sample lambda.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the new inputs
                                               with CutMix applied and their original labels.
        """
        # Find indices for images with label==3 (cat)
        cat_indices = (labels == 3).nonzero(as_tuple=True)[0]
        if len(cat_indices) < 2:
            # Not enough cat images to perform cutmix
            return inputs, labels

        new_inputs = inputs.clone()
        # For each cat sample, select a different random cat image to mix
        permuted = cat_indices[torch.randperm(len(cat_indices))]
        for idx1, idx2 in zip(cat_indices, permuted):
            if idx1 == idx2:
                continue  # ensure a different image is chosen

            # Sample a lambda value from Beta distribution for this pair
            lam = np.random.beta(beta, beta)
            _, _, H, W = inputs.shape

            # Determine the patch size based on lambda
            cut_ratio = np.sqrt(1 - lam)
            cut_w = int(W * cut_ratio)
            cut_h = int(H * cut_ratio)

            # Choose random location for the patch
            cx = np.random.randint(W)
            cy = np.random.randint(H)
            x1 = np.clip(cx - cut_w // 2, 0, W)
            y1 = np.clip(cy - cut_h // 2, 0, H)
            x2 = np.clip(cx + cut_w // 2, 0, W)
            y2 = np.clip(cy + cut_h // 2, 0, H)

            # Replace region in image idx1 with patch from image idx2
            new_inputs[idx1, :, y1:y2, x1:x2] = inputs[idx2, :, y1:y2, x1:x2]
            # Note: Since both images are of the same class (cat), we leave the label unchanged.
        return new_inputs, labels

    def freeze_backbone(self) -> None:
        """
        Freezes the backbone of the model, allowing only the final layer to be trained.
        """
        # Freeze all backbone parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze the classification fc (last layer)
        for param in self.model.fc.parameters():
            param.requires_grad = True

    def visualize_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """
        Registers a forward hook on a chosen layer (here layer1)
        and returns the feature maps produced for the input x.

        Args:
            x (torch.Tensor): Input tensor for which to visualize feature maps.

        Returns:
            torch.Tensor: The feature maps from the specified layer.
        """
        features: List[torch.Tensor] = []

        def hook_fn(
            module: nn.Module, input: Tuple[torch.Tensor, ...], output: torch.Tensor
        ) -> None:
            """Hook function to capture output of a layer."""
            features.append(output)

        # Choose layer1 for visualization (can be changed as needed)
        hook = self.model.layer1.register_forward_hook(hook_fn)
        self.eval()
        with torch.no_grad():
            _ = self(x)
        hook.remove()
        return features[0]

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Performs a single training step.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): A tuple containing input images and labels.
            batch_idx (int): The index of the current batch.

        Returns:
            torch.Tensor: The calculated loss for the current training step.
        """
        inputs, labels = batch
        if self.cfg.get("cutmix_or_mixup", False):
            # cutmix_or_mixup currently hits a PyTorch bug, don't use it!
            # inputs, labels = self.cutmix_or_mixup(
            #     inputs,
            #     labels,
            # )
            pass  # Keep it for potential future fixes
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)

        preds = torch.argmax(outputs, dim=1)
        if self.cfg.get("cutmix_or_mixup", False):
            labels = torch.argmax(
                labels, dim=1
            )  # If labels are one-hot encoded by mixup/cutmix
        self.train_accuracy(preds, labels)
        self.train_f1(preds, labels)

        balanced_acc = balanced_accuracy_score(
            labels.cpu().numpy(),
            preds.cpu().numpy(),
        )
        self.log(
            "train_balanced_accuracy",
            balanced_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            "train_accuracy",
            self.train_accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train_f1",
            self.train_f1,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Performs a single validation step.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): A tuple containing input images and labels.
            batch_idx (int): The index of the current batch.
            dataloader_idx (int): The index of the dataloader (e.g., 0 for main val, 1 for clean_val).

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the validation loss.
        """
        inputs, labels = batch
        outputs = self(inputs)
        val_loss = self.criterion(outputs, labels)

        name = "val" if dataloader_idx == 0 else "clean_val"

        preds = torch.argmax(outputs, dim=1)
        self.val_accuracy(preds, labels)
        self.val_f1(preds, labels)

        # Initialize the lists for this dataloader_idx if they don't exist yet.
        if dataloader_idx not in self.val_preds:
            self.val_preds[dataloader_idx] = []
            self.val_targets[dataloader_idx] = []

        self.val_preds[dataloader_idx].append(preds.cpu())
        self.val_targets[dataloader_idx].append(labels.cpu())

        balanced_acc = balanced_accuracy_score(
            labels.cpu().numpy(),
            preds.cpu().numpy(),
        )

        current_conf = confusion_matrix(
            labels.cpu(), preds.cpu(), labels=list(range(10))
        )
        if dataloader_idx not in self.val_confusion_matrices:
            self.val_confusion_matrices[dataloader_idx] = current_conf
        else:
            self.val_confusion_matrices[dataloader_idx] += current_conf

        support_count = np.bincount(labels.cpu().numpy(), minlength=10)
        if dataloader_idx not in self.val_support:
            self.val_support[dataloader_idx] = support_count
        else:
            self.val_support[dataloader_idx] += support_count

        self.log(
            f"{name}_balanced_accuracy",
            balanced_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            f"{name}_accuracy",
            self.val_accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            f"{name}_f1",
            self.val_f1,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(f"{name}_loss", val_loss, on_step=False, on_epoch=True, prog_bar=False)
        return {f"{name}_loss": val_loss}

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Performs a single test step.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): A tuple containing input images and labels.
            batch_idx (int): The index of the current batch.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the test loss.
        """
        inputs, labels = batch
        outputs = self(inputs)

        test_loss = self.criterion(outputs, labels)

        preds = torch.argmax(outputs, dim=1)
        self.test_accuracy(preds, labels)
        self.test_f1(preds, labels)

        # Store predictions and targets for detailed metrics at epoch end
        self.test_preds.append(preds.cpu())
        self.test_targets.append(labels.cpu())

        balanced_acc = balanced_accuracy_score(
            labels.cpu().numpy(),
            preds.cpu().numpy(),
        )

        if self.test_confusion_matrix is None:
            self.test_confusion_matrix = confusion_matrix(
                labels.cpu(),
                preds.cpu(),
                labels=list(range(10)),
            )
            self.test_support = np.bincount(labels.cpu().numpy(), minlength=10)
        else:
            self.test_confusion_matrix += confusion_matrix(
                labels.cpu(),
                preds.cpu(),
                labels=list(range(10)),
            )
            self.test_support += np.bincount(labels.cpu().numpy(), minlength=10)

        self.log(
            "test_balanced_accuracy",
            balanced_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            "test_accuracy",
            self.test_accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            "test_f1",
            self.test_f1,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.log("test_loss", test_loss, on_step=False, on_epoch=True, prog_bar=False)
        return {"test_loss": test_loss}

    def on_train_epoch_end(self) -> None:
        """
        Actions to perform at the end of a training epoch.
        Resets training metrics and triggers dynamic upsampling if enabled.
        """
        # Reset training metrics
        self.train_accuracy.reset()
        self.train_f1.reset()

        # --- Dynamic Upsampling Logic ---
        # Check if dynamic upsampling is enabled in the config.
        if self.cfg.get("dynamic_upsample", False):
            self.dynamic_upsample()

    def on_validation_epoch_end(self) -> None:
        """
        Actions to perform at the end of a validation epoch.
        Computes and logs per-class metrics (accuracy, F1, precision, recall, support, ratio)
        for all validation dataloaders and resets validation metrics and storage.
        """
        for dataloader_idx, conf_matrix in self.val_confusion_matrices.items():
            # Retrieve the corresponding predictions and targets.
            preds_list = self.val_preds[dataloader_idx]
            targets_list = self.val_targets[dataloader_idx]
            all_val_preds = torch.cat(preds_list)
            all_val_targets = torch.cat(targets_list)

            name = "val" if dataloader_idx == 0 else "clean_val"

            # Per-class accuracy from the confusion matrix.
            # Handle division by zero for classes with no true samples
            per_class_accuracy = np.divide(
                conf_matrix.diagonal(),
                conf_matrix.sum(axis=1),
                out=np.zeros_like(conf_matrix.diagonal(), dtype=float),
                where=conf_matrix.sum(axis=1) != 0,
            )

            # Compute per-class metrics based on the predictions for this particular set.
            y_pred = all_val_preds.numpy()
            y_true = all_val_targets.numpy()
            f1_per_class = f1_score(
                y_true, y_pred, labels=list(range(10)), average=None, zero_division=0
            )
            precision_per_class = precision_score(
                y_true, y_pred, labels=list(range(10)), average=None, zero_division=0
            )
            recall_per_class = recall_score(
                y_true, y_pred, labels=list(range(10)), average=None, zero_division=0
            )

            # Class imbalance (support) for each class already computed per dataloader.
            class_support = self.val_support[dataloader_idx]
            total_samples = class_support.sum()
            class_ratios = (
                (class_support / total_samples)
                if total_samples > 0
                else np.zeros_like(class_support)
            )

            for class_idx in range(10):
                class_name = CIFAR10_CLASSES_REVERSE[class_idx]
                self.log(
                    f"{name}_accuracy_{class_name}",
                    per_class_accuracy[class_idx],
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                )
                self.log(
                    f"{name}_f1_{class_name}",
                    f1_per_class[class_idx],
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                )
                self.log(
                    f"{name}_precision_{class_name}",
                    precision_per_class[class_idx],
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                )
                self.log(
                    f"{name}_recall_{class_name}",
                    recall_per_class[class_idx],
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                )
                self.log(
                    f"{name}_support_{class_name}",
                    class_support[class_idx],
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                )
                self.log(
                    f"{name}_class_ratio_{class_name}",
                    class_ratios[class_idx],
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                )

            max_support = class_support.max()
            min_support = class_support.min() if class_support.sum() > 0 else 0
            imbalance_ratio = (
                max_support / min_support if min_support > 0 else float("inf")
            )
            self.log(
                f"{name}_imbalance_ratio",
                imbalance_ratio,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )

        # Reset metrics and storage for the next epoch.
        self.val_accuracy.reset()
        self.val_f1.reset()
        self.val_confusion_matrices = {}
        self.val_support = {}
        self.val_preds = {}
        self.val_targets = {}

    def on_test_epoch_end(self) -> None:
        """
        Actions to perform at the end of a test epoch.
        Computes and logs per-class metrics (accuracy, F1, precision, recall, support, ratio)
        and resets test metrics and storage.
        """
        if self.test_confusion_matrix is not None and self.test_preds:
            # Combine all predictions and targets
            all_test_preds = torch.cat(self.test_preds)
            all_test_targets = torch.cat(self.test_targets)

            # Handle division by zero for classes with no true samples
            per_class_accuracy = np.divide(
                self.test_confusion_matrix.diagonal(),
                self.test_confusion_matrix.sum(axis=1),
                out=np.zeros_like(self.test_confusion_matrix.diagonal(), dtype=float),
                where=self.test_confusion_matrix.sum(axis=1) != 0,
            )

            # Calculate per-class metrics
            y_pred = all_test_preds.numpy()
            y_true = all_test_targets.numpy()

            f1_per_class = f1_score(
                y_true, y_pred, labels=list(range(10)), average=None, zero_division=0
            )
            precision_per_class = precision_score(
                y_true, y_pred, labels=list(range(10)), average=None, zero_division=0
            )
            recall_per_class = recall_score(
                y_true, y_pred, labels=list(range(10)), average=None, zero_division=0
            )

            # Class distribution
            total_samples = self.test_support.sum()
            class_ratios = (
                self.test_support / total_samples
                if total_samples > 0
                else np.zeros_like(self.test_support)
            )

            for class_idx in range(10):
                class_name = CIFAR10_CLASSES_REVERSE[class_idx]
                self.log(
                    f"test_accuracy_{class_name}",
                    per_class_accuracy[class_idx],
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                )
                self.log(
                    f"test_f1_{class_name}",
                    f1_per_class[class_idx],
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                )
                self.log(
                    f"test_precision_{class_name}",
                    precision_per_class[class_idx],
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                )
                self.log(
                    f"test_recall_{class_name}",
                    recall_per_class[class_idx],
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                )
                self.log(
                    f"test_support_{class_name}",
                    self.test_support[class_idx],
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                )
                self.log(
                    f"test_class_ratio_{class_name}",
                    class_ratios[class_idx],
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                )

            # Log imbalance metrics
            max_support = self.test_support.max()
            min_support = self.test_support.min() if self.test_support.sum() > 0 else 0
            imbalance_ratio = (
                max_support / min_support if min_support > 0 else float("inf")
            )
            self.log(
                "test_imbalance_ratio",
                imbalance_ratio,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )

        # Reset metrics and storage
        self.test_accuracy.reset()
        self.test_f1.reset()
        self.test_confusion_matrix = None
        self.test_support = None
        self.test_preds = []
        self.test_targets = []

    def configure_optimizers(self) -> Tuple[List[torch.optim.Optimizer], List[Any]]:
        """
        Configures the optimizer and learning rate schedulers.

        Uses SGD optimizer with linear warm-up followed by cosine annealing.

        Returns:
            Tuple[List[torch.optim.Optimizer], List[Any]]: A tuple containing a list
                                                            of optimizers and a list
                                                            of learning rate schedulers.
        """
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.cfg.learning_rate,
            momentum=0.9,
            weight_decay=5e-4,
        )

        # Linear warm-up for 5 epochs
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,  # Start with lr * 0.1
            end_factor=1.0,  # End with lr
            total_iters=5,  # 5 epochs
        )

        # Cosine annealing for the remaining epochs
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.cfg.epochs - 5,  # Remaining epochs after warm-up
        )

        # Combine the schedulers
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[5],  # Switch from warmup to cosine at epoch 5
        )

        return [optimizer], [scheduler]

    def dynamic_upsample(self) -> None:
        """
        Dynamically upsamples the training data by adding candidate images with
        the highest uncertainty. This method is called at the end of every training
        epoch (if enabled via self.cfg.dynamic_upsample) to add N (default 50)
        candidate examples to the training dataset.

        The acquisition score is computed as the entropy of the model's softmax
        output (i.e. higher entropy indicates higher uncertainty).

        Assumes candidate images are stored in self.cfg.extra_images_dir.
        The new images are added with the label corresponding to the minority
        class (determined from config downsample_classes).
        """
        num_to_add: int = self.cfg.get("num_dynamic_upsample", 50)
        candidate_dir: str = self.cfg.extra_images_dir
        candidate_files: List[str] = glob.glob(os.path.join(candidate_dir, "*.*"))
        if not candidate_files:
            self.print(
                f"No candidate images found in '{candidate_dir}' for dynamic upsampling.",
            )
            return

        # Use a candidate transform that converts PIL images to float tensors
        candidate_transform = v2.Compose(
            [
                v2.Resize((32, 32)),
                v2.ToTensor(),  # This converts to float and scales to [0, 1]
            ],
        )

        candidate_scores: List[float] = []
        candidate_images: List[np.ndarray] = []

        self.model.eval()
        device: torch.device = self.device

        with torch.no_grad():
            for fpath in candidate_files:
                try:
                    # Load image using PIL and apply the candidate transform.
                    pil_image = Image.open(fpath).convert("RGB")
                    image = candidate_transform(pil_image)
                except Exception as e:
                    self.print(f"Error loading image {fpath}: {e}")
                    continue

                image = image.unsqueeze(0).to(device)  # add batch dimension
                output = self(image)
                probs = F.softmax(output, dim=1)
                # Compute entropy as the acquisition score.
                entropy = -torch.sum(probs * torch.log(probs + 1e-8)).item()
                candidate_scores.append(entropy)
                # Also store the candidate image in numpy format (H x W x C)
                candidate_images.append(np.array(pil_image.resize((32, 32))))

        candidate_scores = np.array(candidate_scores)
        if len(candidate_scores) < num_to_add:
            self.print(
                f"Only {len(candidate_scores)} candidate images available; adding all.",
            )
            top_indices = np.arange(len(candidate_scores))
        else:
            top_indices = np.argsort(candidate_scores)[-num_to_add:]

        selected_images = [candidate_images[i] for i in top_indices]

        target_label: int
        if self.cfg.get("dynamic_upsample_target_class"):
            target_class_name = self.cfg.get("dynamic_upsample_target_class")
            if target_class_name in CIFAR10_CLASSES:
                target_label = CIFAR10_CLASSES[target_class_name]
            else:
                self.print(
                    f"Error: Target class '{target_class_name}' not found in CIFAR10_CLASSES."
                )
                return
        else:
            self.print(
                "Warning: No specific target class for dynamic upsampling found in config"
            )
            return

        try:
            # Accessing the underlying dataset which might be a Subset if random_split was used
            if hasattr(self.trainer.datamodule.train_dataset, "dataset"):
                train_dataset = self.trainer.datamodule.train_dataset.dataset
            else:
                train_dataset = self.trainer.datamodule.train_dataset

        except Exception as e:
            self.print(f"Could not access training dataset from datamodule: {e}")
            return

        try:
            if not selected_images:
                self.print("No images were selected for dynamic upsampling.")
                return

            new_data = np.stack(selected_images, axis=0)
            train_dataset.data = np.concatenate([train_dataset.data, new_data], axis=0)
            train_dataset.targets.extend([target_label] * len(selected_images))
            self.print(
                f"Dynamic upsampling: added {len(selected_images)} images to the training dataset for class {target_label}.",
            )
        except Exception as e:
            self.print(f"Error during dynamic upsampling: {e}")


class ClipClassifier(L.LightningModule):
    """
    A LightningModule for a CLIP-based classifier.
    This model uses a pre-trained CLIP model to classify images based on
    cosine similarity between image embeddings and text embeddings of class names.
    It is designed primarily for zero-shot classification, meaning the CLIP model
    itself is frozen by default.
    """

    def __init__(
        self, cfg: DictConfig, class_weights: Optional[torch.Tensor] = None
    ) -> None:
        """
        Initializes the ClipClassifier.

        Args:
            cfg (DictConfig): Configuration object containing model, training,
                              and data parameters.
            class_weights (Optional[torch.Tensor]): Optional tensor of class weights
                                                     to be used in the loss function.
        """
        super().__init__()
        self.cfg = cfg
        self.class_weights = class_weights

        # Load CLIP model and processor
        clip_model_name = self.cfg.get(
            "clip_model_name", "openai/clip-vit-base-patch32"
        )
        self.processor = CLIPProcessor.from_pretrained(clip_model_name)
        self.model = CLIPModel.from_pretrained(clip_model_name)

        # Freeze CLIP model parameters by default for zero-shot classification
        # If cfg.freeze_backbone is False, it implies fine-tuning the whole CLIP model.
        if self.cfg.freeze_backbone:
            self.freeze_backbone()  # Call custom freeze_backbone for CLIP

        # Prepare text embeddings for CIFAR-10 classes
        self.class_names = [CIFAR10_CLASSES_REVERSE[i] for i in range(10)]
        self.text_prompts = [
            CIFAR10_PROMPT_TEMPLATES[0].format(c) for c in self.class_names
        ]

        # Tokenize text prompts once. The actual embeddings will be computed on `on_fit_start`.
        self.text_inputs = self.processor(
            text=self.text_prompts, return_tensors="pt", padding=True
        ).input_ids
        self.text_features = None  # Will be populated in on_fit_start

        # Setup the loss. Although CLIP is zero-shot, CrossEntropyLoss is used
        # here to align with the existing training pipeline, treating similarity
        # scores as logits. If you were strictly doing zero-shot, loss is not used.
        if self.class_weights is not None:
            self.criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        elif cfg.label_smoothing:
            self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.3)
        else:
            self.criterion = torch.nn.CrossEntropyLoss()

        # Define accuracy metrics.
        self.train_accuracy = Accuracy(num_classes=10, task="multiclass")
        self.val_accuracy = Accuracy(num_classes=10, task="multiclass")
        self.test_accuracy = Accuracy(num_classes=10, task="multiclass")

        # Add F1 score metrics
        self.train_f1 = F1Score(num_classes=10, task="multiclass", average="macro")
        self.val_f1 = F1Score(num_classes=10, task="multiclass", average="macro")
        self.test_f1 = F1Score(num_classes=10, task="multiclass", average="macro")

        self.val_confusion_matrices: Dict[int, np.ndarray] = {}
        self.test_confusion_matrix: Optional[np.ndarray] = None

        # Track support for each class in different splits
        self.val_support: Dict[int, np.ndarray] = {}
        self.test_support: Optional[np.ndarray] = None

        # To track all predictions and targets for more detailed metrics at epoch end
        self.val_preds: Dict[int, List[torch.Tensor]] = {}
        self.val_targets: Dict[int, List[torch.Tensor]] = {}
        self.test_preds: List[torch.Tensor] = []
        self.test_targets: List[torch.Tensor] = []

        # CutMix/MixUp are generally not applicable for zero-shot CLIP classification
        # as it's typically applied to the input images for traditional CNN training.
        # Removing for ClipClassifier to avoid confusion and potential issues.

    def on_fit_start(self) -> None:
        """
        Called before the first training epoch.
        Moves text inputs to the correct device and computes text features once.
        """
        # Ensure text_inputs are on the same device as the model
        self.text_inputs = self.text_inputs.to(self.device)
        # Compute text features once and normalize them
        with torch.no_grad():
            self.text_features = self.model.get_text_features(self.text_inputs)
            self.text_features = self.text_features / self.text_features.norm(
                dim=-1, keepdim=True
            )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the CLIP classifier.

        Args:
            pixel_values (torch.Tensor): Input tensor of pixel values (expected to be
                                         preprocessed according to CLIP's requirements:
                                         resized to 224x224, normalized).

        Returns:
            torch.Tensor: Logits representing similarity scores between image features
                          and class text features.
        """
        # Get image features from the CLIP model
        image_features = self.model.get_image_features(pixel_values)
        # Normalize image features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Compute cosine similarity between image features and class text features
        # (batch_size, clip_embed_dim) @ (clip_embed_dim, num_classes) -> (batch_size, num_classes)
        # These similarity scores serve as logits for CrossEntropyLoss.
        logits = image_features @ self.text_features.T
        return logits

    def freeze_backbone(self) -> None:
        """
        Freezes all parameters of the CLIP model.
        """
        for param in self.model.parameters():
            param.requires_grad = False
        self.print("CLIP model backbone frozen.")

    # The training_step, validation_step, test_step, and on_epoch_end
    # methods are largely generic for classification tasks once `forward`
    # returns logits. They are copied from ResNet18Model, excluding
    # any CutMix/MixUp related logic and ResNet-specific hooks.

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)

        preds = torch.argmax(outputs, dim=1)
        self.train_accuracy(preds, labels)
        self.train_f1(preds, labels)

        balanced_acc = balanced_accuracy_score(
            labels.cpu().numpy(),
            preds.cpu().numpy(),
        )
        self.log(
            "train_balanced_accuracy",
            balanced_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            "train_accuracy",
            self.train_accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train_f1",
            self.train_f1,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int,
    ) -> Dict[str, torch.Tensor]:
        inputs, labels = batch
        outputs = self(inputs)
        val_loss = self.criterion(outputs, labels)

        name = "val" if dataloader_idx == 0 else "clean_val"

        preds = torch.argmax(outputs, dim=1)
        self.val_accuracy(preds, labels)
        self.val_f1(preds, labels)

        if dataloader_idx not in self.val_preds:
            self.val_preds[dataloader_idx] = []
            self.val_targets[dataloader_idx] = []

        self.val_preds[dataloader_idx].append(preds.cpu())
        self.val_targets[dataloader_idx].append(labels.cpu())

        balanced_acc = balanced_accuracy_score(
            labels.cpu().numpy(),
            preds.cpu().numpy(),
        )

        current_conf = confusion_matrix(
            labels.cpu(), preds.cpu(), labels=list(range(10))
        )
        if dataloader_idx not in self.val_confusion_matrices:
            self.val_confusion_matrices[dataloader_idx] = current_conf
        else:
            self.val_confusion_matrices[dataloader_idx] += current_conf

        support_count = np.bincount(labels.cpu().numpy(), minlength=10)
        if dataloader_idx not in self.val_support:
            self.val_support[dataloader_idx] = support_count
        else:
            self.val_support[dataloader_idx] += support_count

        self.log(
            f"{name}_balanced_accuracy",
            balanced_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            f"{name}_accuracy",
            self.val_accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            f"{name}_f1",
            self.val_f1,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(f"{name}_loss", val_loss, on_step=False, on_epoch=True, prog_bar=False)
        return {f"{name}_loss": val_loss}

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        inputs, labels = batch
        outputs = self(inputs)

        test_loss = self.criterion(outputs, labels)

        preds = torch.argmax(outputs, dim=1)
        self.test_accuracy(preds, labels)
        self.test_f1(preds, labels)

        self.test_preds.append(preds.cpu())
        self.test_targets.append(labels.cpu())

        balanced_acc = balanced_accuracy_score(
            labels.cpu().numpy(),
            preds.cpu().numpy(),
        )

        if self.test_confusion_matrix is None:
            self.test_confusion_matrix = confusion_matrix(
                labels.cpu(),
                preds.cpu(),
                labels=list(range(10)),
            )
            self.test_support = np.bincount(labels.cpu().numpy(), minlength=10)
        else:
            self.test_confusion_matrix += confusion_matrix(
                labels.cpu(),
                preds.cpu(),
                labels=list(range(10)),
            )
            self.test_support += np.bincount(labels.cpu().numpy(), minlength=10)

        self.log(
            "test_balanced_accuracy",
            balanced_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            "test_accuracy",
            self.test_accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            "test_f1",
            self.test_f1,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.log("test_loss", test_loss, on_step=False, on_epoch=True, prog_bar=False)
        return {"test_loss": test_loss}

    def on_train_epoch_end(self) -> None:
        self.train_accuracy.reset()
        self.train_f1.reset()

        if self.cfg.get("dynamic_upsample", False):
            self.dynamic_upsample()

    def on_validation_epoch_end(self) -> None:
        for dataloader_idx, conf_matrix in self.val_confusion_matrices.items():
            preds_list = self.val_preds[dataloader_idx]
            targets_list = self.val_targets[dataloader_idx]
            all_val_preds = torch.cat(preds_list)
            all_val_targets = torch.cat(targets_list)

            name = "val" if dataloader_idx == 0 else "clean_val"

            per_class_accuracy = np.divide(
                conf_matrix.diagonal(),
                conf_matrix.sum(axis=1),
                out=np.zeros_like(conf_matrix.diagonal(), dtype=float),
                where=conf_matrix.sum(axis=1) != 0,
            )

            y_pred = all_val_preds.numpy()
            y_true = all_val_targets.numpy()
            f1_per_class = f1_score(
                y_true, y_pred, labels=list(range(10)), average=None, zero_division=0
            )
            precision_per_class = precision_score(
                y_true, y_pred, labels=list(range(10)), average=None, zero_division=0
            )
            recall_per_class = recall_score(
                y_true, y_pred, labels=list(range(10)), average=None, zero_division=0
            )

            class_support = self.val_support[dataloader_idx]
            total_samples = class_support.sum()
            class_ratios = (
                (class_support / total_samples)
                if total_samples > 0
                else np.zeros_like(class_support)
            )

            for class_idx in range(10):
                class_name = CIFAR10_CLASSES_REVERSE[class_idx]
                self.log(
                    f"{name}_accuracy_{class_name}",
                    per_class_accuracy[class_idx],
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                )
                self.log(
                    f"{name}_f1_{class_name}",
                    f1_per_class[class_idx],
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                )
                self.log(
                    f"{name}_precision_{class_name}",
                    precision_per_class[class_idx],
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                )
                self.log(
                    f"{name}_recall_{class_name}",  # Corrected typo: was _class_idx
                    recall_per_class[class_idx],
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                )
                self.log(
                    f"{name}_support_{class_name}",
                    class_support[class_idx],
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                )
                self.log(
                    f"{name}_class_ratio_{class_name}",
                    class_ratios[class_idx],
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                )

            max_support = class_support.max()
            min_support = class_support.min() if class_support.sum() > 0 else 0
            imbalance_ratio = (
                max_support / min_support if min_support > 0 else float("inf")
            )
            self.log(
                f"{name}_imbalance_ratio",
                imbalance_ratio,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )

        self.val_accuracy.reset()
        self.val_f1.reset()
        self.val_confusion_matrices = {}
        self.val_support = {}
        self.val_preds = {}
        self.val_targets = {}

    def on_test_epoch_end(self) -> None:
        """
        Actions to perform at the end of a test epoch.
        Computes and logs per-class metrics (accuracy, F1, precision, recall, support, ratio)
        and resets test metrics and storage.
        """
        if self.test_confusion_matrix is not None and self.test_preds:
            # Combine all predictions and targets
            all_test_preds = torch.cat(self.test_preds)
            all_test_targets = torch.cat(self.test_targets)

            # Handle division by zero for classes with no true samples
            per_class_accuracy = np.divide(
                self.test_confusion_matrix.diagonal(),
                self.test_confusion_matrix.sum(axis=1),
                out=np.zeros_like(self.test_confusion_matrix.diagonal(), dtype=float),
                where=self.test_confusion_matrix.sum(axis=1) != 0,
            )

            # Calculate per-class metrics
            y_pred = all_test_preds.numpy()
            y_true = all_test_targets.numpy()

            f1_per_class = f1_score(
                y_true, y_pred, labels=list(range(10)), average=None, zero_division=0
            )
            precision_per_class = precision_score(
                y_true, y_pred, labels=list(range(10)), average=None, zero_division=0
            )
            recall_per_class = recall_score(
                y_true, y_pred, labels=list(range(10)), average=None, zero_division=0
            )

            # Class distribution
            total_samples = self.test_support.sum()
            class_ratios = (
                self.test_support / total_samples
                if total_samples > 0
                else np.zeros_like(self.test_support)
            )

            for class_idx in range(10):
                class_name = CIFAR10_CLASSES_REVERSE[class_idx]
                self.log(
                    f"test_accuracy_{class_name}",
                    per_class_accuracy[class_idx],
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                )
                self.log(
                    f"test_f1_{class_name}",
                    f1_per_class[class_idx],
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                )
                self.log(
                    f"test_precision_{class_name}",
                    precision_per_class[class_idx],
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                )
                self.log(
                    f"test_recall_{class_name}",
                    recall_per_class[class_idx],
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                )
                self.log(
                    f"test_support_{class_name}",
                    self.test_support[class_idx],
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                )
                self.log(
                    f"test_class_ratio_{class_name}",
                    class_ratios[class_idx],
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                )

            # Log imbalance metrics
            max_support = self.test_support.max()
            min_support = self.test_support.min() if self.test_support.sum() > 0 else 0
            imbalance_ratio = (
                max_support / min_support if min_support > 0 else float("inf")
            )
            self.log(
                "test_imbalance_ratio",
                imbalance_ratio,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )

        # Reset metrics and storage
        self.test_accuracy.reset()
        self.test_f1.reset()
        self.test_confusion_matrix = None
        self.test_support = None
        self.test_preds = []
        self.test_targets = []

    def configure_optimizers(self) -> Tuple[List[torch.optim.Optimizer], List[Any]]:
        """
        Configures the optimizer and learning rate schedulers.

        Uses SGD optimizer with linear warm-up followed by cosine annealing.

        Returns:
            Tuple[List[torch.optim.Optimizer], List[Any]]: A tuple containing a list
                                                            of optimizers and a list
                                                            of learning rate schedulers.
        """
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.cfg.learning_rate,
            momentum=0.9,
            weight_decay=5e-4,
        )

        # Linear warm-up for 5 epochs
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,  # Start with lr * 0.1
            end_factor=1.0,  # End with lr
            total_iters=5,  # 5 epochs
        )

        # Cosine annealing for the remaining epochs
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.cfg.epochs - 5,  # Remaining epochs after warm-up
        )

        # Combine the schedulers
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[5],  # Switch from warmup to cosine at epoch 5
        )

        return [optimizer], [scheduler]

    # In model.py, within class ClipClassifier:

    def dynamic_upsample(self) -> None:
        """
        Dynamically upsamples the training data by adding candidate images with
        the highest uncertainty. This method is called at the end of every training
        epoch (if enabled via self.cfg.dynamic_upsample) to add N (default 50)
        candidate examples to the training dataset.

        The acquisition score is computed as the entropy of the model's softmax
        output (i.e. higher entropy indicates higher uncertainty).

        Assumes candidate images are stored in self.cfg.extra_images_dir.
        The new images are added with the label corresponding to the minority
        class (determined from config dynamic_upsample_target_class).
        Images added to train_dataset.data are 32x32 numpy arrays.
        """
        # import numpy as np # Already imported at module level
        # from PIL import Image # Already imported at module level

        num_to_add: int = self.cfg.get("num_dynamic_upsample", 50)
        candidate_dir: str = self.cfg.extra_images_dir
        candidate_files: List[str] = glob.glob(os.path.join(candidate_dir, "*.*"))
        if not candidate_files:
            self.print(
                f"No candidate images found in '{candidate_dir}' for dynamic upsampling.",
            )
            return

        candidate_scores: List[float] = []
        # Store PIL images resized to 32x32 for adding to dataset.data later
        candidate_images_for_dataset: List[np.ndarray] = []

        self.model.eval()  # Ensure model is in eval mode
        device: torch.device = self.device

        with torch.no_grad():
            for fpath in candidate_files:
                try:
                    pil_image = Image.open(fpath).convert("RGB")
                    # For inference with CLIP model, process with self.processor
                    # The processor typically handles resizing to 224x224, to_tensor, and normalization.
                    inputs_for_clip = self.processor(
                        images=pil_image, return_tensors="pt", padding=True
                    ).to(device)
                    pixel_values_for_clip = inputs_for_clip.pixel_values

                except Exception as e:
                    self.print(
                        f"Error loading/processing image {fpath} for CLIP inference: {e}"
                    )
                    continue

                # output = self.model(pixel_values_for_clip) # This would call CLIPModel's forward
                # Call self.forward for consistency with training steps, which gets image_features
                output_logits = self(
                    pixel_values_for_clip
                )  # self.forward expects preprocessed pixel_values

                probs = F.softmax(output_logits, dim=1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-8)).item()
                candidate_scores.append(entropy)

                # Store the image as a 32x32 numpy array for adding to the dataset
                # The main training dataloader's transform will handle processing this 32x32 image for CLIP later
                img_32x32_for_dataset = pil_image.resize((32, 32))
                candidate_images_for_dataset.append(np.array(img_32x32_for_dataset))

        if not candidate_scores:
            self.print("No candidate scores generated. Skipping dynamic upsample.")
            return

        candidate_scores_np = np.array(candidate_scores)  # Corrected variable name
        if len(candidate_scores_np) < num_to_add:
            self.print(
                f"Only {len(candidate_scores_np)} candidate images available; adding all.",
            )
            top_indices = np.arange(len(candidate_scores_np))
        else:
            top_indices = np.argsort(candidate_scores_np)[-num_to_add:]

        selected_images_for_dataset = [
            candidate_images_for_dataset[i] for i in top_indices
        ]

        target_label: int
        target_class_name_cfg = self.cfg.get("dynamic_upsample_target_class")
        if target_class_name_cfg:
            if target_class_name_cfg in CIFAR10_CLASSES:
                target_label = CIFAR10_CLASSES[target_class_name_cfg]
            else:
                self.print(
                    f"Error: Target class '{target_class_name_cfg}' not found in CIFAR10_CLASSES."
                )
                return
        else:
            self.print(
                "Error: No specific target class for dynamic upsampling found in config."
            )
            return  # Changed from exit()

        try:
            # Accessing the underlying dataset
            # Ensure self.trainer and self.trainer.datamodule are available
            if (
                self.trainer is None
                or self.trainer.datamodule is None
                or self.trainer.datamodule.train_dataset is None
            ):
                self.print(
                    "Trainer, datamodule or train_dataset not available for dynamic upsampling."
                )
                return

            train_dataset_obj = self.trainer.datamodule.train_dataset
            if hasattr(train_dataset_obj, "dataset") and isinstance(
                train_dataset_obj.dataset, DownsampledCIFAR10
            ):
                underlying_train_dataset = train_dataset_obj.dataset
            elif isinstance(train_dataset_obj, DownsampledCIFAR10):
                underlying_train_dataset = train_dataset_obj
            else:
                self.print("Could not access underlying DownsampledCIFAR10 dataset.")
                return

        except Exception as e:
            self.print(f"Could not access training dataset from datamodule: {e}")
            return

        try:
            if not selected_images_for_dataset:
                self.print("No images were selected for dynamic upsampling.")
                return

            new_data_np = np.stack(selected_images_for_dataset, axis=0)
            underlying_train_dataset.data = np.concatenate(
                [underlying_train_dataset.data, new_data_np], axis=0
            )
            underlying_train_dataset.targets.extend(
                [target_label] * len(selected_images_for_dataset)
            )
            self.print(
                f"Dynamic upsampling: added {len(selected_images_for_dataset)} images to the training dataset for class {target_label} ({CIFAR10_CLASSES_REVERSE[target_label]}).",
            )
            # After adding data, the DownsampledCIFAR10's internal normalization might need update IF model_type is resnet18
            if self.cfg.model_type == "resnet18":
                underlying_train_dataset.update_normalization()

        except Exception as e:
            self.print(f"Error during dynamic upsampling data addition: {e}")
