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

from utils import CIFAR10_CLASSES_REVERSE  # if needed


class ResNet18Model(L.LightningModule):
    def __init__(self, cfg: DictConfig, class_weights=None) -> None:
        super().__init__()
        self.cfg = cfg
        # self.model = torchvision.models.resnet18(num_classes=10)
        self.model = timm.create_model("resnet18", num_classes=10, pretrained=False)
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
        )
        self.maxpool = nn.Identity()
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

        self.val_confusion_matrices = {}
        self.test_confusion_matrix = None

        # Track support for each class in different splits
        self.val_support = {}
        self.test_support = None

        # To track all predictions and targets for more detailed metrics at epoch end
        self.val_preds = []
        self.val_targets = []
        self.test_preds = []
        self.test_targets = []

        self.cutmix = v2.CutMix(num_classes=10)
        self.mixup = v2.MixUp(num_classes=10)
        self.cutmix_or_mixup = v2.RandomChoice([self.cutmix, self.mixup])

    def forward(self, x):
        return self.model(x)
    
    def custom_cutmix_cat(self, inputs, labels, beta=1.0):
        """
        Applies CutMix only for images belonging to the cat class (index 3).
        For each cat image in the batch, a different cat image is selected (if available)
        and a random patch from it is pasted over the original image.
        Since both images are cats, the label remains unchanged.
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

    def freeze_backbone(self):
        """Freezes the backbone of the model, allowing only the final layer to be trained."""
        # Freeze all backbone parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Unfreeze the classification fc (last layer)
        for param in self.model.fc.parameters():
            param.requires_grad = True

    def visualize_feature_maps(self, x):
        """Registers a forward hook on a chosen layer (here layer1)
        and returns the feature maps produced for the input x.
        """
        features = []

        def hook_fn(module, input, output) -> None:
            features.append(output)

        # Choose layer1 for visualization (can be changed as needed)
        hook = self.model.layer1.register_forward_hook(hook_fn)
        self.eval()
        with torch.no_grad():
            _ = self(x)
        hook.remove()
        return features[0]

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        if self.cfg.get("cutmix_cat_only", False):
            inputs, labels = self.custom_cutmix_cat(inputs, labels)
        elif self.cfg.get("cutmix_or_mixup", False):
            inputs, labels = self.cutmix_or_mixup(
                inputs,
                labels,
            )
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)

        preds = torch.argmax(outputs, dim=1)
        if self.cfg.get("cutmix_or_mixup", False) and not self.cfg.get("cutmix_cat_only", False):
            labels = torch.argmax(labels, dim=1)
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

    def validation_step(self, batch, batch_idx, dataloader_idx):
        inputs, labels = batch
        outputs = self(inputs)
        val_loss = self.criterion(outputs, labels)

        name = "val" if dataloader_idx == 0 else "clean_val"

        preds = torch.argmax(outputs, dim=1)
        self.val_accuracy(preds, labels)
        self.val_f1(preds, labels)

        # Store predictions and targets for detailed metrics at epoch end
        self.val_preds.append(preds.cpu())
        self.val_targets.append(labels.cpu())

        balanced_acc = balanced_accuracy_score(
            labels.cpu().numpy(),
            preds.cpu().numpy(),
        )

        # Update confusion matrix for this dataloader.
        current_conf = confusion_matrix(labels.cpu(), preds.cpu(), labels=range(10))
        if dataloader_idx not in self.val_confusion_matrices:
            self.val_confusion_matrices[dataloader_idx] = current_conf
        else:
            self.val_confusion_matrices[dataloader_idx] += current_conf

        # Track support per class
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

    def test_step(self, batch, batch_idx):
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
                labels=range(10),
            )
            self.test_support = np.bincount(labels.cpu().numpy(), minlength=10)
        else:
            self.test_confusion_matrix += confusion_matrix(
                labels.cpu(),
                preds.cpu(),
                labels=range(10),
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
        # Reset training metrics
        self.train_accuracy.reset()
        self.train_f1.reset()

        # --- Dynamic Upsampling Logic ---
        # Check if dynamic upsampling is enabled in the config.
        if self.cfg.get("dynamic_upsample", False):
            self.dynamic_upsample()

    def on_validation_epoch_end(self) -> None:
        # Combine all predictions and targets
        all_val_preds = (
            torch.cat(self.val_preds) if self.val_preds else torch.tensor([])
        )
        all_val_targets = (
            torch.cat(self.val_targets) if self.val_targets else torch.tensor([])
        )

        for dataloader_idx, conf_matrix in self.val_confusion_matrices.items():
            name = "val" if dataloader_idx == 0 else "clean_val"

            # Per-class metrics
            per_class_accuracy = conf_matrix.diagonal() / conf_matrix.sum(axis=1)

            # Get per-class support
            class_support = self.val_support[dataloader_idx]

            # Calculate per-class F1, precision, and recall
            y_pred = all_val_preds.numpy()
            y_true = all_val_targets.numpy()

            # Per-class F1, precision, and recall
            f1_per_class = f1_score(
                y_true, y_pred, labels=range(10), average=None, zero_division=0
            )
            precision_per_class = precision_score(
                y_true, y_pred, labels=range(10), average=None, zero_division=0
            )
            recall_per_class = recall_score(
                y_true, y_pred, labels=range(10), average=None, zero_division=0
            )

            # Class imbalance ratio (percentage of each class)
            total_samples = class_support.sum()
            class_ratios = (
                class_support / total_samples
                if total_samples > 0
                else np.zeros_like(class_support)
            )

            # Log all metrics
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

            # Log imbalance metrics
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

        # Reset metrics and storage
        self.val_accuracy.reset()
        self.val_f1.reset()
        self.val_confusion_matrices = {}
        self.val_support = {}
        self.val_preds = []
        self.val_targets = []

    def on_test_epoch_end(self) -> None:
        if self.test_confusion_matrix is not None and self.test_preds:
            # Combine all predictions and targets
            all_test_preds = torch.cat(self.test_preds)
            all_test_targets = torch.cat(self.test_targets)

            per_class_accuracy = (
                self.test_confusion_matrix.diagonal()
                / self.test_confusion_matrix.sum(axis=1)
            )

            # Calculate per-class metrics
            y_pred = all_test_preds.numpy()
            y_true = all_test_targets.numpy()

            f1_per_class = f1_score(
                y_true, y_pred, labels=range(10), average=None, zero_division=0
            )
            precision_per_class = precision_score(
                y_true, y_pred, labels=range(10), average=None, zero_division=0
            )
            recall_per_class = recall_score(
                y_true, y_pred, labels=range(10), average=None, zero_division=0
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

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.cfg.learning_rate,
            momentum=0.9,
            weight_decay=5e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.cfg.epochs,
        )
        return [optimizer], [scheduler]

    def dynamic_upsample(self) -> None:
        """Dynamically upsamples the training data by adding candidate images with
        the highest uncertainty. This method is called at the end of every training
        epoch (if enabled via self.cfg.dynamic_upsample) to add N (default 50)
        candidate examples to the training dataset.

        The acquisition score is computed as the entropy of the model's softmax
        output (i.e. higher entropy indicates higher uncertainty).

        Assumes candidate images are stored in self.cfg.extra_images_dir.
        The new images are added with the label corresponding to the minority
        class (self.cfg.downsample_class).
        """
        import numpy as np
        from PIL import Image

        num_to_add = self.cfg.get("num_dynamic_upsample", 50)
        candidate_dir = self.cfg.extra_images_dir
        candidate_files = glob.glob(os.path.join(candidate_dir, "*.*"))
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

        candidate_scores = []
        candidate_images = []

        self.model.eval()
        device = self.device

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
                output = self.model(image)
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

        # Determine the target label for these dynamic examples.
        if isinstance(self.cfg.downsample_class, str):
            from utils import (
                CIFAR10_CLASSES,
            )  # assuming this conversion exists in your utils

            target_label = CIFAR10_CLASSES[self.cfg.downsample_class]
        else:
            target_label = self.cfg.downsample_class

        try:
            train_dataset = self.trainer.datamodule.train_dataset.dataset
        except Exception:
            self.print("Could not access training dataset from datamodule.")
            return

        try:
            new_data = np.stack(selected_images, axis=0)
            train_dataset.data = np.concatenate([train_dataset.data, new_data], axis=0)
            train_dataset.targets.extend([target_label] * len(selected_images))
            self.print(
                f"Dynamic upsampling: added {len(selected_images)} images to the training dataset for class {target_label}.",
            )
        except Exception as e:
            self.print(f"Error during dynamic upsampling: {e}")
