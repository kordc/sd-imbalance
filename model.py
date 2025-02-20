import os
import glob
import lightning as L
import torch
import torchvision
import torch.nn.functional as F
from omegaconf import DictConfig
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from torchmetrics import Accuracy
from torchvision import transforms

from utils import CIFAR10_CLASSES_REVERSE, CIFAR10_CLASSES  # if needed

class ResNet18Model(L.LightningModule):
    def __init__(self, cfg: DictConfig, class_weights=None):
        super().__init__()
        self.cfg = cfg
        self.model = torchvision.models.resnet18(num_classes=10)
        self.class_weights = class_weights

        # Setup the loss: use class weights, label smoothing, or default cross entropy.
        if self.class_weights is not None:
            self.criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        elif cfg.label_smoothing:
            self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
        else:
            self.criterion = torch.nn.CrossEntropyLoss()

        # Define accuracy metrics.
        self.train_accuracy = Accuracy(num_classes=10, task="multiclass")
        self.val_accuracy = Accuracy(num_classes=10, task="multiclass")
        self.test_accuracy = Accuracy(num_classes=10, task="multiclass")

        self.val_confusion_matrix = None
        self.test_confusion_matrix = None

    def forward(self, x):
        return self.model(x)

    def visualize_feature_maps(self, x):
        """
        Registers a forward hook on a chosen layer (here layer1)
        and returns the feature maps produced for the input x.
        """
        features = []
        def hook_fn(module, input, output):
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
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)

        preds = torch.argmax(outputs, dim=1)
        self.train_accuracy(preds, labels)
        balanced_acc = balanced_accuracy_score(
            labels.cpu().numpy(),
            preds.cpu().numpy(),
        )
        self.log(
            "train_balanced_accuracy",
            balanced_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train_accuracy",
            self.train_accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        val_loss = self.criterion(outputs, labels)

        preds = torch.argmax(outputs, dim=1)
        self.val_accuracy(preds, labels)
        balanced_acc = balanced_accuracy_score(
            labels.cpu().numpy(),
            preds.cpu().numpy(),
        )

        if self.val_confusion_matrix is None:
            self.val_confusion_matrix = confusion_matrix(
                labels.cpu(),
                preds.cpu(),
                labels=range(10),
            )
        else:
            self.val_confusion_matrix += confusion_matrix(
                labels.cpu(),
                preds.cpu(),
                labels=range(10),
            )

        self.log(
            "val_balanced_accuracy",
            balanced_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val_accuracy",
            self.val_accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"val_loss": val_loss}

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)

        test_loss = self.criterion(outputs, labels)

        preds = torch.argmax(outputs, dim=1)
        self.test_accuracy(preds, labels)
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
        else:
            self.test_confusion_matrix += confusion_matrix(
                labels.cpu(),
                preds.cpu(),
                labels=range(10),
            )

        self.log(
            "test_balanced_accuracy",
            balanced_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "test_accuracy",
            self.test_accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log("test_loss", test_loss, on_step=False, on_epoch=True, prog_bar=False)
        return {"test_loss": test_loss}

    def on_train_epoch_end(self):
        # Reset training accuracy metric.
        self.train_accuracy.reset()

        # --- Dynamic Upsampling Logic ---
        # Check if dynamic upsampling is enabled in the config.
        if self.cfg.get("dynamic_upsample", False):
            self.dynamic_upsample()

    def on_validation_epoch_end(self):
        if self.val_confusion_matrix is not None:
            per_class_accuracy = (
                self.val_confusion_matrix.diagonal()
                / self.val_confusion_matrix.sum(axis=1)
            )
            for class_idx, accuracy in enumerate(per_class_accuracy):
                class_name = CIFAR10_CLASSES_REVERSE[class_idx]
                self.log(
                    f"val_accuracy_{class_name}",
                    accuracy,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                )

        self.val_accuracy.reset()
        self.val_confusion_matrix = None

    def on_test_epoch_end(self):
        if self.test_confusion_matrix is not None:
            per_class_accuracy = (
                self.test_confusion_matrix.diagonal()
                / self.test_confusion_matrix.sum(axis=1)
            )
            for class_idx, accuracy in enumerate(per_class_accuracy):
                class_name = CIFAR10_CLASSES_REVERSE[class_idx]
                self.log(
                    f"test_accuracy_{class_name}",
                    accuracy,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                )

        self.test_accuracy.reset()
        self.test_confusion_matrix = None

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

    def dynamic_upsample(self):
        """
        Dynamically upsamples the training data by adding candidate images with
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
        import torch.nn.functional as F
        from PIL import Image
        from torchvision import transforms
    
        num_to_add = self.cfg.get("num_dynamic_upsample", 50)
        candidate_dir = self.cfg.extra_images_dir
        candidate_files = glob.glob(os.path.join(candidate_dir, "*.*"))
        if not candidate_files:
            self.print(f"No candidate images found in '{candidate_dir}' for dynamic upsampling.")
            return
    
        # Use a candidate transform that converts PIL images to float tensors
        candidate_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),  # This converts to float and scales to [0, 1]
        ])
    
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
            self.print(f"Only {len(candidate_scores)} candidate images available; adding all.")
            top_indices = np.arange(len(candidate_scores))
        else:
            top_indices = np.argsort(candidate_scores)[-num_to_add:]
    
        selected_images = [candidate_images[i] for i in top_indices]
    
        # Determine the target label for these dynamic examples.
        if isinstance(self.cfg.downsample_class, str):
            from utils import CIFAR10_CLASSES  # assuming this conversion exists in your utils
            target_label = CIFAR10_CLASSES[self.cfg.downsample_class]
        else:
            target_label = self.cfg.downsample_class
    
        try:
            train_dataset = self.trainer.datamodule.train_dataset.dataset
        except Exception as e:
            self.print("Could not access training dataset from datamodule.")
            return
    
        try:
            new_data = np.stack(selected_images, axis=0)
            train_dataset.data = np.concatenate([train_dataset.data, new_data], axis=0)
            train_dataset.targets.extend([target_label] * len(selected_images))
            self.print(f"Dynamic upsampling: added {len(selected_images)} images to the training dataset for class {target_label}.")
        except Exception as e:
            self.print(f"Error during dynamic upsampling: {e}")