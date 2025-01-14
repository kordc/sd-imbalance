import torch
import torchvision
import lightning as L
from omegaconf import DictConfig
from torchmetrics import Accuracy
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
import numpy as np
from utils import CIFAR10_CLASSES_REVERSE


class ResNet18Model(L.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.model = torchvision.models.resnet18(num_classes=10)
        self.criterion = torch.nn.CrossEntropyLoss()

        # Define accuracy metric
        self.train_accuracy = Accuracy(num_classes=10, task="multiclass")
        self.val_accuracy = Accuracy(num_classes=10, task="multiclass")
        self.test_accuracy = Accuracy(num_classes=10, task="multiclass")

        self.val_confusion_matrix = None
        self.test_confusion_matrix = None

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)

        # Update accuracy
        preds = torch.argmax(outputs, dim=1)
        self.train_accuracy(preds, labels)
        balanced_acc = balanced_accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())
        self.log("train_balanced_accuracy", balanced_acc, on_epoch=True, prog_bar=True)
        self.log("train_accuracy", self.train_accuracy, on_epoch=True, prog_bar=True)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        val_loss = self.criterion(outputs, labels)

        # Update accuracy
        preds = torch.argmax(outputs, dim=1)
        self.val_accuracy(preds, labels)

        # Accumulate confusion matrix
        if self.val_confusion_matrix is None:
            self.val_confusion_matrix = confusion_matrix(labels.cpu(), preds.cpu(), labels=range(10))
        else:
            self.val_confusion_matrix += confusion_matrix(labels.cpu(), preds.cpu(), labels=range(10))

        self.log("val_accuracy", self.val_accuracy, on_epoch=True, prog_bar=True)
        self.log("val_loss", val_loss, on_epoch=True, prog_bar=True)
        return {"val_loss": val_loss}

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)

        # Compute loss
        test_loss = self.criterion(outputs, labels)

        # Compute accuracy and balanced accuracy
        preds = torch.argmax(outputs, dim=1)
        self.test_accuracy(preds, labels)
        balanced_acc = balanced_accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())

        # Accumulate confusion matrix
        if self.test_confusion_matrix is None:
            self.test_confusion_matrix = confusion_matrix(labels.cpu(), preds.cpu(), labels=range(10))
        else:
            self.test_confusion_matrix += confusion_matrix(labels.cpu(), preds.cpu(), labels=range(10))

        # Log metrics
        self.log("test_balanced_accuracy", balanced_acc, on_epoch=True, prog_bar=True)
        self.log("test_accuracy", self.test_accuracy, on_epoch=True, prog_bar=True)
        self.log("test_loss", test_loss, on_epoch=True, prog_bar=False)
        return {"test_loss": test_loss}

    def on_train_epoch_end(self):
        self.train_accuracy.reset()

    def on_validation_epoch_end(self):
        # Compute per-class accuracy
        if self.val_confusion_matrix is not None:
            per_class_accuracy = self.val_confusion_matrix.diagonal() / self.val_confusion_matrix.sum(axis=1)
            for class_idx, accuracy in enumerate(per_class_accuracy):
                class_name = CIFAR10_CLASSES_REVERSE[class_idx]
                self.log(f"val_accuracy_{class_name}", accuracy, on_epoch=True, prog_bar=True)

        # Reset metrics
        self.val_accuracy.reset()
        self.val_confusion_matrix = None

    def on_test_epoch_end(self):
        # Compute per-class accuracy
        if self.test_confusion_matrix is not None:
            per_class_accuracy = self.test_confusion_matrix.diagonal() / self.test_confusion_matrix.sum(axis=1)
            for class_idx, accuracy in enumerate(per_class_accuracy):
                class_name = CIFAR10_CLASSES_REVERSE[class_idx]
                self.log(f"test_accuracy_{class_name}", accuracy, on_epoch=True, prog_bar=True)

        # Reset metrics
        self.test_accuracy.reset()
        self.test_confusion_matrix = None

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.cfg.learning_rate, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.cfg.epochs)
        return [optimizer], [scheduler]
