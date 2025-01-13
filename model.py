import torch
import torchvision
import lightning as L
from omegaconf import DictConfig
from torchmetrics import Accuracy
from sklearn.metrics import balanced_accuracy_score


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

        # Compute balanced accuracy using scikit-learn
        balanced_acc = balanced_accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())
        self.log("val_balanced_accuracy", balanced_acc, on_epoch=True, prog_bar=True)
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

        # Log metrics
        self.log("test_balanced_accuracy", balanced_acc, on_epoch=True, prog_bar=True)
        self.log("test_accuracy", self.test_accuracy, on_epoch=True, prog_bar=True)
        self.log("test_loss", test_loss, on_epoch=True, prog_bar=True)
        return {"test_loss": test_loss}

    def on_train_epoch_end(self):
        self.train_accuracy.reset()

    def on_validation_epoch_end(self):
        self.val_accuracy.reset()

    def on_test_epoch_end(self):
        self.test_accuracy.reset()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.cfg.learning_rate)
