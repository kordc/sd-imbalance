import torch.nn as nn
import torchvision.models as models
import hydra
from omegaconf import DictConfig


class ResNetModel(nn.Module):
    def __init__(self, config: DictConfig):
        super(ResNetModel, self).__init__()
        if config.model.name == 'resnet18':
            self.model = models.resnet18(pretrained=config.model.pretrained)
        elif config.model.name == 'resnet50':
            self.model = models.resnet50(pretrained=config.model.pretrained)
        else:
            raise ValueError(f"Unsupported model {config.model.name}")

        # Modify the final layer for CIFAR-10 or ImageNet
        self.model.fc = nn.Linear(
            self.model.fc.in_features, config.dataset.num_classes)

    def forward(self, x):
        return self.model(x)
