import lightning as L
from lightning.pytorch.loggers import CSVLogger
from omegaconf import DictConfig
import hydra
from data import CIFAR10DataModule  # Assuming CIFAR10DataModule is implemented in `data.py`
from model import ResNet18Model  # Assuming ResNet18Model is implemented in `model.py`
import torch
import pandas as pd
import matplotlib.pyplot as plt
from utils import plot_metrics


# Hydra Integration
@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig):
    # Initialize data module and model
    data_module = CIFAR10DataModule(cfg)
    model = ResNet18Model(cfg)
    torch.set_float32_matmul_precision('high')

    # Set up CSV logger
    logger = CSVLogger(save_dir="logs/", name="cifar10")

    # Set up trainer
    trainer = L.Trainer(
        max_epochs=cfg.epochs, 
        accelerator="auto", 
        devices="auto", 
        logger=logger, 
        log_every_n_steps=1,
    )

    # Train and test the model
    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)

    # Plot metrics
    plot_metrics(logger.log_dir)


if __name__ == "__main__":
    main()
