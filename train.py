import lightning as L
from lightning.pytorch.loggers import CSVLogger
from omegaconf import DictConfig
import hydra
from data import CIFAR10DataModule
from model import ResNet18Model
import torch


# Hydra Integration
@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig):
    data_module = CIFAR10DataModule(cfg)
    model = ResNet18Model(cfg)
    torch.set_float32_matmul_precision('high')

    logger = CSVLogger(save_dir="logs/", name="cifar10")

    trainer = L.Trainer(
        max_epochs=cfg.epochs, 
        accelerator="auto", 
        devices="auto", 
        logger=logger, 
        log_every_n_steps=1,
    )

    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)


if __name__ == "__main__":
    main()
