import hydra
import lightning as L
import torch
import wandb
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from omegaconf import DictConfig, OmegaConf

from data import CIFAR10DataModule
from model import ResNet18Model


@hydra.main(config_path="config", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    L.seed_everything(cfg.seed, workers=True)
    data_module = CIFAR10DataModule(cfg)
    data_module.prepare_data()
    model = ResNet18Model(cfg, class_weights=data_module.class_weights)
    if cfg.compile:
        model = torch.compile(model)
    torch.set_float32_matmul_precision("medium")

    csv_logger = CSVLogger(save_dir="logs/", name="cifar10")
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    wandb_logger = WandbLogger(project="cifar10_project", log_model="all")
    wandb.init(config=config_dict, project="cifar10_project", name=cfg.name)

    trainer = L.Trainer(
        max_epochs=cfg.epochs,
        accelerator="auto",
        devices="auto",
        logger=[csv_logger, wandb_logger],
        log_every_n_steps=1,
    )

    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)


if __name__ == "__main__":
    main()
