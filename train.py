import hydra
import lightning as L
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
from typing import Dict, Any

import wandb
from data import CIFAR10DataModule
from model import ResNet18Model
from utils import (
    visualize_feature_maps,
    visualize_filters,
    CIFAR10_CLASSES,
    set_reproducibility,
    prepare_fine_tune,
)


@hydra.main(config_path="config", config_name="config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    """
    Main training and evaluation function for the CIFAR-10 model.

    This function orchestrates the entire machine learning pipeline:
    1. Sets reproducibility for consistent results.
    2. Initializes the CIFAR-10 data module, preparing datasets.
    3. Initializes the ResNet18 model, optionally loading from a checkpoint.
    4. Compiles the model if specified in the configuration for performance.
    5. Sets PyTorch matrix multiplication precision.
    6. Initializes Weights & Biases logging for experiment tracking.
    7. Conducts the main training and testing phase.
    8. Optionally performs a fine-tuning phase on real data with modified settings.
    9. Optionally visualizes feature maps and convolutional filters of the trained model.

    Args:
        cfg (DictConfig): A Hydra configuration object containing all
                         parameters for data, model, training, and logging.
    """
    set_reproducibility(cfg)
    data_module = CIFAR10DataModule(cfg)
    data_module.prepare_data()
    model: L.LightningModule = ResNet18Model(
        cfg, class_weights=data_module.class_weights
    )

    # Read initial checkpoint if provided
    if cfg.get("checkpoint_path"):
        model = ResNet18Model.load_from_checkpoint(
            checkpoint_path=cfg.checkpoint_path,
            cfg=cfg,
            class_weights=data_module.class_weights,
        )

    if cfg.compile:
        model = torch.compile(model)
    torch.set_float32_matmul_precision("medium")

    config_dict: Dict[str, Any] = OmegaConf.to_container(cfg, resolve=True)

    # Initial phase of training
    if not cfg.finetune_on_checkpoint:
        wandb_logger = WandbLogger(project=cfg.project, log_model=True)
        wandb.init(config=config_dict, project=cfg.project, name=cfg.name)

        trainer = L.Trainer(
            max_epochs=cfg.epochs,
            accelerator="auto",
            devices="auto",
            logger=[wandb_logger],
            log_every_n_steps=1,
            check_val_every_n_epoch=1,
        )

        trainer.fit(model, datamodule=data_module)
        trainer.test(model, datamodule=data_module)

    # Fine-tuning on real data
    if cfg.finetune_on_checkpoint and not cfg.fine_tune_on_real_data:
        print(
            "To enable finetune_on_checkpoint, you must also enable fine_tune_on_real_data"
        )
    if cfg.fine_tune_on_real_data:
        print(cfg)
        wandb_logger2 = WandbLogger(project=cfg.project, log_model=True)
        wandb.init(
            config=config_dict,
            project=cfg.project,
            name=cfg.name + "_fine_tuned",
            resume="allow",
        )
        prepare_fine_tune(cfg)

        data_module = CIFAR10DataModule(cfg)
        data_module.prepare_data()

        fine_tune_trainer = L.Trainer(
            max_epochs=cfg.epochs,
            accelerator="auto",
            devices="auto",
            logger=[wandb_logger2],
            log_every_n_steps=1,
            check_val_every_n_epoch=1,
        )
        if not cfg.finetune_on_checkpoint:
            model = ResNet18Model(cfg, class_weights=data_module.class_weights)
        else:
            if not cfg.checkpoint_path:
                print("\n\nYou must provide a checkpoint path!\n\n")
                return
            model = ResNet18Model.load_from_checkpoint(
                checkpoint_path=cfg.checkpoint_path,
                cfg=cfg,
                class_weights=data_module.class_weights,
            )
        if cfg.freeze_backbone:
            model.freeze_backbone()
        if cfg.compile:
            model = torch.compile(model)
        fine_tune_trainer.fit(model, datamodule=data_module)
        fine_tune_trainer.test(model, datamodule=data_module)

    if cfg.get("visualize_trained_model", False):
        feature_map_img, resized_img = visualize_feature_maps(
            model, data_module, return_image=True
        )
        filter_img = visualize_filters(model, return_image=True)

        if feature_map_img is not None:
            wandb.log(
                {"feature_maps": wandb.Image(feature_map_img, caption="Feature Maps")}
            )
        if resized_img is not None:
            wandb.log(
                {"resized_img": wandb.Image(resized_img, caption="Resized Image")}
            )
        if filter_img is not None:
            wandb.log({"conv_filters": wandb.Image(filter_img, caption="Conv Filters")})


if __name__ == "__main__":
    main()
