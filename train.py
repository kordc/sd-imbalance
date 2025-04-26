import hydra
import lightning as L
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf

import wandb
from data import CIFAR10DataModule
from model import ResNet18Model
from utils import (
    visualize_feature_maps,
    visualize_filters,
    CIFAR10_CLASSES,
    set_reproducibility,
)


@hydra.main(config_path="config", config_name="config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    set_reproducibility(cfg)
    data_module = CIFAR10DataModule(cfg)
    data_module.prepare_data()
    model = ResNet18Model(cfg, class_weights=data_module.class_weights)

    if cfg.get("checkpoint_path"):
        model = ResNet18Model.load_from_checkpoint(
            checkpoint_path=cfg.checkpoint_path,
            cfg=cfg,
            class_weights=data_module.class_weights,
        )

    if cfg.compile:
        model = torch.compile(model)
    torch.set_float32_matmul_precision("medium")

    config_dict = OmegaConf.to_container(cfg, resolve=True)

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

    if cfg.fine_tune_on_real_data:
        print(cfg)
        wandb_logger2 = WandbLogger(project=cfg.project, log_model=True)
        wandb.init(
            config=config_dict,
            project=cfg.project,
            name=cfg.name + "_fine_tuned",
            resume="allow",
        )
        # Fine-tune the model on real data
        cfg.downsample_class = None
        cfg.naive_oversample = False
        cfg.naive_undersample = False
        cfg.keep_only_cat = False
        cfg.smote = False
        cfg.adasyn = False
        cfg.label_smoothing = False
        cfg.class_weighting = False
        cfg.epochs = 100
        cfg.add_extra_images = False
        for class_name in CIFAR10_CLASSES:
            cfg.downsample_classes[class_name] = 0.1
            cfg.extra_images_per_class[class_name] = 0
        cfg.dynamic_upsample = False
        cfg.cutmix_or_mixup = False
        cfg.name += "_fine_tuned"
        cfg.naive_undersample = True
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
        feature_map_img, reseized_img = visualize_feature_maps(
            model, data_module, return_image=True
        )
        filter_img = visualize_filters(model, return_image=True)

        if feature_map_img is not None:
            wandb.log(
                {"feature_maps": wandb.Image(feature_map_img, caption="Feature Maps")}
            )
        if reseized_img is not None:
            wandb.log(
                {"resized_img": wandb.Image(reseized_img, caption="Resized Image")}
            )
        if filter_img is not None:
            wandb.log({"conv_filters": wandb.Image(filter_img, caption="Conv Filters")})


if __name__ == "__main__":
    main()
