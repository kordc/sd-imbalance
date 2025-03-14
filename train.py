import hydra
import lightning as L
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf

import wandb
from data import CIFAR10DataModule
from model import ResNet18Model
from utils import visualize_feature_maps, visualize_filters, apply_gradcam


@hydra.main(config_path="config", config_name="config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    L.seed_everything(cfg.seed, workers=True)
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

    if cfg.get("visualize_feature_maps", False):
        visualize_feature_maps(model, data_module)
        return

    config_dict = OmegaConf.to_container(cfg, resolve=True)
    wandb_logger = WandbLogger(project="cifar10_project", log_model=True)
    wandb.init(config=config_dict, project="cifar10_project", name=cfg.name)

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

    if cfg.get("visualize_trained_model", False):
        # Generate visualizations
        feature_map_img, reseized_img = visualize_feature_maps(model, data_module, return_image=True)
        filter_img = visualize_filters(model, return_image=True)
        # gradcam_img = apply_gradcam(model, data_module, return_image=True)
        
        # Log images to wandb
        if feature_map_img is not None:
            wandb.log({"feature_maps": wandb.Image(feature_map_img, caption="Feature Maps")})
        if reseized_img is not None:
            wandb.log({"resized_img": wandb.Image(reseized_img, caption="Resized Image")})
        if filter_img is not None:
            wandb.log({"conv_filters": wandb.Image(filter_img, caption="Conv Filters")})
        # if gradcam_img:
        #     wandb.log({"gradcam": wandb.Image(gradcam_img, caption="GradCAM Results")})


if __name__ == "__main__":
    main()
