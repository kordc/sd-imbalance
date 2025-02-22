import hydra
import lightning as L
import torch
import wandb
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf

from data import CIFAR10DataModule
from model import ResNet18Model
from utils import CIFAR10_CLASSES


@hydra.main(config_path="config", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    L.seed_everything(cfg.seed, workers=True)
    data_module = CIFAR10DataModule(cfg)
    data_module.prepare_data()
    model = ResNet18Model(cfg, class_weights=data_module.class_weights)

    if cfg.get("checkpoint_path"):
        print(f"Loading checkpoint from {cfg.checkpoint_path}")
        model = ResNet18Model.load_from_checkpoint(
            checkpoint_path=cfg.checkpoint_path,
            cfg=cfg,
            class_weights=data_module.class_weights
        )
    
    if cfg.compile:
        model = torch.compile(model)
    torch.set_float32_matmul_precision("medium")

    if cfg.get("visualize_feature_maps", False):
        import random
        import matplotlib.pyplot as plt
        import torchvision.transforms.functional as TF
        from PIL import Image

        cat_label = CIFAR10_CLASSES["cat"]
        # Access the underlying dataset (random_split creates a Subset)
        if hasattr(data_module.train_dataset, "dataset"):
            dataset = data_module.train_dataset.dataset
        else:
            dataset = data_module.train_dataset

        # Find indices for samples with the "cat" label.
        cat_indices = [i for i, target in enumerate(dataset.targets) if target == cat_label]
        if not cat_indices:
            print("No cat samples found in the dataset.")
        else:
            chosen_index = random.choice(cat_indices)
            sample, label = dataset[chosen_index]
            
            # Save the original sample image.
            # If sample is a tensor, convert to PIL image.
            if torch.is_tensor(sample):
                sample = sample.cpu()  # Ensure it's on CPU.
                mean = [0.485, 0.456, 0.406]
                std = [0.229, 0.224, 0.225]
                def unnormalize(tensor, mean, std):
                    # Create tensors for mean and std and apply unnormalization.
                    mean_tensor = torch.tensor(mean, device=tensor.device).view(-1, 1, 1)
                    std_tensor = torch.tensor(std, device=tensor.device).view(-1, 1, 1)
                    return tensor * std_tensor + mean_tensor
    
                sample = unnormalize(sample, mean, std)
                sample = sample.clamp(0, 1)
                # Optionally convert to 8-bit values explicitly:
                sample_img = TF.to_pil_image(sample)
                sample_img_resized = sample_img.resize((512, 512), resample=Image.BICUBIC)
            elif isinstance(sample, Image.Image):
                sample_img = sample
            else:
                raise TypeError("Unsupported image type for saving the sample.")
            
            sample_save_path = "sample_image.png"
            sample_img_resized.save(sample_save_path)
            print(f"Sample image saved to {sample_save_path}")
            
            sample = dataset[chosen_index][0].unsqueeze(0).to(model.device)
            # Get feature maps from a designated layer (see method below)
            feature_maps = model.visualize_feature_maps(sample)

            # Plot and save a few feature maps (e.g. first 8 channels)
            num_maps = min(feature_maps.shape[1], 8)
            fig, axs = plt.subplots(1, num_maps, figsize=(num_maps * 2, 2))
            for i in range(num_maps):
                fm = feature_maps[0, i].cpu().numpy()
                axs[i].imshow(fm, cmap="viridis")
                axs[i].axis("off")
            feature_maps_save_path = "feature_maps.png"
            plt.savefig(feature_maps_save_path, bbox_inches="tight")
            plt.close()
            print(f"Feature maps saved to {feature_maps_save_path}")
        return

    # csv_logger = CSVLogger(save_dir="logs/", name="cifar10")
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    wandb_logger = WandbLogger(project="cifar10_project", log_model="True")
    wandb.init(config=config_dict, project="cifar10_project", name=cfg.name)

    trainer = L.Trainer(
        max_epochs=cfg.epochs,
        accelerator="auto",
        devices="auto",
        logger=[wandb_logger],
        log_every_n_steps=1,
    )

    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
