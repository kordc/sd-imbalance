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
    set_reproducibility,
    prepare_fine_tune,
)


@hydra.main(config_path="config", config_name="config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    """
    Orchestrates the entire machine learning pipeline for training and evaluating a
    ResNet18 model on the CIFAR-10 dataset.

    This function performs the following steps:

    1.  **Reproducibility**: Sets random seeds for PyTorch, NumPy, and Python for
        consistent experiment results based on the provided configuration.
    2.  **Data Preparation**: Initializes `CIFAR10DataModule` to handle data loading,
        preprocessing, augmentation, and potentially class imbalance techniques
        (e.g., oversampling, undersampling, synthetic data integration) as defined
        in the configuration. It also computes class weights if enabled.
    3.  **Model Initialization**: Instantiates the `ResNet18Model`. It can
        optionally load pre-trained weights or resume training from a specified
        checkpoint path.
    4.  **Model Compilation (PyTorch 2.0)**: If `cfg.compile` is True, the model
        is compiled using `torch.compile` for potential performance improvements.
    5.  **Precision Setting**: Sets the PyTorch matrix multiplication precision
        to "medium" for balanced performance and numerical stability.
    6.  **WandB Logging**: Initializes Weights & Biases (WandB) for comprehensive
        experiment tracking, including configuration logging, metrics, and optionally
        model artifacts.
    7.  **Initial Training and Testing Phase**:
        *   If `cfg.finetune_on_checkpoint` is False, a `L.Trainer` is initialized
            and the model is trained on the prepared dataset for `cfg.epochs`.
        *   After training, the model is evaluated on the test set.
    8.  **Fine-tuning Phase (Optional)**:
        *   If `cfg.fine_tune_on_real_data` is True, a second training phase is
            initiated. This phase typically re-initializes the data module
            (potentially with different imbalance settings focused on real data)
            and fine-tunes the model.
        *   It can optionally freeze the backbone of the ResNet for transfer
            learning or continue from the previously loaded checkpoint.
        *   A new WandB run is initialized, often with a `_fine_tuned` suffix
            to distinguish it.
    9.  **Visualization (Optional)**: If `cfg.visualize_trained_model` is True,
        feature maps from an intermediate layer and the convolutional filters
        of the trained model are generated and logged to WandB, providing insights
        into the model's learned representations.

    Args:
        cfg (DictConfig): A Hydra configuration object parsed from `config/config.yaml`.
                          It contains all parameters for data loading, augmentations,
                          model architecture, training hyperparameters, imbalance
                          handling strategies, synthetic data integration, logging
                          settings, and visualization options.

                          Key configuration parameters include:
                          - `seed` (int): Random seed for reproducibility.
                          - `project` (str): WandB project name.
                          - `name` (str): WandB run name.
                          - `epochs` (int): Number of training epochs.
                          - `compile` (bool): Whether to compile the model with `torch.compile`.
                          - `checkpoint_path` (str, optional): Path to a model checkpoint to load.
                          - `finetune_on_checkpoint` (bool): If true, assumes fine-tuning is desired
                                                             from a checkpoint.
                          - `fine_tune_on_real_data` (bool): If true, enables a second fine-tuning
                                                            phase on real data.
                          - `freeze_backbone` (bool): If true during fine-tuning, freezes ResNet backbone.
                          - `visualize_trained_model` (bool): If true, generates and logs visualizations.

    Returns:
        None: The function does not return any value. Results and metrics are
              logged to Weights & Biases and potentially saved as checkpoints.
    """
    set_reproducibility(cfg)

    # Initialize data module
    data_module = CIFAR10DataModule(cfg)
    data_module.prepare_data()  # This call prepares datasets, computes class weights, etc.

    # Initialize model
    model: L.LightningModule = ResNet18Model(
        cfg, class_weights=data_module.class_weights
    )

    # Load model from checkpoint if specified
    if cfg.get("checkpoint_path"):
        model = ResNet18Model.load_from_checkpoint(
            checkpoint_path=cfg.checkpoint_path,
            cfg=cfg,
            class_weights=data_module.class_weights,
        )

    # Compile model for performance if enabled
    if cfg.compile:
        model = torch.compile(model)

    # Set PyTorch matrix multiplication precision
    torch.set_float32_matmul_precision("medium")

    # Convert Hydra config to a standard dictionary for WandB logging
    config_dict: Dict[str, Any] = OmegaConf.to_container(cfg, resolve=True)

    # --- Initial phase of training ---
    # This block executes if we are not specifically resuming a fine-tuning phase
    if not cfg.finetune_on_checkpoint:
        wandb_logger = WandbLogger(project=cfg.project, log_model=True)
        # Initialize WandB run
        wandb.init(config=config_dict, project=cfg.project, name=cfg.name)

        # Initialize Lightning Trainer for the first training phase
        trainer = L.Trainer(
            max_epochs=cfg.epochs,
            accelerator="auto",  # Automatically selects available accelerator (GPU/CPU)
            devices="auto",  # Uses all available devices
            logger=[wandb_logger],  # Logs metrics and model to WandB
            log_every_n_steps=1,  # Logs every step
            check_val_every_n_epoch=1,  # Runs validation every epoch
        )

        # Train and test the model
        trainer.fit(model, datamodule=data_module)
        trainer.test(model, datamodule=data_module)

    # --- Fine-tuning on real data ---
    # This block handles the specific case of fine-tuning, potentially after initial training
    # or resuming from a checkpoint for fine-tuning.
    if cfg.finetune_on_checkpoint and not cfg.fine_tune_on_real_data:
        print(
            "To enable finetune_on_checkpoint, you must also enable fine_tune_on_real_data"
        )

    if cfg.fine_tune_on_real_data:
        print(cfg)  # Print the current configuration for review

        # Initialize a new WandB logger for the fine-tuning phase
        wandb_logger2 = WandbLogger(project=cfg.project, log_model=True)
        # Initialize WandB run for fine-tuning, potentially resuming the previous run
        wandb.init(
            config=config_dict,
            project=cfg.project,
            name=cfg.name + "_fine_tuned",  # Appends "_fine_tuned" to the run name
            resume="allow",  # Allows resuming if the run already exists
        )

        # Prepare configuration for fine-tuning (e.g., reset data settings)
        prepare_fine_tune(cfg)

        # Re-initialize data module with potentially modified settings for fine-tuning
        data_module = CIFAR10DataModule(cfg)
        data_module.prepare_data()

        # Initialize Lightning Trainer for the fine-tuning phase
        fine_tune_trainer = L.Trainer(
            max_epochs=cfg.epochs,
            accelerator="auto",
            devices="auto",
            logger=[wandb_logger2],
            log_every_n_steps=1,
            check_val_every_n_epoch=1,
        )

        # Determine how to initialize the model for fine-tuning
        if not cfg.finetune_on_checkpoint:
            # If not loading from checkpoint, initialize a fresh model
            model = ResNet18Model(cfg, class_weights=data_module.class_weights)
        else:
            # If fine-tuning from a checkpoint, ensure a checkpoint path is provided
            if not cfg.checkpoint_path:
                print("\n\nYou must provide a checkpoint path!\n\n")
                return
            # Load the model from the specified checkpoint
            model = ResNet18Model.load_from_checkpoint(
                checkpoint_path=cfg.checkpoint_path,
                cfg=cfg,
                class_weights=data_module.class_weights,
            )

        # Freeze backbone if specified for transfer learning
        if cfg.freeze_backbone:
            model.freeze_backbone()

        # Compile model for fine-tuning if enabled
        if cfg.compile:
            model = torch.compile(model)

        # Train and test the model during the fine-tuning phase
        fine_tune_trainer.fit(model, datamodule=data_module)
        fine_tune_trainer.test(model, datamodule=data_module)

    # --- Visualization of trained model (optional) ---
    # Generates and logs feature maps and convolutional filters to WandB.
    if cfg.get("visualize_trained_model", False):
        # Generate feature map and resized input image
        feature_map_img, resized_img = visualize_feature_maps(
            model, data_module, return_image=True
        )
        # Generate convolutional filter image
        filter_img = visualize_filters(model, return_image=True)

        # If fine-tuning was involved, re-initialize WandB to ensure visualizations
        # are logged to the correct run or a new dedicated visualization run.
        if cfg.get("finetune_on_checkpoint", False):
            wandb.init(
                config=config_dict,
                project=cfg.project,
                name=cfg.name
                + "viz",  # Appends "viz" to the run name for visualizations
                resume="allow",
            )

        # Log generated images to WandB
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
