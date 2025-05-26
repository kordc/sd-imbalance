# train.py

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
    # data_module.prepare_data() will be called:
    # - explicitly before trainer.test() in test_only mode
    # - implicitly by trainer.fit() or trainer.test() if not called before

    config_dict: Dict[str, Any] = OmegaConf.to_container(cfg, resolve=True)  # For WandB

    # --- Test-Only Mode ---
    if cfg.get("test_only", False):
        print("INFO: Running in test-only mode.")
        data_module.prepare_data()  # Crucial to prepare the correct test_dataset

        if not cfg.get("checkpoint_path"):
            print(
                "ERROR: 'checkpoint_path' must be provided in config for test-only mode."
            )
            return

        model = ResNet18Model.load_from_checkpoint(
            checkpoint_path=cfg.checkpoint_path,
            cfg=cfg,
            class_weights=data_module.class_weights,
        )
        print(f"INFO: Loaded model from checkpoint: {cfg.checkpoint_path}")

        if cfg.compile:
            print("INFO: Compiling model for test-only mode...")
            model = torch.compile(model)

        torch.set_float32_matmul_precision("medium")

        run_name_suffix = "_test_only"
        if cfg.get("test_on_real", False):
            run_name_suffix += "_real"

        wandb_logger_test = WandbLogger(
            project=cfg.project, name=cfg.name + run_name_suffix, log_model=False
        )
        wandb.init(
            config=config_dict,
            project=cfg.project,
            name=cfg.name + run_name_suffix,
            resume="allow",
        )

        print("INFO: Initializing Trainer for testing...")
        tester = L.Trainer(
            accelerator="auto",
            devices="auto",
            logger=[wandb_logger_test],
        )

        print("INFO: Starting testing...")
        tester.test(model, datamodule=data_module)
        print("INFO: Testing finished.")

        if cfg.get("visualize_trained_model", False):
            print("INFO: Visualizing model post-testing...")
            # Ensure wandb run is active for logging visualizations
            if wandb.run is None:
                wandb.init(
                    config=config_dict,
                    project=cfg.project,
                    name=cfg.name + run_name_suffix + "_viz",
                    resume="allow",
                )

            feature_map_img, resized_img = visualize_feature_maps(
                model, data_module, return_image=True
            )
            filter_img = visualize_filters(model, return_image=True)
            if feature_map_img is not None:
                wandb.log(
                    {
                        "feature_maps_test_only": wandb.Image(
                            feature_map_img, caption="Feature Maps (Test Only)"
                        )
                    }
                )
            if resized_img is not None:
                wandb.log(
                    {
                        "resized_img_test_only": wandb.Image(
                            resized_img, caption="Resized Image (Test Only)"
                        )
                    }
                )
            if filter_img is not None:
                wandb.log(
                    {
                        "conv_filters_test_only": wandb.Image(
                            filter_img, caption="Conv Filters (Test Only)"
                        )
                    }
                )
            print("INFO: Visualization finished.")

        if wandb.run:
            wandb.finish()
        return  # Exit after test-only mode

    # --- Full Training/Fine-tuning Mode (Original Logic) ---
    print("INFO: Running in standard training/fine-tuning mode.")
    data_module.prepare_data()  # Prepare data for training, val, and test

    model: L.LightningModule = ResNet18Model(
        cfg, class_weights=data_module.class_weights
    )

    if cfg.get("checkpoint_path") and not (
        cfg.finetune_on_checkpoint or cfg.fine_tune_on_real_data
    ):  # Only load if not starting fine-tuning from scratch
        print(
            f"INFO: Loading model from checkpoint for continued training or evaluation: {cfg.checkpoint_path}"
        )
        # Note: If finetune_on_checkpoint is true, loading happens inside that block.
        # This handles cases where checkpoint_path is for resuming normal training or just initial eval.
        model = ResNet18Model.load_from_checkpoint(
            checkpoint_path=cfg.checkpoint_path,
            cfg=cfg,
            class_weights=data_module.class_weights,
        )

    if cfg.compile:
        print("INFO: Compiling model...")
        model = torch.compile(model)

    torch.set_float32_matmul_precision("medium")

    # --- Initial phase of training ---
    if (
        not cfg.finetune_on_checkpoint
    ):  # This condition means we are doing initial training
        wandb_logger_train = WandbLogger(
            project=cfg.project, name=cfg.name, log_model=True
        )  # Log model checkpoints
        if (
            wandb.run
        ):  # If a run (e.g. from test_only) was active and not finished, finish it.
            print(
                "Warning: Previous wandb run was active. Finishing it before starting a new one for training."
            )
            wandb.finish()
        wandb.init(
            config=config_dict, project=cfg.project, name=cfg.name
        )  # Fresh run for training

        trainer = L.Trainer(
            max_epochs=cfg.epochs,
            accelerator="auto",
            devices="auto",
            logger=[wandb_logger_train],
            log_every_n_steps=1,
            check_val_every_n_epoch=1,
        )
        if cfg.epochs > 0:
            print("INFO: Starting initial training phase...")
            trainer.fit(model, datamodule=data_module)
            print("INFO: Initial training finished.")
        else:
            print("INFO: cfg.epochs is 0, skipping initial trainer.fit().")

        print("INFO: Starting testing after initial training phase (or if epochs=0)...")
        trainer.test(
            model, datamodule=data_module
        )  # Test after fit, or if only evaluation is intended with epochs=0
        print("INFO: Testing finished.")
        # wandb.finish() # Let WandB finish at the very end or when a new phase starts

    # --- Fine-tuning on real data ---
    if (
        cfg.fine_tune_on_real_data
    ):  # This block is independent of finetune_on_checkpoint for its execution
        print("INFO: Preparing for fine-tuning phase...")

        # Potentially re-init wandb for fine-tuning to have a distinct run or segment
        ft_run_name = cfg.name + "_fine_tuned"
        if wandb.run and wandb.run.name != ft_run_name:  # If a different run is active
            wandb.finish()
        if wandb.run is None or wandb.run.name != ft_run_name:
            wandb.init(
                config=config_dict,
                project=cfg.project,
                name=ft_run_name,
                resume="allow",
            )

        wandb_logger_ft = WandbLogger(
            project=cfg.project, name=ft_run_name, log_model=True
        )

        # Original logic for fine-tuning config and data module re-init
        if cfg.finetune_on_checkpoint and not cfg.checkpoint_path:
            print(
                "\n\nERROR: 'fine_tune_on_real_data' and 'finetune_on_checkpoint' are true, but 'checkpoint_path' is not provided!\n\n"
            )
            if wandb.run:
                wandb.finish()
            return

        prepare_fine_tune(cfg)  # Modifies cfg in-place for fine-tuning data settings

        # Re-initialize data module with fine-tuning settings
        # This call to prepare_data() is critical for fine-tuning specific data setup
        fine_tune_data_module = CIFAR10DataModule(cfg)
        fine_tune_data_module.prepare_data()

        fine_tune_trainer = L.Trainer(
            max_epochs=cfg.epochs,  # Uses the same epochs, or you might add cfg.fine_tune_epochs
            accelerator="auto",
            devices="auto",
            logger=[wandb_logger_ft],
            log_every_n_steps=1,
            check_val_every_n_epoch=1,
        )

        if cfg.finetune_on_checkpoint:
            print(
                f"INFO: Loading model from checkpoint for fine-tuning: {cfg.checkpoint_path}"
            )
            model_ft = ResNet18Model.load_from_checkpoint(  # Use a new variable or overwrite model
                checkpoint_path=cfg.checkpoint_path,
                cfg=cfg,  # cfg might have been modified by prepare_fine_tune
                class_weights=fine_tune_data_module.class_weights,
            )
        else:  # Fine-tune the model trained in the initial phase, or a fresh one if initial phase was skipped
            print(
                "INFO: Initializing/using existing model for fine-tuning (not from a specific fine-tune checkpoint)."
            )
            # If initial training happened, 'model' is already trained.
            # If initial training was skipped (epochs=0) and no checkpoint_path, 'model' is fresh.
            # We need to ensure 'model' is correctly set up with new class_weights if data changed.
            model_ft = ResNet18Model(
                cfg, class_weights=fine_tune_data_module.class_weights
            )
            # If 'model' was already loaded or trained, and we want to continue with it:
            # model_ft = model # but update its config and weights if necessary
            # For simplicity, let's assume if not finetune_on_checkpoint, we use the current 'model' state
            # or re-initialize if that's cleaner. Here, re-initializing with new data params.
            # If we want to continue from 'model' trained in the first phase:
            # model.cfg = cfg # update config
            # model.class_weights = fine_tune_data_module.class_weights # update weights
            # model_ft = model
            # The provided code initializes a new model or loads one.
            # If initial training produced 'model', and we want to fine-tune THAT 'model'
            # without loading another checkpoint, we should use that.
            # Current logic: if finetune_on_checkpoint=False, it creates a NEW model.
            # Let's adjust to use the existing 'model' if it was trained/loaded before this block.
            if (
                not cfg.get("checkpoint_path") and not cfg.finetune_on_checkpoint
            ):  # if model is from initial training
                model_ft = model  # Use the model from the previous stage
                model_ft.cfg = cfg  # Update its config
                model_ft.class_weights = (
                    fine_tune_data_module.class_weights
                )  # Update class weights for new data
            else:  # if checkpoint_path was for initial load, or no initial training, make a fresh model
                model_ft = ResNet18Model(
                    cfg, class_weights=fine_tune_data_module.class_weights
                )

        if cfg.freeze_backbone:
            print("INFO: Freezing model backbone for fine-tuning...")
            model_ft.freeze_backbone()

        if cfg.compile:
            print("INFO: Compiling model for fine-tuning...")
            model_ft = torch.compile(model_ft)

        if cfg.epochs > 0:
            print("INFO: Starting fine-tuning training phase...")
            fine_tune_trainer.fit(model_ft, datamodule=fine_tune_data_module)
            print("INFO: Fine-tuning training finished.")
        else:
            print("INFO: cfg.epochs is 0, skipping fine_tune_trainer.fit().")

        print("INFO: Starting testing after fine-tuning phase...")
        fine_tune_trainer.test(model_ft, datamodule=fine_tune_data_module)
        print("INFO: Testing after fine-tuning finished.")
        model = model_ft

    if cfg.get("visualize_trained_model", False) and not cfg.get("test_only", False):
        print("INFO: Visualizing trained model...")
        viz_run_name = cfg.name
        if cfg.fine_tune_on_real_data:
            viz_run_name += "_fine_tuned"
        viz_run_name += "_viz"

        if wandb.run is None or wandb.run.name != viz_run_name.replace("_viz", ""):
            if wandb.run:
                wandb.finish()
            wandb.init(
                config=config_dict,
                project=cfg.project,
                name=viz_run_name,
                resume="allow",
            )

        vis_data_module = data_module
        if cfg.fine_tune_on_real_data:
            vis_data_module = fine_tune_data_module

        feature_map_img, resized_img = visualize_feature_maps(
            model, vis_data_module, return_image=True
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
        print("INFO: Visualization finished.")

    if wandb.run:
        wandb.finish()
    print("INFO: Main script execution completed.")


if __name__ == "__main__":
    main()
