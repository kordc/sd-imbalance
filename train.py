# train.py

import hydra
import lightning as L
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
from typing import Dict, Any

import wandb
from data import CIFAR10DataModule
from model import ResNet18Model, ClipClassifier  # Updated import
from utils import (
    visualize_feature_maps,
    visualize_filters,
    set_reproducibility,
    prepare_fine_tune,
)


@hydra.main(config_path="config", config_name="config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    """
    Orchestrates the entire machine learning pipeline for training and evaluating
    a model (ResNet18 or CLIP-based) on the CIFAR-10 dataset.
    (Docstring largely the same, updated for model flexibility)
    """
    set_reproducibility(cfg)

    # Initialize data module
    data_module = CIFAR10DataModule(cfg)
    # data_module.prepare_data() will be called explicitly or implicitly.

    config_dict: Dict[str, Any] = OmegaConf.to_container(cfg, resolve=True)

    # --- Test-Only Mode ---
    if cfg.get("test_only", False):
        print("INFO: Running in test-only mode.")
        data_module.prepare_data()

        if not cfg.get("checkpoint_path"):
            print(
                "ERROR: 'checkpoint_path' must be provided in config for test-only mode."
            )
            return

        # Model selection for test-only mode
        if cfg.model_type == "resnet18":
            model = ResNet18Model.load_from_checkpoint(
                checkpoint_path=cfg.checkpoint_path,
                cfg=cfg,
                class_weights=data_module.class_weights,  # class_weights might not be needed for test-only loading from checkpoint
            )
        elif cfg.model_type == "clip":
            # ClipClassifier might not need class_weights if loading full state.
            # If it's a generic L.LightningModule.load_from_checkpoint,
            # ensure constructor args match or are handled.
            # For simplicity, pass them; the model's load_from_checkpoint or init should handle it.
            model = ClipClassifier.load_from_checkpoint(
                checkpoint_path=cfg.checkpoint_path,
                cfg=cfg,
                class_weights=data_module.class_weights,
            )
            # For CLIP in test-only, we need to ensure text_features are initialized
            # This typically happens in on_fit_start, which isn't called in test-only from checkpoint.
            # So we call it manually, or ensure it's part of saved state / re-initialized.
            # A common pattern is to call setup hooks if needed:
            model.trainer = L.Trainer()  # Dummy trainer for device context
            model.on_fit_start()  # To compute text_features if not loaded
        else:
            raise ValueError(f"Unsupported model_type: {cfg.model_type}")
        print(
            f"INFO: Loaded model ({cfg.model_type}) from checkpoint: {cfg.checkpoint_path}"
        )

        if (
            cfg.compile and cfg.model_type == "resnet18"
        ):  # CLIP compilation can be tricky, enable selectively
            print(f"INFO: Compiling model ({cfg.model_type}) for test-only mode...")
            try:
                model = torch.compile(model)
            except Exception as e:
                print(
                    f"Warning: Failed to compile model {cfg.model_type}: {e}. Continuing without compilation."
                )

        torch.set_float32_matmul_precision("medium")

        run_name_suffix = f"_{cfg.model_type}_test_only"
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

        if cfg.get("visualize_trained_model", False) and cfg.model_type == "resnet18":
            print("INFO: Visualizing ResNet18 model post-testing...")
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
        elif cfg.get("visualize_trained_model", False) and cfg.model_type == "clip":
            print(
                "INFO: Visualization for CLIP model is not implemented in this script."
            )

        if wandb.run:
            wandb.finish()
        return

    # --- Full Training/Fine-tuning Mode ---
    print(
        f"INFO: Running in standard training/fine-tuning mode for model_type: {cfg.model_type}."
    )
    data_module.prepare_data()

    # Model Initialization
    if cfg.model_type == "resnet18":
        model_class = ResNet18Model
    elif cfg.model_type == "clip":
        model_class = ClipClassifier
    else:
        raise ValueError(f"Unsupported model_type: {cfg.model_type}")

    model: L.LightningModule
    if cfg.get("checkpoint_path") and not (
        cfg.finetune_on_checkpoint or cfg.fine_tune_on_real_data
    ):
        print(
            f"INFO: Loading model ({cfg.model_type}) from checkpoint for continued training or evaluation: {cfg.checkpoint_path}"
        )
        model = model_class.load_from_checkpoint(
            checkpoint_path=cfg.checkpoint_path,
            cfg=cfg,
            class_weights=data_module.class_weights,
        )
        if cfg.model_type == "clip":  # Ensure text_features are ready
            model.trainer = L.Trainer()
            model.on_fit_start()
    else:
        model = model_class(cfg, class_weights=data_module.class_weights)

    if (
        cfg.compile and cfg.model_type == "resnet18"
    ):  # CLIP compilation can be tricky, enable selectively
        print(f"INFO: Compiling model ({cfg.model_type})...")
        try:
            model = torch.compile(model)
        except Exception as e:
            print(
                f"Warning: Failed to compile model {cfg.model_type}: {e}. Continuing without compilation."
            )

    torch.set_float32_matmul_precision("medium")

    base_run_name = f"{cfg.name}_{cfg.model_type}"

    # --- Initial phase of training ---
    if not cfg.finetune_on_checkpoint:
        wandb_logger_train = WandbLogger(
            project=cfg.project, name=base_run_name, log_model=True
        )
        if wandb.run:
            print(
                "Warning: Previous wandb run was active. Finishing it before starting a new one for training."
            )
            wandb.finish()
        wandb.init(config=config_dict, project=cfg.project, name=base_run_name)

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
        trainer.test(model, datamodule=data_module)
        print("INFO: Testing finished.")

    # --- Fine-tuning on real data ---
    if cfg.fine_tune_on_real_data:
        print("INFO: Preparing for fine-tuning phase...")

        ft_run_name = f"{base_run_name}_fine_tuned"
        if wandb.run and wandb.run.name != ft_run_name:
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

        if cfg.finetune_on_checkpoint and not cfg.checkpoint_path:
            print(
                "\n\nERROR: 'fine_tune_on_real_data' and 'finetune_on_checkpoint' are true, but 'checkpoint_path' is not provided!\n\n"
            )
            if wandb.run:
                wandb.finish()
            return

        prepare_fine_tune(cfg)

        fine_tune_data_module = CIFAR10DataModule(
            cfg
        )  # cfg now has fine-tune data settings
        fine_tune_data_module.prepare_data()

        fine_tune_trainer = L.Trainer(
            max_epochs=cfg.epochs,
            accelerator="auto",
            devices="auto",
            logger=[wandb_logger_ft],
            log_every_n_steps=1,
            check_val_every_n_epoch=1,
        )

        model_ft: L.LightningModule
        if cfg.finetune_on_checkpoint:
            print(
                f"INFO: Loading model ({cfg.model_type}) from checkpoint for fine-tuning: {cfg.checkpoint_path}"
            )
            model_ft = model_class.load_from_checkpoint(
                checkpoint_path=cfg.checkpoint_path,
                cfg=cfg,
                class_weights=fine_tune_data_module.class_weights,
            )
            if cfg.model_type == "clip":  # Ensure text_features are ready
                model_ft.trainer = L.Trainer()
                model_ft.on_fit_start()
        else:
            print(
                "INFO: Initializing/using existing model for fine-tuning (not from a specific fine-tune checkpoint)."
            )
            # Use the model trained in the initial phase, or a fresh one if initial phase was skipped.
            # Ensure 'model' is correctly set up with new class_weights if data changed.
            if (
                not cfg.get("checkpoint_path")
                and not cfg.finetune_on_checkpoint
                and cfg.epochs > 0
            ):
                model_ft = model  # Use the model from the previous stage
                model_ft.cfg = cfg  # Update its config
                model_ft.class_weights = (
                    fine_tune_data_module.class_weights
                )  # Update class weights
                # Re-initialize criterion if class_weights changed and it's used.
                if model_ft.class_weights is not None:
                    model_ft.criterion = torch.nn.CrossEntropyLoss(
                        weight=model_ft.class_weights.to(model_ft.device)
                        if hasattr(model_ft, "device")
                        else model_ft.class_weights
                    )
                elif cfg.label_smoothing:
                    model_ft.criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.3)
                else:
                    model_ft.criterion = torch.nn.CrossEntropyLoss()

                if cfg.model_type == "clip":  # Ensure text_features are ready
                    model_ft.trainer = L.Trainer()
                    model_ft.on_fit_start()  # Call if text features might need re-calc with new device or context
            else:  # if checkpoint_path was for initial load, or no initial training, make a fresh model
                model_ft = model_class(
                    cfg, class_weights=fine_tune_data_module.class_weights
                )

        if cfg.freeze_backbone:
            print(
                f"INFO: Freezing model ({cfg.model_type}) backbone for fine-tuning..."
            )
            model_ft.freeze_backbone()

        if (
            cfg.compile and cfg.model_type == "resnet18"
        ):  # CLIP compilation can be tricky
            print(f"INFO: Compiling model ({cfg.model_type}) for fine-tuning...")
            try:
                model_ft = torch.compile(model_ft)
            except Exception as e:
                print(
                    f"Warning: Failed to compile model {cfg.model_type} for fine-tuning: {e}. Continuing without compilation."
                )

        if cfg.epochs > 0:
            print("INFO: Starting fine-tuning training phase...")
            fine_tune_trainer.fit(model_ft, datamodule=fine_tune_data_module)
            print("INFO: Fine-tuning training finished.")
        else:
            print("INFO: cfg.epochs is 0, skipping fine_tune_trainer.fit().")

        print("INFO: Starting testing after fine-tuning phase...")
        fine_tune_trainer.test(model_ft, datamodule=fine_tune_data_module)
        print("INFO: Testing after fine-tuning finished.")
        model = model_ft  # Update the main model variable to the fine-tuned one

    # Visualization after all training/fine-tuning
    if cfg.get("visualize_trained_model", False) and not cfg.get("test_only", False):
        if cfg.model_type == "resnet18":
            print("INFO: Visualizing trained ResNet18 model...")
            viz_run_name = base_run_name
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
                    {
                        "feature_maps": wandb.Image(
                            feature_map_img, caption="Feature Maps"
                        )
                    }
                )
            if resized_img is not None:
                wandb.log(
                    {"resized_img": wandb.Image(resized_img, caption="Resized Image")}
                )
            if filter_img is not None:
                wandb.log(
                    {"conv_filters": wandb.Image(filter_img, caption="Conv Filters")}
                )
            print("INFO: ResNet18 Visualization finished.")
        elif cfg.model_type == "clip":
            print(
                "INFO: Visualization for CLIP model is not implemented in this script."
            )

    if wandb.run:
        wandb.finish()
    print("INFO: Main script execution completed.")


if __name__ == "__main__":
    main()
