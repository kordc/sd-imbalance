# tests/test_training_configs.py
import subprocess
import os
import shutil
import pytest

# Get the root directory of the repository (where train.py is located)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


@pytest.fixture(autouse=True)
def cleanup_artifacts():
    """
    Fixture to clean up generated artifacts (e.g., WandB logs, image files,
    Hydra outputs) after each test run.
    """
    yield
    print("\nCleaning up artifacts...")
    artifacts_to_remove = [
        "wandb",
        "train.log",
        "sample_image.png",
        "conv_filters.png",
        ".hydra",
    ]
    for item in artifacts_to_remove:
        path_to_remove = os.path.join(REPO_ROOT, item)
        if os.path.isdir(path_to_remove):
            print(f"Removing directory: {path_to_remove}")
            shutil.rmtree(path_to_remove, ignore_errors=True)
        elif os.path.isfile(path_to_remove):
            print(f"Removing file: {path_to_remove}")
            os.remove(path_to_remove)
    print("Cleanup complete.")


def run_training_command(overrides: list[str]):
    """
    Helper function to execute the train.py script with specified Hydra overrides.
    It enforces 1 epoch and disables extra data addition for all tests.
    """
    base_command = [
        "uv",
        "run",
        "train.py",
        "epochs=1",  # Always 1 epoch as requested
        "add_extra_images=False",  # Always False for extra data as requested
        "name=test_run",  # Generic WandB run name for testing (will be overridden)
        "project=test_project",  # Generic WandB project name for testing
        "visualize_trained_model=False",  # Disable visualizations by default for faster tests
        "finetune_on_checkpoint=False",  # Ensure this is off for simple tests
        "fine_tune_on_real_data=False",  # Ensure this is off for simple tests unless explicitly tested
    ]
    command = base_command + overrides
    print(f"\nExecuting command: {' '.join(command)}")
    try:
        # Use capture_output=True to suppress stdout/stderr unless an error occurs
        # or if you want to inspect output for debugging.
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            check=True,  # Raises CalledProcessError if the command returns a non-zero exit code
            capture_output=True,
            text=True,
        )
        print("Command executed successfully.")
        # Optionally print stdout/stderr for successful runs if needed for debugging
        # print(f"Stdout:\n{result.stdout}")
        # if result.stderr:
        #     print(f"Stderr:\n{result.stderr}")
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}")
        print(f"Stdout:\n{e.stdout}")
        print(f"Stderr:\n{e.stderr}")
        raise  # Re-raise the exception to fail the test


def test_default_run_no_extra_data():
    """
    Tests a basic training run with default config, but explicitly
    setting epochs to 1 and ensuring no extra images are added.
    """
    run_training_command(["name=test_default_run_no_extra_data"])


def test_downsample_specific_classes():
    """
    Tests downsampling 'cat' and 'airplane' classes to different ratios.
    """
    run_training_command(
        [
            "downsample_classes.cat=0.1",
            "downsample_classes.airplane=0.5",
            "name=test_downsample_specific_classes",
        ]
    )


def test_smote_oversampling_enabled():
    """
    Tests enabling SMOTE for oversampling.
    Ensures other resampling methods are explicitly disabled to avoid conflicts.
    """
    run_training_command(
        [
            "smote=True",
            "naive_oversample=False",
            "naive_undersample=False",
            "adasyn=False",
            "name=test_smote_oversampling_enabled",
        ]
    )


def test_adasyn_oversampling_enabled():
    """
    Tests enabling ADASYN for oversampling.
    Ensures other resampling methods are explicitly disabled to avoid conflicts.
    """
    run_training_command(
        [
            "adasyn=True",
            "naive_oversample=False",
            "naive_undersample=False",
            "smote=False",
            "name=test_adasyn_oversampling_enabled",
        ]
    )


def test_naive_oversampling_enabled():
    """
    Tests enabling naive random oversampling.
    Ensures other resampling methods are explicitly disabled to avoid conflicts.
    """
    run_training_command(
        [
            "naive_oversample=True",
            "smote=False",
            "naive_undersample=False",
            "adasyn=False",
            "name=test_naive_oversampling_enabled",
        ]
    )


def test_naive_undersampling_enabled():
    """
    Tests enabling naive random undersampling.
    Ensures other resampling methods are explicitly disabled to avoid conflicts.
    """
    run_training_command(
        [
            "naive_undersample=True",
            "naive_oversample=False",
            "smote=False",
            "adasyn=False",
            "name=test_naive_undersampling_enabled",
        ]
    )


def test_label_smoothing_enabled():
    """
    Tests enabling label smoothing.
    """
    run_training_command(["label_smoothing=True", "name=test_label_smoothing_enabled"])


def test_class_weighting_enabled():
    """
    Tests enabling class weighting.
    """
    run_training_command(["class_weighting=True", "name=test_class_weighting_enabled"])


def test_different_model_settings():
    """
    Tests running with a modified batch size and a custom WandB run name.
    """
    run_training_command(
        [
            "batch_size=64",
            "compile=False",
            "learning_rate=0.001",
            "val_size=0.3",
            "seed=51",
            "name=test_different_model_settings",
        ]
    )


def test_dynamic_upsample_cat_class():
    """
    Tests enabling dynamic upsampling for the 'cat' class.
    Note: This will attempt to read from `extra_images_dir` defined in config.yaml.
    If that directory is empty or doesn't exist, the `dynamic_upsample` function
    is designed to handle it gracefully (it will print a warning and return).
    """
    run_training_command(
        [
            "dynamic_upsample=True",
            "dynamic_upsample_target_class=cat",  # Required for dynamic upsampling
            "name=test_dynamic_upsample_cat_class",
        ]
    )


def test_pretrained_model_enabled():
    """
    Tests running the model initialized with pretrained weights.
    """
    run_training_command(["pretrained=True", "name=test_pretrained_model_enabled"])


def test_visualize_trained_model_and_check_files():
    """
    Tests enabling model visualization, and asserts that the output image files are created.
    """
    run_training_command(
        [
            "visualize_trained_model=True",
            "name=test_visualize_trained_model_and_check_files",
        ]
    )
    # After successful command execution, verify the files exist
    assert os.path.exists(os.path.join(REPO_ROOT, "sample_image.png"))
    assert os.path.exists(os.path.join(REPO_ROOT, "conv_filters.png"))


def test_fine_tune_on_real_data_enabled():
    """
    Tests enabling the fine-tuning phase on real data.
    """
    run_training_command(
        ["fine_tune_on_real_data=True", "name=test_fine_tune_on_real_data_enabled"]
    )


def test_freeze_backbone_during_finetuning():
    """
    Tests freezing the backbone during the fine-tuning phase.
    """
    run_training_command(
        [
            "fine_tune_on_real_data=True",
            "freeze_backbone=True",
            "name=test_freeze_backbone_during_finetuning",
        ]
    )


# todo: checkpoint_path, finetune_on_checkpoint, augmentations, test_augmentations, add_extra_images, normalize_synthetic, similarity_filter, dynamic_upsample
