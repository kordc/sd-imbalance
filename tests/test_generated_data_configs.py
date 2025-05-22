import subprocess
import os
import shutil
import pytest
from unittest.mock import patch, MagicMock
import numpy as np
import torch
from PIL import Image

# Get the root directory of the repository (where train.py is located)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


# --- Mock CIFAR10 Dataset Class (slightly larger for better mocking of filters) ---
class MockCIFAR10(torch.utils.data.Dataset):
    """
    A mock CIFAR10 dataset that simulates the behavior of torchvision.datasets.CIFAR10
    for testing purposes. Provides 100 blank images for each of the 10 classes
    to allow for more robust sampling in tests (e.g., similarity filtering).
    """

    def __init__(self, root, train=True, transform=None, download=False):
        self.transform = transform
        self.classes = [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ]
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}

        # Generate mock data: 100 blank 32x32x3 images for each of 10 classes = 1000 total
        samples_per_class = 100
        total_samples = samples_per_class * 10
        self.data = np.zeros((total_samples, 32, 32, 3), dtype=np.uint8)
        self.targets = []
        for i in range(10):  # For each of the 10 classes
            self.targets.extend([i] * samples_per_class)

        # Store original mean/std for DownsampledCIFAR10 initialization
        self.original_mean = np.array([0.5, 0.5, 0.5])
        self.original_std = np.array([0.1, 0.1, 0.1])
        self.data = self.data.astype(np.uint8)
        self.targets = list(self.targets)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_array = self.data[index]
        target = self.targets[index]
        img = Image.fromarray(img_array)
        if self.transform is not None:
            img = self.transform(img)
        return img, target


# --- Pytest Fixture for Cleanup (shared) ---
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
        "feature_maps.png",
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


# --- Fixture for Dummy Generated Data ---
@pytest.fixture
def dummy_generated_data_dir():
    """
    Creates a temporary directory with dummy generated images and cleans it up.
    """
    dummy_dir = os.path.join(REPO_ROOT, "generated_data", "dummy_test_data")
    os.makedirs(dummy_dir, exist_ok=True)

    cifar10_classes = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]
    num_dummy_images_per_class = (
        20  # Provide enough for extra_images_per_class settings
    )

    for class_name in cifar10_classes:
        for i in range(num_dummy_images_per_class):
            # Create a 32x32 RGB image with random noise
            img_array = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img.save(os.path.join(dummy_dir, f"{class_name}_{i}.png"))

    yield dummy_dir

    print(f"\nCleaning up dummy generated data directory: {dummy_dir}")
    shutil.rmtree(dummy_dir, ignore_errors=True)


# --- Helper to run training command with generated data mocking ---
def run_training_command_with_generated_data(
    overrides: list[str], extra_images_dir: str
):
    """
    Helper function to execute the train.py script with specified Hydra overrides,
    a mocked CIFAR10 dataset, and configured generated data directory.
    """
    base_command = [
        "uv",
        "run",
        "train.py",
        "epochs=1",  # Always 1 epoch
        f"extra_images_dir={extra_images_dir}",  # Use the dummy dir
        "project=test_generated_data_project",  # Specific project for these tests
        "visualize_trained_model=False",  # Disable visualizations by default
        "finetune_on_checkpoint=False",
        "fine_tune_on_real_data=False",
    ]
    command = base_command + overrides
    print(f"\nExecuting command: {' '.join(command)}")

    # Patch torchvision.datasets.CIFAR10 and external libs (CLIP, OpenCV)
    # The order of patches matters if one imports another (e.g., data imports torchvision)
    with patch("data.torchvision.datasets.CIFAR10", new=MockCIFAR10) as _:
        with patch("data.clip", new=MagicMock()) as mock_clip:  # Mock clip
            with patch("data.cv2", new=MagicMock()) as mock_cv2:  # Mock cv2
                # Configure mocks for specific functions that might be called
                # Mock clip.load to return a dummy model and preprocess function
                mock_clip.load.return_value = (
                    MagicMock(encode_image=MagicMock(return_value=torch.randn(1, 512))),
                    MagicMock(),
                )
                # Mock cv2 functions that normalize_synthetic="clahe" might call
                mock_cv2.createCLAHE.return_value.apply.return_value = np.zeros(
                    (32, 32), dtype=np.uint8
                )  # Dummy CLAHE output
                mock_cv2.cvtColor.return_value = np.zeros(
                    (32, 32, 3), dtype=np.uint8
                )  # Dummy color conversion
                mock_cv2.split.return_value = (
                    np.zeros((32, 32)),
                    np.zeros((32, 32)),
                    np.zeros((32, 32)),
                )
                mock_cv2.merge.return_value = np.zeros((32, 32, 3))

                try:
                    _ = subprocess.run(
                        command,
                        cwd=REPO_ROOT,
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    print("Command executed successfully.")
                    # print(f"Stdout:\n{result.stdout}") # Uncomment for debugging
                    # if result.stderr: print(f"Stderr:\n{result.stderr}") # Uncomment for debugging
                except subprocess.CalledProcessError as e:
                    print(f"Command failed with exit code {e.returncode}")
                    print(f"Stdout:\n{e.stdout}")
                    print(f"Stderr:\n{e.stderr}")
                    raise  # Re-raise the exception to fail the test


# --- Test Cases for Generated Data ---


def test_add_extra_images_enabled(dummy_generated_data_dir):
    """
    Tests enabling adding extra images from the dummy directory.
    """
    run_training_command_with_generated_data(
        [
            "add_extra_images=True",
            "name=gen_test_add_extra_images",
            # Ensure extra_images_per_class is set to add some images.
            # If not specified, it adds all found images. Let's add all.
            "extra_images_per_class.airplane=20",  # Add all dummy images for airplane
            "extra_images_per_class.cat=20",
        ],
        dummy_generated_data_dir,
    )


def test_normalize_synthetic_mean_std(dummy_generated_data_dir):
    """
    Tests enabling mean_std normalization for synthetic images.
    """
    run_training_command_with_generated_data(
        [
            "add_extra_images=True",
            "normalize_synthetic=mean_std",
            "extra_images_per_class.airplane=5",  # Need to add images for normalization to apply
            "name=gen_test_normalize_mean_std",
        ],
        dummy_generated_data_dir,
    )


def test_normalize_synthetic_clahe(dummy_generated_data_dir):
    """
    Tests enabling CLAHE normalization for synthetic images.
    Requires mocking cv2.
    """
    run_training_command_with_generated_data(
        [
            "add_extra_images=True",
            "normalize_synthetic=clahe",
            "extra_images_per_class.airplane=5",  # Need to add images
            "name=gen_test_normalize_clahe",
        ],
        dummy_generated_data_dir,
    )


def test_similarity_filter_original(dummy_generated_data_dir):
    """
    Tests similarity filtering against original dataset samples.
    Requires mocking clip.
    """
    run_training_command_with_generated_data(
        [
            "add_extra_images=True",
            "similarity_filter=original",
            "similarity_threshold=0.5",
            "reference_sample_size=10",  # Must be <= samples per class in MockCIFAR10
            "extra_images_per_class.cat=5",  # Add a few images to be filtered
            "name=gen_test_similarity_filter_original",
        ],
        dummy_generated_data_dir,
    )


def test_similarity_filter_synthetic(dummy_generated_data_dir):
    """
    Tests similarity filtering against other synthetic samples.
    Requires mocking clip.
    """
    run_training_command_with_generated_data(
        [
            "add_extra_images=True",
            "similarity_filter=synthetic",
            "similarity_threshold=0.5",
            "reference_sample_size=5",
            "extra_images_per_class.cat=5",  # Add a few images to be filtered
            "name=gen_test_similarity_filter_synthetic",
        ],
        dummy_generated_data_dir,
    )


def test_dynamic_upsample_with_generated_data(dummy_generated_data_dir):
    """
    Tests enabling dynamic upsampling with the dummy generated data directory
    as the candidate pool.
    """
    run_training_command_with_generated_data(
        [
            "dynamic_upsample=True",
            "dynamic_upsample_target_class=cat",  # Required for dynamic upsampling
            "num_dynamic_upsample=5",  # Attempt to add 5 images
            "name=gen_test_dynamic_upsample",
        ],
        dummy_generated_data_dir,
    )
