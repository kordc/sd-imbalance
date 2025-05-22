# ImbalanceSD: Your Toolkit for Training & Generating Data to Combat Imbalance 📊✨

Imbalanced datasets are a common headache in machine learning, often leading to models that perform poorly on minority classes. This repository offers a powerful experimental framework to tackle this challenge head-on!

## 🚀 Quick Start

Want to dive straight in? Here's how to get your environment ready and run a default training:

1.  **Install `uv`** (if you don't have it already):
    *   Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`
    *   Windows: `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/0.5.26/install.ps1 | iex"`
2.  **Sync Dependencies:**
    ```sh
    uv sync
    ```
3.  **Run a Training Experiment:**
    ```sh
    uv run train.py
    ```
    (Don't forget to configure your WandB account for experiment tracking! See [WandB Setup](#wandb-account-configuration) below).

## 🛠️ Setup: Get Your Environment Ready!

To get started with `ImbalanceSD`, follow these simple steps:

### ⚡ Install `uv` & Dependencies

We use `uv` for lightning-fast package management and virtual environment creation. It's awesome for reproducibility!

*   **Linux:**
    ```sh
    curl -LsSf https://astral.sh/uv/install.sh | sh
    uv sync
    ```
*   **Windows:**
    ```sh
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/0.5.26/install.ps1 | iex"
    uv sync
    ```
    The `uv sync` command will create a virtual environment (`.venv`) and install all necessary packages.

### 🔑 WandB Account Configuration

We highly recommend configuring your Weights & Biases (WandB) account to track, visualize, and compare your experiments with ease!

1.  Go to [https://wandb.ai/authorize](https://wandb.ai/authorize) to get your API key.
2.  Set it as a system environment variable:
    ```sh
    WANDB_API_KEY=$YOUR_API_KEY
    ```
    ⚠️ **Heads up!** This step is *obligatory* if you want to see interactive plots and track experiments in the cloud. You'll still get local logs, but the full WandB dashboard experience requires this.

### 🧹 Code Linting & Formatting with Ruff

To ensure consistent and clean code, we use `ruff`. Before pushing changes to the `main` branch, make sure your code adheres to the style guidelines:

```sh
uvx ruff check .
uvx ruff format .
```

## 🚀 Usage: Running Your Experiments
- The project uses [Hydra](https://hydra.cc/) for configuration management, allowing you to easily override parameters from the config/config.yaml file via command-line arguments.
- All scripts are designed to run within the virtual environment created by uv. We primarily use the uv run approach for simplicity.

## 🖼️ Generating Data
This repository includes scripts to generate synthetic data using diffusion models. Currently, our data generation efforts are tailored specifically for CIFAR10.

To generate data, please use the provided scripts. Remember that a folder with generated samples should follow a consistent naming strategy: CLASSNAME_(...).jpg (e.g., airplane_0001.jpg, cat_500.jpg).

For more detailed information on data generation scripts and their usage, please refer to the dedicated [README](scripts/README.md).

## 🏋️‍♀️ Training Your Model
o kick off a training run with the default settings (as defined in [config/config.yaml](config/config.yaml)), simply execute:
```sh
uv run train.py
```

### Overriding Configuration Parameters
One of the coolest features here is how easy it is to customize your experiments. You can modify any parameter from [config/config.yaml](config/config.yaml) by passing `key=value` pairs on the command line. For nested parameters, use dot notation (e.g., `section.param=value`).

**Examples**:
1. **Downsample multiple classes and add synthetic images:**

    This example downsamples 'airplane', 'automobile', and 'cat' classes to 1% of their original size and adds 4950 synthetic images for each of these classes from a specified directory, while also enabling synthetic image normalization and similarity filtering.
    ```py
    uv run train.py \
    downsample_classes.airplane=0.01 \
    downsample_classes.automobile=0.01 \
    downsample_classes.cat=0.01 \
    add_extra_images=True \
    extra_images_dir=/path/to/your/generated_data/ \
    extra_images_per_class.airplane=4950 \
    extra_images_per_class.automobile=4950 \
    extra_images_per_class.cat=4950 \
    normalize_synthetic="mean_std" \
    similarity_filter="original" \
    similarity_threshold=0.8
    ```
2. **Enable SMOTE for oversampling:**
    ```py
    python train.py smote=True
    ```
3. Train for fewer epochs with a larger batch size and a specific WandB run name:
    ```py
    python train.py epochs=10 batch_size=128 name="my_short_run"
    ```

## ⚙️ Configuration Options: Customize Your Experiments!
Here's an overview of key parameters you can configure:

### General Project Settings
*   `epochs`: Number of training epochs.
*   `compile`: Enable PyTorch 2.0 `torch.compile` for model optimization.
*   `batch_size`: Training and validation batch size.
*   `learning_rate`: Initial learning rate for the optimizer.
*   `val_size`: Proportion of the training data to use for validation.
*   `num_workers`: Number of data loading workers.
*   `seed`: Random seed for reproducibility.
*   `checkpoint_path`: Path to a model checkpoint to load.
*   `visualize_trained_model`: Boolean flag to generate and log feature maps and filters at the end of training.
*   `freeze_backbone`: If `True`, freezes the backbone of the ResNet model during fine-tuning (only the classification head is trained).
*   `fine_tune_on_real_data`: If `True`, the model undergoes a second training phase specifically on real (not synthetic) data after the initial training.
*   `pretrained`: If `True`, initializes the ResNet18 model with ImageNet pre-trained weights.
*   `finetune_on_checkpoint`: If `True` and `checkpoint_path` is provided, will load the checkpoint and continue training.

### Imbalance Handling and Regularization
*   `naive_oversample`: If `True`, applies naive random oversampling to balance classes.
*   `smote`: If `True`, applies SMOTE (Synthetic Minority Over-sampling Technique).
*   `adasyn`: If `True`, applies ADASYN (Adaptive Synthetic Sampling) for oversampling.
*   `naive_undersample`: If `True`, applies naive random undersampling to balance classes.
*   `label_smoothing`: If `True`, applies label smoothing to the cross-entropy loss.
*   `class_weighting`: If `True`, applies class weights to the loss function based on class frequencies.

**Note**: Only one of `naive_oversample`, `smote`, `adasyn`, or `naive_undersample` can be active at a time.

### Dataset Settings
*   `augmentations`: List of augmentations to apply during training (e.g., `ToTensor`, `RandomCrop`, `RandomHorizontalFlip`, `Resize`).
*   `test_augmentations`: List of augmentations for testing (e.g., `ToTensor`, `Resize`).

### Synthetic Data Integration
These parameters control the addition and filtering of diffusion-generated images. Note, that currently this repository works only with CIFAR10 data.
*   `add_extra_images`: If `True`, extra images from `extra_images_dir` will be added to the training dataset.
*   `extra_images_dir`: Path to the directory containing synthetic images. Images are expected to be named `CLASSNAME_idx.png` (e.g., `cat_1.png`).
*   `downsample_classes`: A dictionary specifying which classes to downsample and by what ratio (e.g., `downsample_classes.cat=0.01` to keep 1% of cat images).
*   `extra_images_per_class`: A dictionary specifying how many synthetic images to add for each class (e.g., `extra_images_per_class.cat=4950`).
*   `normalize_synthetic`: Method to normalize synthetic images:
    *   `None`: No normalization.
    *   `mean_std`: Normalizes synthetic images to have the same mean and std as the original dataset.
    *   `clahe`: Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) and then adjusts mean/std.
*   `similarity_filter`: Method for filtering synthetic images based on CLIP similarity:
    *   `None`: No similarity filtering.
    *   `original`: Filters synthetic images by their similarity to a small reference set from the original dataset.
    *   `synthetic`: Filters synthetic images by their similarity to a small reference set from other synthetic images.
*   `similarity_threshold`: Threshold for similarity filtering (0.0-1.0). Images with scores below this threshold are discarded.
*   `reference_sample_size`: Number of reference images to use for similarity filtering.

### Dynamic Upsampling (Active Learning)
*   `dynamic_upsample`: If `True`, enables dynamic upsampling based on model uncertainty at the end of each training epoch.
*   `num_dynamic_upsample`: Number of new images to add from the candidate pool during each dynamic upsampling step.
*   `dynamic_upsample_target_class`: Targeted class for dynamic upsampling.

### WandB Integration
*   `name`: The name for the Weights & Biases run.
*   `project`: The Weights & Biases project name.

## 💖 Contributing
We welcome contributions! If you have ideas for improvements, new features, or bug fixes, feel free to open an issue or submit a pull request.