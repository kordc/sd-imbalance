# Scripts for Data Management and Image Generation

This directory contains scripts supporting the process of data management and synthetic image generation within the project. Each script has a specific task, ranging from downloading reference data, generating diffusion-based images, to augmenting existing datasets.

All image generation scripts are designed so that their output is compatible with the `_add_extra_images` mechanism in the `data.py` file. This means that filenames include a class name prefix (e.g., `cat_imageid.jpg`), and all files for a given generation batch are located in a single, flat target directory (e.g., `internet_reference/`).

## Table of Contents

1.  [`flux_redux_augment.py`](#1-flux_redux_augmentpy)
2.  [`generate_cats_from_lora.py`](#2-generate_cats_from_lorapy)
3.  [`generate_cats_sdxl_turbo.py`](#3-generate_cats_sdxl_turbopy)
4.  [`generate_cifar.py`](#4-generate_cifarpy)
5.  [`generate_original_reference_from_openimages.py`](#5-generate-original-reference-from-openimagespy)
6.  [`save_cifar_10.py`](#6-save_cifar_10py)

---

### 1. `flux_redux_augment.py`

**Purpose:** This script is used to expand existing datasets (e.g., cat images) by applying specialized augmentation based on Flux Redux diffusion models. It utilizes `FluxPriorReduxPipeline` and `FluxPipeline` to generate synthetic variants of input images.

**Key Features:**
-   **`FluxReduxAugment` Class:** Implements a `torchvision`-like transform that takes a PIL image and returns its augmented version with a specified `probability`.
-   **Generation:** For each input image, it generates multiple augmented copies, varying the `seed` for each, which ensures diversity.
-   **Compatibility with `data.py`:** Output filenames (e.g., `cat_originalfilename_aug_0.png`) include a class prefix and the original filename, allowing `data.py` to parse them correctly.

**Usage:** Running the script initiates the augmentation process for images located in `input_dir` (default: `notebooks/cats`) and saves them to `output_dir` (default: `notebooks/cats_redux`).

---

### 2. `generate_cats_from_lora.py`

**Purpose:** This script generates synthetic cat images using a base Stable Diffusion XL (SDXL) model combined with a custom LoRA (Low-Rank Adaptation) model. The goal is to create highly detailed and specific cat images based on elaborate prompts.

**Key Features:**
-   **Prompt Generation:** Uses predefined lists of cat breeds, prepositions, furniture/environment elements, camera angles, and gaze directions to dynamically create complex and diverse prompts.
-   **LoRA Integration:** Loads a LoRA model from the Hugging Face Hub and applies it to the base `StableDiffusionXLPipeline`, allowing the generation of images with specific aesthetics or characteristics that the base model might lack.
-   **Compatibility with `data.py`:** Images are saved in a flat target directory (default: `blurry_LoRA_full_prompt`) with filenames starting with `cat_` (e.g., `cat_00000_on_sofa.png`), which is consistent with the format expected by `data.py`.

**Usage:** The script requires authentication with the Hugging Face Hub (e.g., via `huggingface-cli login`) and access to the specified LoRA repository. Upon execution, it generates a specified number of images to the defined folder.

---

### 3. `generate_cats_sdxl_turbo.py`

**Purpose:** This script is used to generate synthetic cat images using the Stable Diffusion XL Turbo model. It employs a simplified prompt creation strategy (`modifier + class`), inspired by SYNAuG research, to quickly and efficiently generate a large number of diverse images.

**Key Features:**
-   **Simplified Prompting:** Prompts are constructed in a simple format like `"a photo of {selected_modifier} cat"`, where `{selected_modifier}` is randomly selected from a list of qualitative/descriptive modifiers.
-   **SDXL Turbo:** Leverages SDXL Turbo, a model optimized for fast and single-step inference processes, ideal for batch image generation.
-   **Compatibility with `data.py`:** Images are saved in a flat target directory (default: `sdxl_turbo_synaug_style`) with filenames starting with `cat_` (e.g., `cat_00000_realistic.png`).

**Usage:** Running the script initiates the generation of cat images to the specified folder.

---

### 4. `generate_cifar.py`

**Purpose:** This is the most comprehensive script for synthetic data generation, designed to create images for all 10 CIFAR-10 classes. It utilizes SDXL Turbo and an elaborate, dictionary-based concept structure for each class to generate highly diverse and realistic images.

**Key Features:**
-   **Generation for All Classes:** Iterates through the list of CIFAR-10 classes, creating images for each.
-   **Extensive Prompting:** Prompts are constructed in the format `"{selected_quality} photo of a {selected_descriptor} {class_name} {selected_context}"`, where `descriptor_type`, `context_action`, and `quality_modifiers` are randomly chosen from extensive, predefined, class-specific lists.
-   **Output Structure:** Generated images are saved in subfolders corresponding to classes (e.g., `base_folder/airplane/`, `base_folder/cat/`), with filenames in the format `class_name_index.png` (e.g., `airplane_000000.png`).
-   **Memory Management:** Includes mechanisms for clearing GPU memory (`torch.cuda.empty_cache`, `gc.collect`) to minimize Out-Of-Memory (OOM) issues during long-duration generation runs.

**Usage:** The script requires adequate GPU resources due to the large-scale image generation (default: 100,000 images per class). Running it will start the generation of all synthetic images for CIFAR-10.

---

### 5. `generate_original_reference_from_openimages.py`

**Purpose:** This script is used to download real-world images from the extensive Open Images Dataset V6, which are intended to serve as "references" or additional data in the project. It maps CIFAR-10 class names to their corresponding concepts in Open Images.

**Key Features:**
-   **Download from Open Images:** Utilizes the `openimages-py` library for programmatic image downloading.
-   **Class Mapping:** Contains a `CIFAR10_TO_OID_CONCEPT_MAP` dictionary that translates CIFAR-10 class names (e.g., "automobile") to their corresponding Open Images concept names (e.g., "Car").
-   **Flat Output Structure:** All downloaded images are moved and saved into a single, flat target directory (default: `internet_reference`). Filenames are formatted as `class_name_image_id.jpg` (e.g., `airplane_0a1b2c3d4e5f.jpg`), which is crucial for `data.py` to correctly assign classes.
-   **Temporary File Cleanup:** The script manages temporary directories created by the `openimages-py` library, removing them after the download is complete.

**Usage:** Requires the `openimages-py` library to be installed (`pip install openimages-py`). Upon execution, the script will download a specified number of images for each class to the designated directory.

---

### 6. `save_cifar_10.py`

**Purpose:** This simple script is used to download the original CIFAR-10 training dataset and save each image as a separate PNG file. The images are organized in a directory structure where each subdirectory corresponds to a single CIFAR-10 class.

**Key Features:**
-   **CIFAR-10 Download:** Uses `torchvision.datasets.CIFAR10` for easy dataset download and loading.
-   **Data Organization:** Creates a root directory (default: `cifar10_raw`), and within it, subdirectories for each of the 10 CIFAR-10 classes.
-   **Image Saving:** Each image from the dataset is saved as a PNG file in its respective class folder. Filenames are numbered, e.g., `00000.png`, `00001.png`, etc.

**Important Note:** The file structure created by this script (`cifar10_raw/class_name/image_index.png`) **differs** from the flat structure and naming convention (`class_name_image_id.jpg`) expected by the `_add_extra_images` method in `data.py`. This script is intended for preparing a *raw, original* version of CIFAR-10, not for additional images to be used for data expansion.

**Usage:** Running the script will download (if not already downloaded) and save the CIFAR-10 images to the specified directory.

---