# Results

## Trainings and Evaluations

| Method Name                                              | Test Accuracy | Test Balanced Accuracy | Cat Class Accuracy |
| :------------------------------------------------------- | :------------ | :--------------------- | :----------------- |
| All classes downsampled to 1% + Full Synthetic           | 0.493         | N/A                    | 0.429              |
| Cat downsampled to 1%                                    | 0.764         | N/A                    | 0.036              |
| Cat downsampled to 1% + ADASYN                           | 0.803         | N/A                    | 0.010              |
| Cat downsampled to 1% + FLUX.1-Redux augmentation        | 0.815         | N/A                    | 0.073              |
| Cat downsampled to 1% + SD3.5L-Turbo                     | 0.817         | N/A                    | 0.094              |
| Cat downsampled to 1% + SDXL                             | 0.811         | N/A                    | 0.061              |
| Cat downsampled to 1% + SDXL + similarity filter         | 0.815         | N/A                    | 0.075              |
| Cat downsampled to 1% + SDXL + LoRA                      | 0.819         | N/A                    | 0.117              |
| Cat downsampled to 1% + class weighting                  | 0.766         | 0.764                  | 0.041              |
| Cat downsampled to 1% + label smoothing                  | 0.808         | 0.807                  | 0.000              |
| Downsampled + more augmentations                         | 0.150         | 0.148                  | 0.000              |
| Downsampled + random-oversample                          | 0.812         | 0.812                  | 0.008              |
| Downsampled + random-oversample + more augmentations     | 0.147         | 0.148                  | 0.000              |
| Downsampled + random-undersample                         | 0.217         | 0.217                  | 0.119              |
| Downsampled + SMOTE                                      | 0.811         | 0.809                  | 0.012              |
| FLUX1.dev                                                | 0.815         | 0.813                  | 0.039              |
| flux1.schnell                                            | 0.810         | 0.808                  | 0.022              |
| Full CIFAR10                                             | 0.868         | N/A                    | 0.748              |
| Full CIFAR10 + Full Synthetic                            | 0.871         | N/A                    | 0.741              |
| Full data + more augmentations                           | 0.105         | 0.105                  | 0.006              |
| stable-diffusion-3.5-large-turbo                         | 0.817         | 0.817                  | 0.094              |

## Data Generation

Below are the methods and scripts used for generating and augmenting the dataset with images.

### 1. Image Generation using BeautifulPrompt (SDXL-Turbo)

This method utilized an initial prompt generation strategy based on the BeautifulPrompt approach ([BeautifulPrompt: Towards Automatic Prompt Engineering for Text-to-Image Synthesis](https://arxiv.org/abs/2305.14728)). It focused on creating complex prompts for specific classes (e.g., cats) to generate realistic images.

-   **Example prompt structure:**
    ```
    a photo taken from behind of a American Wirehair cat next to the cabinet, looking to the right. The cat has realistic fur textures, intricate details, and sharp features, with soft lighting and a clear focus. The image has a shallow depth of field, emphasizing the cat in fine detail. 8k, cinematic, photorealistic
    ```
-   Various variables were used, such as: `angle` (e.g., "a photo taken from above"), `breed` (100 breeds, e.g., British Longhair), `preposition` (6, e.g., "on"), `furniture` (61 objects, e.g., "sofa"), `gaze` (12 gaze directions, e.g., "looking up").
-   Applied to the `stable-diffusion-3.5-large-turbo` model.

### 2. Image Generation in SYNAuG Style (SDXL-Turbo)

The second approach to generating synthetic data for the `cat` class used a simplified prompt structure, inspired by the "modifier + class" style from SYNAuG research.

-   **Prompt structure:** `"a photo of {selected_modifier} cat"`
-   **Modifiers:** Randomly selected from a list of 22 terms (e.g., "realistic", "detailed", "photorealistic", "studio quality").
-   **Number of images generated:** 7,000 cat images.
-   **Model:** `stabilityai/sdxl-turbo`.

### 3. Generating Extended CIFAR-10 Images (SDXL-Turbo with Expanded Concepts)

The most comprehensive image generation strategy covered all 10 CIFAR-10 classes, utilizing expanded lists of descriptors and contexts to create a highly diverse synthetic dataset.

-   **Prompt structure:** `"{selected_quality} photo of a {selected_descriptor} {class_name} {selected_context}"`
-   **Expanded lists:**
    -   `descriptor_type`: Specific types for each class (e.g., "jet airplane" for `airplane`, "sedan car" for `automobile`, "tabby cat" for `cat`).
    -   `context_action`: Specific actions/contexts for each class (e.g., "flying in the clear blue sky" for `airplane`, "driving on a multi-lane highway" for `automobile`, "sitting comfortably on a plush sofa" for `cat`).
    -   `quality_modifiers`: General quality modifiers (e.g., "photorealistic", "high resolution", "cinematic lighting").
-   **Target number of images:** 10,000 images per class.
-   **Model:** `stabilityai/sdxl-turbo`.

### 4. Flux Redux Augmentation

A data augmentation method using a specific Flux Redux technique was also applied, which transforms existing input images to generate their synthetic variants.

-   **Input:** 50 original images (e.g., cats).
-   **Process:** Each image is passed through the Flux Redux pipeline (`black-forest-labs/FLUX.1-Redux-dev` as prior and `black-forest-labs/FLUX.1-dev` as the main pipeline), with a dynamically changing seed for each augmentation.
-   **Number of augmentations generated:** 99 augmentations per image, resulting in a total of approximately 4950 new examples.

### Resources Used and Generation Times

**Selected models for evaluation and generation (with approximate generation times for 6,000 images):**

-   `stable-diffusion-3.5-large-turbo` - 1.5 hours
-   `black-forest-labs/FLUX.1-schnell` - 2.5 hours
-   `black-forest-labs/FLUX.1-dev` - 14.5 hours
-   `SDXL Turbo` - TBD
-   `FLUX Redux` - TBD

All operations were performed on a Lambda machine with the following configuration: NVIDIA A100 (40GB VRAM), 8x vCPU, 1TB SSD, 1Gbps.