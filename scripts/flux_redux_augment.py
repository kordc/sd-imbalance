# scripts/flux_redux_augment.py

import random
import os
import glob
import torch
from diffusers import FluxPipeline, FluxPriorReduxPipeline
from PIL import Image
from tqdm import tqdm
from typing import Optional, List  # Import List for type hinting


class FluxReduxAugment:
    """A custom torchvision-like transform that applies the Flux Redux augmentation
    with a specified probability. Expects a PIL image as input and returns a PIL image.

    This class initializes two Flux Diffusion pipelines (Prior Redux and main Flux)
    and uses them to generate augmented versions of input images.
    """

    def __init__(
        self,
        guidance_scale: float = 2.5,
        num_inference_steps: int = 50,
        seed: int = 0,
        device: str = "cuda",
        probability: float = 1.0,  # probability to apply the augmentation
    ) -> None:
        """
        Initializes the FluxReduxAugment transform.

        Args:
            guidance_scale (float): Classifier-free guidance scale for the diffusion process.
            num_inference_steps (int): Number of diffusion steps.
            seed (int): Initial random seed for reproducibility; can be updated for each call.
            device (str): The device to load the diffusion models onto (e.g., "cuda" or "cpu").
            probability (float): The probability (0.0 to 1.0) of applying the augmentation.
                                 If a random number is higher than this, the original image is returned.
        """
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.seed = seed
        self.device = device
        self.probability = probability

        print(f"Loading Flux Redux pipelines on device: {self.device}...")
        # Load the Flux Redux pipelines once
        self.flux_prior_redux: FluxPriorReduxPipeline = (
            FluxPriorReduxPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-Redux-dev",
                torch_dtype=torch.bfloat16,
            ).to(self.device)
        )

        self.flux_pipeline: FluxPipeline = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            text_encoder=None,
            text_encoder_2=None,
            torch_dtype=torch.bfloat16,
        ).to(self.device)
        print("Flux Redux pipelines loaded.")

    def __call__(self, image: Image.Image) -> Image.Image:
        """
        Apply the Flux Redux augmentation to the input PIL image with the given probability.

        Args:
            image (Image.Image): The input PIL image to be augmented.

        Returns:
            Image.Image: The augmented image, or the original image if augmentation is skipped.
        """
        # Decide whether to apply the augmentation.
        if random.random() > self.probability:
            # Skip augmentation; return the original image.
            return image

        # Pass the image through the prior redux pipeline.
        # Ensure image is in RGB and suitable for the model
        if image.mode != "RGB":
            image = image.convert("RGB")

        pipe_prior_output = self.flux_prior_redux(image)

        # Create a generator on the correct device and seed it.
        generator = torch.Generator(self.device).manual_seed(self.seed)

        # Generate an augmented image using the Flux pipeline.
        result = self.flux_pipeline(
            guidance_scale=self.guidance_scale,
            num_inference_steps=self.num_inference_steps,
            generator=generator,
            **pipe_prior_output,
        )
        # Return the first image from the result (as a PIL image).
        return result.images[0]


def main() -> None:
    """
    Main function to orchestrate the Flux Redux augmentation process.
    It reads images from a specified input directory, applies Flux Redux augmentation
    multiple times per image, and saves the augmented images to an output directory.
    Output filenames are structured to be compatible with `data.py`'s `_add_extra_images` method.
    """
    input_dir: str = "./notebooks/cats"  # Assumes this directory contains cat images
    output_dir: str = "./notebooks/cats_redux"
    # The class name representing images from input_dir.
    # Here, since input_dir is 'cats', we assume the class is 'cat'.
    # This should be adjusted if input_dir contains other classes.
    image_class_name: str = "cat"

    os.makedirs(output_dir, exist_ok=True)

    # Look for common image file extensions.
    image_paths: List[str] = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))

    # Sort and limit to first 50 images.
    image_paths = sorted(image_paths)[:50]

    if not image_paths:
        print(f"No images found in {input_dir}. Exiting.")
        return

    num_aug_per_image: int = (
        99  # Example: 50 images * 99 augmentations = 4950 new examples
    )

    # Initialize the augmentation instance with probability 1.0.
    flux_augment = FluxReduxAugment(
        guidance_scale=2.5,
        num_inference_steps=50,
        seed=0,  # initial seed; will be updated for each augmentation
        device="cuda" if torch.cuda.is_available() else "cpu",  # Use CUDA if available
        probability=1.0,  # always apply the augmentation
    )

    # Ensure pipelines are loaded before starting processing loop
    try:
        # Accessing an attribute forces lazy loading if not already done, or just confirms device
        _ = flux_augment.flux_pipeline.device
        _ = flux_augment.flux_prior_redux.device
        print(f"Flux pipelines initialized on {flux_augment.device}.")
    except Exception as e:
        print(
            f"Failed to initialize Flux pipelines: {e}. Please check your setup (Hugging Face token, VRAM, etc.). Exiting."
        )
        return

    total_augmented_images: int = 0
    # Use tqdm to show progress over images
    for img_path in tqdm(image_paths, desc="Processing images for Flux Redux"):
        img: Optional[Image.Image] = None
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"\nError opening or converting image {img_path}: {e}. Skipping.")
            continue

        original_file_base_name: str = os.path.splitext(os.path.basename(img_path))[0]

        # Save the original image with a class prefix to match data.py's expectations
        # e.g., "cat_originalfilename_orig.png"
        original_output_name: str = (
            f"{image_class_name}_{original_file_base_name}_orig.png"
        )
        original_output_path: str = os.path.join(output_dir, original_output_name)
        if not os.path.exists(
            original_output_path
        ):  # Avoid saving the original multiple times
            img.save(original_output_path)

        # tqdm for augmentations per image
        for i in tqdm(
            range(num_aug_per_image),
            desc=f"Augmenting {original_file_base_name}",
            leave=False,
        ):
            try:
                # Update seed for variation (use a large range for more unique seeds)
                flux_augment.seed = random.randint(0, 2**32 - 1)
                augmented_img: Image.Image = flux_augment(img)

                # Ensure the output filename starts with the class name
                # e.g., "cat_originalfilename_aug_0.png"
                out_name: str = (
                    f"{image_class_name}_{original_file_base_name}_aug_{i}.png"
                )
                out_path: str = os.path.join(output_dir, out_name)

                # Check if file already exists to avoid overwriting or redundant saving
                if not os.path.exists(out_path):
                    augmented_img.save(out_path)
                    total_augmented_images += 1
            except Exception as e:
                tqdm.write(
                    f"Error augmenting image {img_path} (aug {i}): {e}. Skipping this augmentation."
                )
                # Clear CUDA cache on error to recover memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                torch.cuda.synchronize()  # Ensure operations are complete before clearing

    print(
        f"\nFinished. Total {total_augmented_images} augmented images generated in {output_dir}."
    )


if __name__ == "__main__":
    # Optional: Clear CUDA cache before starting main process to free up memory
    if torch.cuda.is_available():
        print("Clearing CUDA cache before starting...")
        torch.cuda.empty_cache()
        torch.cuda.synchronize()  # Ensure cleanup is complete
        print("CUDA cache cleared.")

    main()
