import random
import os
import glob
import torch
from diffusers import FluxPipeline, FluxPriorReduxPipeline
from PIL import Image
from tqdm import tqdm
from typing import Optional, List
import argparse  # New import for command-line arguments


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
        probability: float = 1.0,
        prior_redux_model_id: str = "black-forest-labs/FLUX.1-Redux-dev",
        flux_model_id: str = "black-forest-labs/FLUX.1-dev",
        torch_dtype: torch.dtype = torch.bfloat16,
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
            prior_redux_model_id (str): Hugging Face model ID for FluxPriorReduxPipeline.
            flux_model_id (str): Hugging Face model ID for FluxPipeline.
            torch_dtype (torch.dtype): The torch data type to use for model weights (e.g., torch.bfloat16, torch.float16).
        """
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.seed = seed
        self.device = device
        self.probability = probability
        self.prior_redux_model_id = prior_redux_model_id
        self.flux_model_id = flux_model_id
        self.torch_dtype = torch_dtype

        print(
            f"Loading Flux Redux pipelines on device: {self.device} with dtype {self.torch_dtype}..."
        )

        self.flux_prior_redux: FluxPriorReduxPipeline = (
            FluxPriorReduxPipeline.from_pretrained(
                self.prior_redux_model_id,
                torch_dtype=self.torch_dtype,
            ).to(self.device)
        )

        self.flux_pipeline: FluxPipeline = FluxPipeline.from_pretrained(
            self.flux_model_id,
            text_encoder=None,  # Not used by Flux Redux pipeline
            text_encoder_2=None,  # Not used by Flux Redux pipeline
            torch_dtype=self.torch_dtype,
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
        if random.random() > self.probability:
            return image

        if image.mode != "RGB":
            image = image.convert("RGB")

        # The prior redux pipeline usually expects a PIL image directly
        pipe_prior_output = self.flux_prior_redux(image)

        generator = torch.Generator(self.device).manual_seed(self.seed)

        result = self.flux_pipeline(
            guidance_scale=self.guidance_scale,
            num_inference_steps=self.num_inference_steps,
            generator=generator,
            **pipe_prior_output,
        )
        return result.images[0]


def main() -> None:
    """
    Main function to orchestrate the Flux Redux augmentation process.
    It reads images from a specified input directory, applies Flux Redux augmentation
    multiple times per image, and saves the augmented images to an output directory.
    Output filenames are structured to be compatible with `data.py`'s `_add_extra_images` method.
    """
    parser = argparse.ArgumentParser(
        description="Apply Flux Redux augmentation to images from a directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,  # Show default values in help
    )

    # General / I/O arguments
    parser.add_argument(
        "--input_dir",
        type=str,
        default="./generated_data/cats_chosen",
        help="Directory containing input images.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./generated_data/cats_redux",
        help="Directory to save augmented images.",
    )
    parser.add_argument(
        "--image_class_name",
        type=str,
        default="cat",
        help="Name of the image class, used for output filenames (e.g., 'cat').",
    )
    parser.add_argument(
        "--num_aug_per_image",
        type=int,
        default=99,
        help="Number of augmented images to generate per input image.",
    )
    parser.add_argument(
        "--max_input_images",
        type=int,
        default=50,
        help="Maximum number of input images to process. Set to -1 to process all.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="If set, skips processing input images that already have their augmented outputs in the output directory. Useful for resuming interrupted runs.",
    )

    # FluxReduxAugment specific arguments
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=2.5,
        help="Classifier-free guidance scale for the diffusion process.",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of diffusion steps.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Initial random seed for the diffusion process. "
        "If --fixed_seed_per_aug is not set, this seed is used for the first augmentation, "
        "and subsequent augmentations for the same image will use random seeds.",
    )
    parser.add_argument(
        "--fixed_seed_per_aug",
        action="store_true",
        help="If set, the initial --seed will be used for ALL augmentations (i.e., fixed for each new image and each augmentation for that image), "
        "instead of randomizing it per augmentation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="The device to load the diffusion models onto (e.g., 'cuda' or 'cpu').",
    )
    parser.add_argument(
        "--probability",
        type=float,
        default=1.0,
        help="The probability (0.0 to 1.0) of applying the augmentation. "
        "If a random number is higher than this, the original image is returned without augmentation.",
    )
    parser.add_argument(
        "--prior_redux_model_id",
        type=str,
        default="black-forest-labs/FLUX.1-Redux-dev",
        help="Hugging Face model ID for FluxPriorReduxPipeline.",
    )
    parser.add_argument(
        "--flux_model_id",
        type=str,
        default="black-forest-labs/FLUX.1-dev",
        help="Hugging Face model ID for FluxPipeline.",
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="The torch data type to use for model weights.",
    )

    args = parser.parse_args()

    # Convert torch_dtype string to actual torch.dtype
    if args.torch_dtype == "float16":
        torch_dtype_val = torch.float16
    elif args.torch_dtype == "bfloat16":
        torch_dtype_val = torch.bfloat16
    elif args.torch_dtype == "float32":
        torch_dtype_val = torch.float32
    else:  # This should not be reached due to 'choices' in argparse
        raise ValueError(f"Unknown torch_dtype: {args.torch_dtype}")

    # Use arguments for main function variables
    input_dir: str = args.input_dir
    output_dir: str = args.output_dir
    image_class_name: str = args.image_class_name
    num_aug_per_image: int = args.num_aug_per_image
    max_input_images: int = args.max_input_images
    fixed_seed_per_aug: bool = args.fixed_seed_per_aug
    skip_existing: bool = args.skip_existing

    os.makedirs(output_dir, exist_ok=True)

    image_paths: List[str] = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))

    image_paths = sorted(image_paths)
    if max_input_images != -1:
        image_paths = image_paths[:max_input_images]

    if not image_paths:
        print(f"No images found in {input_dir}. Exiting.")
        return

    # Initialize FluxReduxAugment with parsed arguments
    flux_augment = FluxReduxAugment(
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
        device=args.device,
        probability=args.probability,
        prior_redux_model_id=args.prior_redux_model_id,
        flux_model_id=args.flux_model_id,
        torch_dtype=torch_dtype_val,  # Use the converted dtype
    )

    try:
        _ = flux_augment.flux_pipeline.device
        _ = flux_augment.flux_prior_redux.device
        print(f"Flux pipelines initialized on {flux_augment.device}.")
    except Exception as e:
        print(
            f"Failed to initialize Flux pipelines: {e}. "
            "Please check your setup (Hugging Face token, VRAM, etc.). Exiting."
        )
        return

    total_augmented_images: int = 0
    for img_path in tqdm(image_paths, desc="Processing images for Flux Redux"):
        original_file_base_name: str = os.path.splitext(os.path.basename(img_path))[0]
        original_output_name: str = (
            f"{image_class_name}_{original_file_base_name}_orig.png"
        )
        original_output_path: str = os.path.join(output_dir, original_output_name)

        # Optimization: Check if all expected files for this original image already exist
        if skip_existing:
            all_aug_exist = True
            for i in range(num_aug_per_image):
                out_name: str = (
                    f"{image_class_name}_{original_file_base_name}_aug_{i}.png"
                )
                out_path: str = os.path.join(output_dir, out_name)
                if not os.path.exists(out_path):
                    all_aug_exist = False
                    break
            if all_aug_exist and os.path.exists(original_output_path):
                tqdm.write(
                    f"Skipping '{original_file_base_name}': all augmented files already exist."
                )
                continue

        img: Optional[Image.Image] = None
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"\nError opening or converting image {img_path}: {e}. Skipping.")
            continue

        # Save original image if it doesn't exist
        if not os.path.exists(original_output_path):
            img.save(original_output_path)

        for i in tqdm(
            range(num_aug_per_image),
            desc=f"Augmenting {original_file_base_name}",
            leave=False,
        ):
            try:
                # Decide the seed for this augmentation based on --fixed_seed_per_aug
                if fixed_seed_per_aug:
                    flux_augment.seed = args.seed  # Use the fixed seed from arguments
                else:
                    flux_augment.seed = random.randint(
                        0, 2**32 - 1
                    )  # Randomize for each aug

                out_name: str = (
                    f"{image_class_name}_{original_file_base_name}_aug_{i}.png"
                )
                out_path: str = os.path.join(output_dir, out_name)

                if os.path.exists(out_path) and skip_existing:
                    continue  # Skip this specific augmentation if it already exists and skip_existing is True

                augmented_img: Image.Image = flux_augment(img)

                augmented_img.save(out_path)
                total_augmented_images += 1
            except Exception as e:
                tqdm.write(
                    f"Error augmenting image {img_path} (aug {i}): {e}. Skipping this augmentation."
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                torch.cuda.synchronize()
    print(
        f"\nFinished. Total {total_augmented_images} augmented images generated in {output_dir}."
    )


if __name__ == "__main__":
    if torch.cuda.is_available():
        print("Clearing CUDA cache before starting...")
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("CUDA cache cleared.")

    main()
