import os
import random
import torch
from diffusers import StableDiffusionXLPipeline
from tqdm import tqdm
import gc
from typing import List, Optional  # Import List, Optional, Any for type hinting
import argparse  # Import argparse


# --- GLOBAL MODIFIER LIST (can be overridden or augmented via CLI) ---
cat_modifiers: List[str] = [
    "realistic",
    "detailed",
    "simple",
    "studio quality",
    "professional",
    "photorealistic",
    "high resolution",
    "clear",
    "sharp focus",
    "natural light",
    "soft light",
    "cinematic",
    "close-up",
    "full body",
    "adorable",
    "fluffy",
    "sleek",
    "sitting",
    "sleeping",
    "looking at camera",
    "illustration style",
    "painted",
    "sketch style",
]


def set_seeds(seed: int = 42) -> None:
    """
    Fix all possible seeds to ensure reproducibility across Python, NumPy (if used), and PyTorch.

    Args:
        seed (int): The seed value to be set for all libraries.
    """
    random.seed(seed)
    # NumPy is not explicitly used for random numbers in this script,
    # but including it can be good practice if other dependencies use it.
    # import numpy as np # Uncomment if np is used elsewhere for randomness
    # np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  # Set to False for reproducibility
    os.environ["PYTHONHASHSEED"] = str(seed)


def make_img(
    folder: str,
    num_images: int,
    modifier_list: List[str],
    model_id: str,
    torch_dtype: torch.dtype,
    variant: Optional[str],
    use_safetensors: bool,
    device: str,
    guidance_scale: float,
    num_inference_steps: int,
    image_width: int,
    image_height: int,
    output_format: str,
    output_filename_prefix: str,
    hf_token: Optional[str] = None,
) -> None:
    """
    Generates cat images using the SDXL-Turbo pipeline with a simplified
    prompting style based on the SYNAuG paper (modifier + class).

    Args:
        folder (str): The directory where generated images will be saved.
        num_images (int): The total number of images to generate.
        modifier_list (List[str]): A list of descriptive modifiers to be used
                                   in image prompts.
        model_id (str): Hugging Face model ID for the Stable Diffusion XL Turbo pipeline.
        torch_dtype (torch.dtype): The torch data type to use for model weights.
        variant (Optional[str]): Model variant to load (e.g., "fp16").
        use_safetensors (bool): Whether to use safetensors for model loading.
        device (str): The device to load the diffusion model onto (e.g., "cuda" or "cpu").
        guidance_scale (float): Classifier-free guidance scale.
        num_inference_steps (int): Number of inference steps.
        image_width (int): Width of the generated images.
        image_height (int): Height of the generated images.
        output_format (str): Format to save the images (e.g., 'png', 'jpeg').
        output_filename_prefix (str): Prefix for the generated image filenames.
        hf_token (Optional[str]): Hugging Face authentication token for model download.
    """
    if not modifier_list:
        print("Error: The modifier list is empty. Please provide valid modifiers.")
        return

    output_dir: str = folder
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving images to: {output_dir}")

    print(f"Loading SDXL-Turbo pipeline: {model_id}...")
    pipe: Optional[StableDiffusionXLPipeline] = None
    try:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            variant=variant,
            use_safetensors=use_safetensors,
            token=hf_token,  # Pass HF token
        )
        print("Pipeline loaded.")
    except Exception as e:
        print(f"Error loading pipeline '{model_id}': {e}")
        print(
            "Please ensure you have the necessary libraries installed, sufficient VRAM, "
            "and are logged in (`huggingface-cli login`) or provided a token via --hf_token."
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return

    print(f"Moving pipeline to {device}...")
    try:
        pipe = pipe.to(device)
        print(f"Pipeline moved to {device}.")
    except Exception as e:
        print(f"Error moving pipeline to {device}: {e}. Exiting.")
        if pipe is not None:
            del pipe
        gc.collect()
        return

    print(f"Starting image generation for {num_images} images...")
    for i in tqdm(range(num_images), desc="Generating Images"):
        selected_modifier: str = random.choice(modifier_list)
        prompt: str = f"a photo of {selected_modifier} cat"

        try:
            with torch.inference_mode():
                image_result = pipe(
                    prompt=prompt,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    width=image_width,
                    height=image_height,
                )
                if image_result and image_result.images:
                    image_to_save = image_result.images[0]
                else:
                    tqdm.write(
                        f"\nWarning: Image generation failed for prompt: '{prompt}'. No image returned."
                    )
                    continue

            # Sanitize modifier for filename to avoid issues with special characters
            sanitized_modifier = (
                selected_modifier.replace(" ", "_")
                .replace("/", "_")
                .replace("\\", "_")
                .replace(":", "_")
            )
            output_path: str = os.path.join(
                output_dir,
                f"{output_filename_prefix}_{i:05d}_{sanitized_modifier}.{output_format}",
            )

            image_to_save.save(output_path)

        except torch.cuda.OutOfMemoryError:
            tqdm.write(
                f"\nCUDA OutOfMemoryError for image {i} with prompt: '{prompt}'. Skipping, clearing cache."
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.synchronize()
            continue
        except Exception as e:
            tqdm.write(
                f"\nError generating image {i} with prompt: '{prompt}': {e}. Skipping."
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.synchronize()
            continue

    print(f"\nFinished generating {num_images} images.")
    # Clean up pipeline resources
    if pipe is not None:
        del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate cat images using SDXL-Turbo with SYNAuG-style prompting.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # General / Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./sdxl_turbo_synaug_style",
        help="Directory to save the generated images.",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=10000,
        help="Total number of images to generate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility across Python, NumPy (if used), and PyTorch.",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        default="png",
        choices=["png", "jpeg", "webp"],
        help="Output image format for generated images.",
    )
    parser.add_argument(
        "--output_filename_prefix",
        type=str,
        default="cat",
        help="Prefix for the generated image filenames (e.g., 'cat_00001_...').",
    )
    parser.add_argument(
        "--modifier_file",
        type=str,
        default=None,
        help="Path to a text file containing modifiers, one per line. If not provided, a default internal list will be used.",
    )

    # Diffusion Pipeline / Model arguments
    parser.add_argument(
        "--model_id",
        type=str,
        default="stabilityai/sdxl-turbo",
        help="Hugging Face model ID for the Stable Diffusion XL Turbo pipeline.",
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="The torch data type to use for model weights.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="fp16",  # Default for SDXL-Turbo
        help="Model variant to load (e.g., 'fp16'). Set to 'None' to disable variant loading. (String 'None' will be converted to actual None).",
    )
    parser.add_argument(
        "--use_safetensors",
        action="store_true",  # Default behavior for SDXL-Turbo is to use safetensors
        help="Whether to use safetensors for model loading (default behavior for SDXL-Turbo).",
    )
    parser.add_argument(
        "--no_safetensors",
        action="store_false",
        dest="use_safetensors",  # This makes --no_safetensors set use_safetensors to False
        help="Explicitly disable using safetensors for model loading.",
    )
    parser.set_defaults(
        use_safetensors=True
    )  # Set the true default for use_safetensors

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="The device to load the diffusion model onto (e.g., 'cuda' or 'cpu').",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=0.0,
        help="Classifier-free guidance scale. SDXL-Turbo typically uses 0.0.",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=1,
        help="Number of inference steps. SDXL-Turbo typically uses 1.",
    )
    parser.add_argument(
        "--image_width",
        type=int,
        default=512,
        help="Width of the generated images.",
    )
    parser.add_argument(
        "--image_height",
        type=int,
        default=512,
        help="Height of the generated images.",
    )
    # Hugging Face authentication token
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Hugging Face authentication token. Overrides HF_TOKEN environment variable. Required for private models.",
    )

    args = parser.parse_args()

    # Set seeds at the very beginning based on CLI arg for full reproducibility
    set_seeds(args.seed)

    # Convert torch_dtype string to actual torch.dtype
    if args.torch_dtype == "float16":
        torch_dtype_val = torch.float16
    elif args.torch_dtype == "bfloat16":
        torch_dtype_val = torch.bfloat16
    elif args.torch_dtype == "float32":
        torch_dtype_val = torch.float32
    else:  # This should not be reached due to 'choices' in argparse
        raise ValueError(f"Unknown torch_dtype: {args.torch_dtype}")

    # Convert variant string "None" to actual None
    variant_val = args.variant if args.variant.lower() != "none" else None

    # Load modifiers from file if provided, otherwise use default
    current_modifier_list: List[str] = []
    if args.modifier_file:
        try:
            with open(args.modifier_file, "r") as f:
                current_modifier_list = [line.strip() for line in f if line.strip()]
            print(
                f"Loaded {len(current_modifier_list)} modifiers from {args.modifier_file}."
            )
            if not current_modifier_list:
                print(
                    "Warning: The provided modifier file is empty. Falling back to default internal modifiers."
                )
                current_modifier_list = cat_modifiers
        except FileNotFoundError:
            print(
                f"Error: Modifier file not found at {args.modifier_file}. Falling back to default internal modifiers."
            )
            current_modifier_list = cat_modifiers
        except Exception as e:
            print(
                f"Error reading modifier file {args.modifier_file}: {e}. Falling back to default internal modifiers."
            )
            current_modifier_list = cat_modifiers
    else:
        current_modifier_list = cat_modifiers  # Use the internal default list

    # Clear CUDA cache before starting, useful for fresh runs or when switching models
    if torch.cuda.is_available():
        print("Clearing CUDA cache before starting...")
        torch.cuda.empty_cache()
        gc.collect()  # Explicitly collect garbage
        torch.cuda.synchronize()
        print("CUDA cache cleared.")

    # Pass token to make_img function. Prioritize CLI arg, then env var.
    hf_token_to_use = args.hf_token or os.getenv("HF_TOKEN")

    # Call make_img with all collected parameters
    make_img(
        folder=args.output_dir,
        num_images=args.num_images,
        modifier_list=current_modifier_list,
        model_id=args.model_id,
        torch_dtype=torch_dtype_val,
        variant=variant_val,
        use_safetensors=args.use_safetensors,
        device=args.device,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        image_width=args.image_width,
        image_height=args.image_height,
        output_format=args.output_format,
        output_filename_prefix=args.output_filename_prefix,
        hf_token=hf_token_to_use,
    )

    print("Script finished.")
    # Final cleanup (might be redundant if already done in make_img, but harmless)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
