import numpy as np
import random
import torch
from tqdm import tqdm
import os
from typing import List, Dict, Any, Optional
import argparse  # Import argparse
import gc  # For garbage collection

""" 
This script generates cat images using a Stable Diffusion XL base model
combined with an optional custom LoRA (Low-Rank Adaptation) checkpoint.
It uses a structured prompting approach (inspired by BeautifulPrompt)
to create diverse cat images with specific poses, contexts, and camera angles.
The output filenames are formatted to be compatible with data.py's _add_extra_images method.
"""


def set_seeds(seed: int = 42) -> None:
    """
    Fix all possible seeds to ensure reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to be set for all libraries.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  # Set to False for reproducibility
    os.environ["PYTHONHASHSEED"] = str(seed)


# --- Prompting Lists (kept internal as their structure is part of the generation logic) ---
cat_breeds: List[str] = [
    "Abyssinian",
    "Aegean",
    "American Bobtail",
    "American Curl",
    "American Ringtail",
    "American Shorthair",
    "American Wirehair",
    "Aphrodite Giant",
    "Arabian Mau",
    "Asian",
    "Asian Semi-longhair",
    "Australian Mist",
    "Balinese",
    "Bambino",
    "Bengal",
    "Birman",
    "Bombay",
    "Brazilian Shorthair",
    "British Longhair",
    "British Shorthair",
    "Burmese",
    "Burmilla",
    "California Spangled",
    "Chantilly-Tiffany",
    "Chartreux",
    "Chausie",
    "Colorpoint Shorthair",
    "Cornish Rex",
    "Cymric",
    "Cyprus",
    "Devon Rex",
    "Donskoy",
    "Dragon Li",
    "Dwelf",
    "Egyptian Mau",
    "European Shorthair",
    "Exotic Shorthair",
    "Foldex",
    "German Rex",
    "Havana Brown",
    "Highlander",
    "Himalayan",
    "Japanese Bobtail",
    "Javanese",
    "Kanaani",
    "Karelian Bobtail",
    "Khao Manee",
    "Kinkalow",
    "Korat",
    "Korean Bobtail",
    "Korn Ja",
    "Kurilian Bobtail",
    "Lambkin",
    "LaPerm",
    "Lykoi",
    "Maine Coon",
    "Manx",
    "Mekong Bobtail",
    "Minskin",
    "Minuet",
    "Munchkin",
    "Nebelung",
    "Neva Masquerade",
    "Norwegian Forest cat",
    "Ocicat",
    "Ojos Azules",
    "Oriental Bicolor",
    "Oriental Longhair",
    "Oriental Shorthair",
    "Persian",
    "Peterbald",
    "Pixie-bob",
    "Ragamuffin",
    "Ragdoll",
    "Raas",
    "Russian Blue",
    "Russian White, Russian Black and Russian Tabby",
    "Sam Sawet",
    "Savannah",
    "Scottish Fold",
    "Selkirk Rex",
    "Serengeti",
    "Siamese",
    "Siberian",
    "Singapura",
    "Snowshoe",
    "Sokoke",
    "Somali",
    "Sphynx",
    "Suphalak",
    "Thai",
    "Thai Lilac",
    "Tonkinese",
    "Toybob",
    "Toyger",
    "Turkish Angora",
    "Turkish Van",
    "Turkish Vankedisi",
    "Ukrainian Levkoy",
    "York Chocolate",
]

prepositions: List[str] = ["on", "under", "next to", "beside", "in front of", "behind"]

furniture_or_outdoor: Dict[str, List[str]] = {
    "on": [
        "sofa",
        "armchair",
        "dining table",
        "coffee table",
        "bed",
        "bookshelf",
        "cabinet",
        "cupboard",
        "chair",
        "desk",
        "ottoman",
        "recliner",
        "side table",
        "nightstand",
        "TV stand",
        "TV cabinet",
        "end table",
        "love seat",
        "couch",
        "lounge chair",
        "bean bag",
        "bar stool",
        "console table",
        "chest of drawers",
        "vanity",
        "shelf",
        "rock",
        "tree stump",
        "fence",
        "bench",
        "picnic table",
        "grass",
        "car hood",
        "roof",
        "garden wall",
        "park bench",
        "log",
        "fallen tree",
        "sand dune",
        "boulder",
    ],
    "under": [
        "sofa",
        "bed",
        "coffee table",
        "side table",
        "desk",
        "recliner",
        "nightstand",
        "chest of drawers",
        "love seat",
        "couch",
        "bookshelf",
        "tree",
        "bush",
        "bridge",
        "rock",
        "table",
        "park bench",
        "porch",
        "car",
        "wooden deck",
        "fallen tree",
        "shade",
        "overhang",
    ],
    "next to": [
        "sofa",
        "armchair",
        "dining table",
        "coffee table",
        "bookshelf",
        "cabinet",
        "cupboard",
        "chair",
        "ottoman",
        "recliner",
        "side table",
        "nightstand",
        "TV stand",
        "end table",
        "love seat",
        "lounge chair",
        "console table",
        "vanity",
        "window sill",
        "tree",
        "bush",
        "fence",
        "bench",
        "log",
        "rock",
        "flower bed",
        "pathway",
        "pond",
        "stream",
        "hill",
        "park bench",
        "fire hydrant",
        "fallen tree",
    ],
    "beside": [
        "sofa",
        "armchair",
        "dining table",
        "coffee table",
        "bed",
        "bookshelf",
        "chair",
        "ottoman",
        "recliner",
        "side table",
        "nightstand",
        "love seat",
        "couch",
        "lounge chair",
        "vanity",
        "tree",
        "rock",
        "bush",
        "fence",
        "bench",
        "stream",
        "pathway",
        "log",
        "garden wall",
        "pond",
        "flower bed",
        "hill",
        "fallen tree",
    ],
    "in front of": [
        "sofa",
        "armchair",
        "coffee table",
        "cabinet",
        "cupboard",
        "TV stand",
        "console table",
        "chest of drawers",
        "bed",
        "recliner",
        "side table",
        "dining chairs",
        "love seat",
        "tree",
        "rock",
        "fence",
        "bush",
        "bench",
        "stream",
        "pathway",
        "building",
        "garden wall",
        "pond",
        "waterfall",
        "hill",
        "statue",
    ],
    "behind": [
        "sofa",
        "armchair",
        "bookshelf",
        "recliner",
        "cabinet",
        "TV stand",
        "chest of drawers",
        "love seat",
        "tree",
        "bush",
        "fence",
        "rock",
        "log",
        "bench",
        "building",
        "hill",
        "statue",
        "garden wall",
        "shed",
    ],
}
camera_angles: List[str] = [
    "a photo taken from above",
    "a photo taken from below",
    "a side-view photo",
    "a front-facing photo",
    "a photo taken from behind",
]

gaze_directions: List[str] = [
    "looking straight ahead",
    "looking up",
    "looking down",
    "looking to the left",
    "looking to the right",
    "looking up and to the left",
    "looking up and to the right",
    "looking down and to the left",
    "looking down and to the right",
    "eyes closed",
    "looking over its shoulder",
]


def make_img(
    folder: str,
    num_inference_steps: int,
    guidance_scale: float,
    num_images: int,
    pipe: Any,  # DiffusionPipeline object, required
    image_width: int,
    image_height: int,
    output_format: str,
    output_filename_prefix: str,
) -> None:
    """
    Generates cat images using a given Stable Diffusion XL pipeline (potentially with LoRA).
    It constructs prompts based on predefined lists of cat breeds, prepositions, furniture/outdoor elements,
    camera angles, and gaze directions, along with quality modifiers.

    Args:
        folder (str): The directory to save the generated images.
        num_inference_steps (int): Number of inference steps for the diffusion model.
        guidance_scale (float): Classifier-free guidance scale.
        num_images (int): The total number of images to generate.
        pipe (Any): The pre-loaded DiffusionPipeline object (e.g., StableDiffusionXLPipeline).
        image_width (int): Width of the generated images.
        image_height (int): Height of the generated images.
        output_format (str): Format to save the images (e.g., 'png', 'jpeg').
        output_filename_prefix (str): Prefix for the generated image filenames.
    """
    output_dir: str = folder
    os.makedirs(output_dir, exist_ok=True)

    print(f"Starting image generation for {num_images} images into {output_dir}...")
    for i in tqdm(range(num_images), desc="Generating Images"):
        # Random choices driven by the seed set in main
        breed: str = random.choice(cat_breeds)
        preposition: str = random.choice(prepositions)
        furniture: str = random.choice(furniture_or_outdoor[preposition])
        angle: str = random.choice(camera_angles)
        gaze: str = random.choice(gaze_directions)

        prompt: str = (
            f"{angle} of a {breed} cat {preposition} the {furniture}, {gaze}. "
            "The cat has realistic fur textures, intricate details, and sharp features, "
            "with soft lighting and a clear focus. The image has a shallow depth of field, "
            "emphasizing the cat in fine detail. 8k, cinematic, photorealistic"
        )

        try:
            result = pipe(
                prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=image_width,
                height=image_height,
            )
            image = result.images[0]

            # Construct output filename
            output_path: str = os.path.join(
                output_dir,
                f"{output_filename_prefix}_{i:05d}_{preposition.replace(' ', '_')}_{furniture.replace(' ', '_')}.{output_format}",
            )
            image.save(output_path)

        except Exception as e:
            tqdm.write(f"Failed to generate image {i} with prompt '{prompt}': {e}")
            # Attempt to clear CUDA cache on error, useful for OOM
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            torch.cuda.synchronize()

    print(f"Generated {num_images} images and saved them in {output_dir}.")


if __name__ == "__main__":
    from diffusers import DiffusionPipeline
    from huggingface_hub import whoami
    from slugify import slugify  # Used in original for default repo_id creation
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Generate cat images using Stable Diffusion XL with optional LoRA.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # General / Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./lora_generated_cats",
        help="Directory to save the generated images.",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=100000,
        help="Total number of images to generate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility across Python, NumPy, and PyTorch.",
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

    # Diffusion Pipeline / Model arguments
    parser.add_argument(
        "--base_model_id",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="Hugging Face model ID for the base Stable Diffusion XL pipeline.",
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="The torch data type to use for model weights.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="The device to load the diffusion model onto (e.g., 'cuda' or 'cpu').",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=25,
        help="Number of inference steps for the diffusion model.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.0,
        help="Classifier-free guidance scale.",
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

    # LoRA arguments
    parser.add_argument(
        "--lora_repo_id",
        type=str,
        default=None,
        help="Hugging Face repository ID for the LoRA weights (e.g., 'your_username/your_lora_repo'). "
        "If not provided and authenticated, attempts to use 'HF_USERNAME/cifar10cats'.",
    )
    parser.add_argument(
        "--lora_weight_name",
        type=str,
        default="pytorch_lora_weights.safetensors",
        help="Filename of the LoRA weights within the repository.",
    )
    parser.add_argument(
        "--no_lora",
        action="store_true",
        help="If set, skip loading any LoRA weights, even if lora_repo_id is provided.",
    )

    # Hugging Face authentication token
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Hugging Face authentication token. Overrides HF_TOKEN environment variable.",
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

    # Clear CUDA cache before starting, useful for fresh runs or when switching models
    if torch.cuda.is_available():
        print("Clearing CUDA cache before starting...")
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()  # Explicitly collect garbage
        print("CUDA cache cleared.")

    # Authenticate to Hugging Face Hub (optional, but good practice for private repos or rate limits)
    hf_token = args.hf_token or os.getenv("HF_TOKEN")
    if hf_token is None and Path("~/.cache/huggingface/token").expanduser().exists():
        try:
            hf_token = (
                Path("~/.cache/huggingface/token").expanduser().read_text().strip()
            )
        except Exception:
            pass  # Ignore if token file can't be read

    username = None
    if hf_token:
        try:
            username = whoami(token=hf_token)["name"]
            print(f"Hugging Face authentication successful for user: {username}.")
        except Exception as e:
            print(
                f"Warning: Hugging Face authentication failed with provided token: {e}."
            )
            print(
                "         Some models/LoRAs might require authentication to download."
            )
            hf_token = None  # Invalidate token if auth fails
    else:
        print(
            "No Hugging Face token provided or found. Some models/LoRAs might not be accessible."
        )

    print(f"Loading DiffusionPipeline base model: {args.base_model_id}...")
    try:
        # Load the base pipeline
        base_pipe = DiffusionPipeline.from_pretrained(
            args.base_model_id,
            torch_dtype=torch_dtype_val,
            variant="fp16"
            if torch_dtype_val == torch.float16
            else None,  # Use variant for fp16
            token=hf_token,  # Pass token to from_pretrained
        ).to(args.device)
        print("Base pipeline loaded.")
    except Exception as e:
        print(f"Error loading base pipeline '{args.base_model_id}': {e}. Exiting.")
        exit(1)

    # LoRA loading logic
    lora_repo_id_to_load = args.lora_repo_id
    if not args.no_lora:
        if lora_repo_id_to_load is None:
            # Fallback to original derivation if no lora_repo_id provided AND authentication was successful
            if username:
                output_dir_slug = slugify(
                    "cifar10cats"
                )  # This slug is specific to the original example's training context
                lora_repo_id_to_load = f"{username}/{output_dir_slug}"
                print(
                    f"No LoRA repo ID specified, defaulting to {lora_repo_id_to_load} based on authenticated user."
                )
            else:
                print(
                    "No LoRA repo ID specified and not authenticated to derive default. Skipping LoRA loading."
                )
                lora_repo_id_to_load = None  # Ensure it remains None

        if lora_repo_id_to_load:
            print(
                f"Attempting to load LoRA weights from {lora_repo_id_to_load} ({args.lora_weight_name})..."
            )
            try:
                base_pipe.load_lora_weights(
                    lora_repo_id_to_load,
                    weight_name=args.lora_weight_name,
                    token=hf_token,  # Pass token to load_lora_weights
                )
                print("LoRA weights loaded successfully.")
            except Exception as e:
                print(
                    f"Error loading LoRA weights from '{lora_repo_id_to_load}': {e}. "
                    "Ensure the LoRA exists and is accessible. Proceeding without LoRA."
                )
        else:
            print(
                "Skipping LoRA loading as no repository ID was provided or could be derived."
            )
    else:
        print("LoRA loading explicitly skipped by --no_lora flag.")

    # Call make_img with all collected parameters
    make_img(
        folder=args.output_dir,
        num_images=args.num_images,
        pipe=base_pipe,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        image_width=args.image_width,
        image_height=args.image_height,
        output_format=args.output_format,
        output_filename_prefix=args.output_filename_prefix,
    )

    print("Script finished.")
    # Clean up pipeline and clear CUDA cache one last time
    if "base_pipe" in locals() and base_pipe is not None:
        del base_pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
