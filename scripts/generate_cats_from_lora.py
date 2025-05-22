# scripts/generate_cats_from_lora.py

import numpy as np
import random
import torch
from tqdm import tqdm
import os
from typing import List, Dict, Any, Optional  # Added Optional, Dict, Any for typing

# This script generates cat images using a Stable Diffusion XL base model
# combined with a custom LoRA (Low-Rank Adaptation) checkpoint.
# It uses a structured prompting approach (inspired by BeautifulPrompt)
# to create diverse cat images with specific poses, contexts, and camera angles.
# The output filenames are formatted to be compatible with data.py's _add_extra_images method.


def set_seeds(seed: int = 42) -> None:
    """
    Fix all possible seeds to ensure reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to be set for all libraries.
    """
    # Python random module
    random.seed(seed)
    # NumPy
    np.random.seed(seed)
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU
    torch.backends.cudnn.deterministic = True  # Ensures deterministic behavior
    torch.backends.cudnn.benchmark = False  # Disables optimizations for reproducibility
    # Environment variable for other libraries or hash-based operations
    os.environ["PYTHONHASHSEED"] = str(seed)


# Example usage
set_seeds(42)


# Expanded list of cat breeds
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


# Expanded list of prepositions
prepositions: List[str] = ["on", "under", "next to", "beside", "in front of", "behind"]

# Furniture types that fit with the prepositions
furniture_or_outdoor: Dict[str, List[str]] = {
    "on": [
        # Indoor furniture
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
        # Outdoor/natural elements
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
        # Indoor furniture
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
        # Outdoor/natural elements
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
        # Indoor furniture
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
        # Outdoor/natural elements
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
        # Indoor furniture
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
        # Outdoor/natural elements
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
        # Indoor furniture
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
        # Outdoor/natural elements
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
        # Indoor furniture
        "sofa",
        "armchair",
        "bookshelf",
        "recliner",
        "cabinet",
        "TV stand",
        "chest of drawers",
        "love seat",
        # Outdoor/natural elements
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

# List of gaze directions
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
    folder: str = "./tmp",
    num_inference_steps: int = 25,
    guidance_scale: float = 0.0,
    num_images: int = 4,
    pipe: Optional[
        Any
    ] = None,  # Using Any for DiffusionPipeline to avoid circular import if not imported
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
        pipe (Optional[Any]): The pre-loaded DiffusionPipeline object (e.g., StableDiffusionXLPipeline).
                              It's recommended to pass the pipeline to avoid re-loading.
    """
    if pipe is None:
        print(
            "Error: Diffusion pipeline must be provided. Please load it before calling make_img."
        )
        return

    # Directory to save generated images
    output_dir: str = folder
    os.makedirs(output_dir, exist_ok=True)
    set_seeds(42)  # Ensure reproducibility for random choices

    print(f"Starting image generation for {num_images} images into {output_dir}...")
    for i in tqdm(range(num_images), desc="Generating Images"):
        # Randomly select components for the prompt
        breed: str = random.choice(cat_breeds)
        preposition: str = random.choice(prepositions)
        furniture: str = random.choice(furniture_or_outdoor[preposition])
        angle: str = random.choice(camera_angles)
        gaze: str = random.choice(gaze_directions)

        # Note: The original code had redundant random.choice calls for camera_angles and gaze_directions
        # that were not assigned to variables. These are removed for clarity.

        # Construct the prompt using f-strings for readability
        prompt: str = (
            f"{angle} of a {breed} cat {preposition} the {furniture}, {gaze}. "
            "The cat has realistic fur textures, intricate details, and sharp features, "
            "with soft lighting and a clear focus. The image has a shallow depth of field, "
            "emphasizing the cat in fine detail. 8k, cinematic, photorealistic"
        )
        # print(prompt) # Uncomment to see the prompt for each image

        # Ensure pipeline is on the correct dtype for generation
        pipe.to(torch.float16)  # SDXL uses float16 by default for performance

        try:
            # Generate image
            result = pipe(
                prompt,
                num_inference_steps=num_inference_steps,
                # cross_attention_kwargs={"scale": 1.0}, # This might be specific to certain LoRAs, keep if needed
                guidance_scale=guidance_scale,
                width=512,
                height=512,
            )
            image = result.images[0]  # Get the first image from the list

            # Save the generated image
            # Filename format: class_name_index_preposition_furniture.png
            # This format is compatible with data.py's parsing (class_name is "cat")
            output_path: str = os.path.join(
                output_dir,
                f"cat_{i:05d}_{preposition.replace(' ', '_')}_{furniture.replace(' ', '_')}.png",
            )
            image.save(output_path)

        except Exception as e:
            tqdm.write(f"Failed to generate image {i} with prompt '{prompt}': {e}")
            # Clear CUDA cache to recover from potential OOM errors
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            torch.cuda.synchronize()

    print(f"Generated {num_images} images and saved them in {output_dir}.")


if __name__ == "__main__":
    import torch
    from diffusers import DiffusionPipeline
    from huggingface_hub import whoami
    from slugify import slugify
    from pathlib import Path
    import gc  # Import gc for garbage collection

    # Clear CUDA cache before starting
    if torch.cuda.is_available():
        print("Clearing CUDA cache before starting...")
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        print("CUDA cache cleared.")

    print("Loading DiffusionPipeline for LoRA generation...")
    # Load the base pipeline
    try:
        base_pipe = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
        ).to("cuda")  # Ensure it's moved to GPU if available
        print("Base pipeline loaded.")
    except Exception as e:
        print(f"Error loading base pipeline: {e}. Exiting.")
        exit(1)

    # Authenticate with Hugging Face Hub to load private LoRA if necessary
    try:
        # Assuming Hugging Face token is accessible or login has been done
        # The path should point to the directory containing the token, typically ~/.cache/huggingface/token
        # or it can be explicitly set via huggingface_hub.login()
        _ = whoami(
            token=os.getenv(
                "HF_TOKEN", Path("/root/.cache/huggingface/token").read_text().strip()
            )
        )
        print("Hugging Face authentication successful.")
    except Exception as e:
        print(
            f"Hugging Face authentication failed: {e}. Make sure you are logged in (huggingface-cli login) or HF_TOKEN env var is set."
        )
        # Proceed without LoRA if auth fails, or exit if LoRA is critical
        pass  # Allow script to continue, it might work if LoRA is public or not strictly needed by the pipe

    # Define LoRA parameters
    username = whoami(
        token=os.getenv(
            "HF_TOKEN", Path("/root/.cache/huggingface/token").read_text().strip()
        )
    )["name"]
    output_dir_slug = slugify("cifar10cats")  # Slugify for a safe directory name
    repo_id = f"{username}/{output_dir_slug}"

    print(f"Attempting to load LoRA weights from {repo_id}...")
    try:
        # Load LoRA weights onto the base pipeline
        base_pipe.load_lora_weights(
            repo_id, weight_name="pytorch_lora_weights.safetensors"
        )
        print("LoRA weights loaded successfully.")
    except Exception as e:
        print(
            f"Error loading LoRA weights: {e}. Ensure the LoRA exists and is accessible. Proceeding without LoRA."
        )
        # If LoRA loading fails, you might want to exit or handle this differently based on requirements.

    # Call the image generation function
    # The output folder name also includes "cat" to be compatible with data.py's parsing
    make_img(
        folder="./lora_generated_cats",  # A specific folder for LoRA generated cats
        num_images=100000,  # Example number, adjust as needed
        pipe=base_pipe,
        num_inference_steps=25,  # Adjusted based on common usage for SDXL
        guidance_scale=7.0,  # Common guidance scale for SDXL
    )

    print("Script finished.")
    # Final cleanup of the pipeline
    if "base_pipe" in locals() and base_pipe is not None:
        del base_pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
