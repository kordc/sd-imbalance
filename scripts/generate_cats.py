# !pip install diffusers transformers torch tqdm sentencepiece accelerate ipywidgets
# !huggingface-cli login
import os
import random
import torch
from diffusers import StableDiffusionXLPipeline
from tqdm import tqdm
import gc

# --- NEW: List of Modifiers inspired by SYNAuG paper ---
# The paper mentions using 20 modifiers, here's a sample list
cat_modifiers = [
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
    "sketch style",  # Added some style variations
]

# --- REMOVED the previous long list of cat_prompts ---


def make_img(folder: str = "./tmp", num_images=10000, modifier_list=None) -> None:
    """
    Generates cat images using SDXL-Turbo pipeline with a simplified
    prompting style based on the SYNAuG paper (modifier + class).
    """
    if modifier_list is None or len(modifier_list) == 0:
        print("Error: A list of modifiers must be provided.")
        return

    # Directory to save generated images
    output_dir = folder
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving images to: {output_dir}")

    # Load SDXL-Turbo pipeline
    print("Loading SDXL-Turbo pipeline...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    print("Pipeline loaded.")

    # Move to GPU and apply optimizations
    print("Moving pipeline to GPU and applying optimizations...")
    pipe = pipe.to("cuda")
    pipe.enable_attention_slicing(slice_size="auto")
    print("Optimizations applied.")

    # Generate synthetic images
    print(f"Starting image generation for {num_images} images...")
    for i in tqdm(range(num_images), desc="Generating Images"):
        # --- NEW PROMPTING STRATEGY based on SYNAuG ---
        selected_modifier = random.choice(modifier_list)
        prompt = f"a photo of {selected_modifier} cat"  # Using the "modifier + class" structure
        # ---------------------------------------------

        try:
            # Generate image using SDXL-Turbo parameters
            image = pipe(
                prompt=prompt,
                guidance_scale=0.0,  # SDXL-Turbo specific
                num_inference_steps=1,  # SDXL-Turbo specific
                width=512,
                height=512,
            ).images[0]

            # Define output path
            output_path = os.path.join(
                output_dir, f"{i:05d}_cat_{selected_modifier.replace(' ', '_')}.png"
            )  # Added modifier to filename

            # Save the image
            image.save(output_path)

        except Exception as e:
            print(f"\nError generating image {i} with prompt: '{prompt}'")
            print(f"Error details: {e}")
            continue

    print(f"\nFinished generating {num_images} images.")


if __name__ == "__main__":
    # Clear CUDA cache before starting
    print("Clearing CUDA cache...")
    torch.cuda.empty_cache()
    gc.collect()
    print("Cache cleared.")

    # Run the image generation
    make_img(
        folder="./sdxl_turbo_synaug_style",  # Updated folder name
        num_images=7000,
        modifier_list=cat_modifiers,  # Pass the list of modifiers
    )

    print("Script finished.")
