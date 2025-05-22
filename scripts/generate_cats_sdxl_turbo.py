import os
import random
import torch
from diffusers import StableDiffusionXLPipeline
from tqdm import tqdm
import gc
from typing import List, Optional  # Import List, Optional for type hinting

"""
This script generates cat images using the Stable Diffusion XL Turbo pipeline.
It employs a simplified "modifier + class" prompting strategy (inspired by SYNAuG paper)
to create diverse cat images. Output filenames are formatted to be compatible
with data.py's _add_extra_images method.
"""
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


def make_img(
    folder: str = "./tmp",
    num_images: int = 10000,
    modifier_list: Optional[List[str]] = None,
) -> None:
    """
    Generates cat images using the SDXL-Turbo pipeline with a simplified
    prompting style based on the SYNAuG paper (modifier + class).

    Args:
        folder (str): The directory where generated images will be saved.
        num_images (int): The total number of images to generate.
        modifier_list (Optional[List[str]]): A list of descriptive modifiers to be used
                                               in image prompts. If None or empty, an error is printed.
    """
    if modifier_list is None or len(modifier_list) == 0:
        print("Error: A list of modifiers must be provided.")
        return

    output_dir: str = folder
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving images to: {output_dir}")

    print("Loading SDXL-Turbo pipeline...")
    pipe: Optional[StableDiffusionXLPipeline] = None
    try:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/sdxl-turbo",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        )
        print("Pipeline loaded.")
    except Exception as e:
        print(f"Error loading pipeline: {e}")
        print(
            "Please ensure you have the necessary libraries installed, sufficient VRAM, and are logged in (`huggingface-cli login`)."
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return

    if torch.cuda.is_available():
        print("Moving pipeline to GPU and applying optimizations...")
        try:
            pipe = pipe.to("cuda")
            print("Optimizations applied.")
        except Exception as e:
            print(f"Error moving pipeline to GPU: {e}. Trying CPU.")
            try:
                pipe = pipe.to("cpu")
                print("Running on CPU.")
            except Exception as cpu_e:
                print(f"Error moving pipeline to CPU: {cpu_e}. Exiting.")
                del pipe
                gc.collect()
                return
    else:
        print("Warning: CUDA not available. Running on CPU will be very slow.")
        try:
            pipe = pipe.to("cpu")
        except Exception as e:
            print(f"Error moving pipeline to CPU: {e}. Exiting.")
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
                    guidance_scale=0.0,
                    num_inference_steps=1,
                    width=512,
                    height=512,
                )
                if image_result and image_result.images:
                    image_to_save = image_result.images[0]
                else:
                    tqdm.write(
                        f"\nWarning: Image generation failed for prompt: '{prompt}'. No image returned."
                    )
                    continue

            output_path: str = os.path.join(
                output_dir, f"cat_{i:05d}_{selected_modifier.replace(' ', '_')}.png"
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
    if pipe is not None:
        del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    print("Clearing CUDA cache...")
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.synchronize()
    print("Cache cleared.")

    make_img(
        folder="./sdxl_turbo_synaug_style",
        num_images=1,
        modifier_list=cat_modifiers,
    )

    print("Script finished.")
