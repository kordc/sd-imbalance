# !pip install diffusers transformers torch tqdm accelerate ipywidgets Pillow
# !huggingface-cli login # You might need to run this in your terminal if you haven't before

import os
import random
import torch
from diffusers import StableDiffusionXLPipeline
from tqdm import tqdm
import gc
from PIL import Image # Import PIL

# --- CIFAR-10 Classes ---
cifar10_classes = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]

# --- NEW: Class-Specific Modifiers ---
# Define relevant modifiers for each class to avoid nonsensical combinations
class_specific_modifiers = {
    "airplane": [
        "flying", "in the sky", "on the runway", "parked", "jet", "propeller plane",
        "military airplane", "passenger airplane", "taking off", "landing", "realistic",
        "detailed", "photorealistic", "high resolution", "clear", "sharp focus",
        "cinematic", "side view", "from below", "distant view", "modern", "vintage"
    ],
    "automobile": [
        "driving on the road", "parked", "on a street", "sedan", "suv", "convertible",
        "vintage car", "modern car", "sports car", "red car", "blue car", "realistic",
        "detailed", "photorealistic", "high resolution", "clear", "sharp focus",
        "cinematic", "side view", "front view", "in a city", "on a highway"
    ],
    "bird": [
        "perched on a branch", "flying", "in a nest", "on the ground", "songbird",
        "bird of prey", "colorful bird", "small bird", "realistic", "detailed",
        "photorealistic", "high resolution", "clear", "sharp focus", "natural light",
        "in a forest", "close-up", "in profile", "wings spread"
    ],
    "cat": [
        "sitting", "sleeping", "playing", "looking at camera", "domestic cat",
        "tabby cat", "black cat", "white cat", "fluffy", "sleek", "in a house",
        "on a sofa", "realistic", "detailed", "photorealistic", "high resolution",
        "clear", "sharp focus", "natural light", "soft light", "close-up", "full body", "adorable"
    ],
    "deer": [
        "standing in a forest", "grazing in a field", "running", "buck", "doe",
        "fawn", "with antlers", "alert", "realistic", "detailed", "photorealistic",
        "high resolution", "clear", "sharp focus", "natural light", "in the wild",
        "side view", "majestic"
    ],
    "dog": [
        "sitting", "running", "playing fetch", "panting", "domestic dog",
        "golden retriever", "german shepherd", "small dog", "large dog", "loyal",
        "realistic", "detailed", "photorealistic", "high resolution", "clear",
        "sharp focus", "natural light", "in a park", "on a leash", "close-up", "full body"
    ],
    "frog": [
        "sitting on a lily pad", "hopping", "in a pond", "green frog", "tree frog",
        "camouflaged", "close-up", "realistic", "detailed", "photorealistic",
        "high resolution", "clear", "sharp focus", "natural light", "wet", "in a swamp"
    ],
    "horse": [
        "running in a field", "standing", "grazing", "brown horse", "white horse",
        "black horse", "with a saddle", "wild horse", "domestic horse", "realistic",
        "detailed", "photorealistic", "high resolution", "clear", "sharp focus",
        "natural light", "cinematic", "side view", "majestic", "on a farm"
    ],
    "ship": [
        "sailing on the water", "docked in a harbor", "large ship", "small boat",
        "sailboat", "cruise ship", "container ship", "battleship", "realistic",
        "detailed", "photorealistic", "high resolution", "clear", "sharp focus",
        "on the ocean", "on a river", "side view", "distant view", "vintage", "modern"
    ],
    "truck": [
        "driving on the road", "parked", "pickup truck", "semi-truck", "dump truck",
        "delivery truck", "red truck", "blue truck", "realistic", "detailed",
        "photorealistic", "high resolution", "clear", "sharp focus", "cinematic",
        "side view", "on a highway", "at a construction site", "modern", "old"
    ]
}


def make_cifar10_style_images(
    base_folder: str = "./cifar10_synaug_sdxl_class_specific", # Updated folder name
    num_images_per_class: int = 500,
    classes: list = None,
    modifier_map: dict = None # Takes the modifier dictionary now
) -> None:
    """
    Generates images for specified classes using SDXL-Turbo pipeline
    with CLASS-SPECIFIC SYNAuG prompting ("a photo of {modifier} {class}").
    Saves images into class-specific subfolders.

    Args:
        base_folder (str): The root directory to save generated class folders.
        num_images_per_class (int): How many images to generate for each class.
        classes (list): A list of class names (strings).
        modifier_map (dict): A dictionary mapping class names to lists of modifier strings.
    """
    if classes is None or len(classes) == 0:
        print("Error: A list of classes must be provided.")
        return
    if modifier_map is None or len(modifier_map) == 0:
        print("Error: A class-to-modifier dictionary must be provided.")
        return

    # Base directory for all generated images
    os.makedirs(base_folder, exist_ok=True)
    print(f"Base output directory: {base_folder}")

    # Load SDXL-Turbo pipeline
    print("Loading SDXL-Turbo pipeline...")
    try:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/sdxl-turbo",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        )
        print("Pipeline loaded.")
    except Exception as e:
        print(f"Error loading pipeline: {e}")
        print("Please ensure you have the necessary libraries installed, sufficient VRAM, and are logged in (`huggingface-cli login`).")
        return

    # Move to GPU and apply optimizations
    if torch.cuda.is_available():
        print("Moving pipeline to GPU and applying optimizations...")
        pipe = pipe.to("cuda")
        pipe.enable_attention_slicing(slice_size='auto')
        print("Optimizations applied.")
    else:
        print("Warning: CUDA not available. Running on CPU will be very slow.")
        pipe = pipe.to("cpu")


    # Generate synthetic images for each class
    for class_name in classes:
        class_folder = os.path.join(base_folder, class_name)
        os.makedirs(class_folder, exist_ok=True)
        print(f"\n--- Generating images for class: '{class_name}' ---")
        print(f"Saving to: {class_folder}")

        # --- Get the modifiers SPECIFIC to this class ---
        current_modifiers = modifier_map.get(class_name)
        if not current_modifiers:
            print(f"Warning: No specific modifiers found for class '{class_name}'. Skipping this class.")
            continue # Skip to the next class if no modifiers are defined

        generated_count = 0
        pbar = tqdm(total=num_images_per_class, desc=f"Generating {class_name}")
        while generated_count < num_images_per_class:
            # --- SYNAuG PROMPTING with CLASS-SPECIFIC modifier ---
            selected_modifier = random.choice(current_modifiers)
            prompt = f"a photo of {selected_modifier} {class_name}"
            # ------------------------------------------------------

            try:
                # Generate image using SDXL-Turbo parameters
                image = pipe(
                    prompt=prompt,
                    guidance_scale=0.0,
                    num_inference_steps=1,
                    width=512,
                    height=512,
                ).images[0]

                # Define output path
                safe_modifier = selected_modifier.replace(' ','_').replace('/','_')
                output_path = os.path.join(class_folder, f"{class_name}_{generated_count:04d}_{safe_modifier}.png")

                # Save the image
                image.save(output_path)
                generated_count += 1
                pbar.update(1)

            except torch.cuda.OutOfMemoryError:
                 print(f"\nCUDA OutOfMemoryError occurred for class '{class_name}', prompt: '{prompt}'. Skipping image, clearing cache.")
                 torch.cuda.empty_cache()
                 gc.collect()
                 # Consider adding a small delay if OOM happens frequently
                 # import time
                 # time.sleep(1)
                 continue # Skip to the next iteration

            except Exception as e:
                print(f"\nError generating image {generated_count} for class '{class_name}' with prompt: '{prompt}'")
                print(f"Error details: {e}")
                continue # Skip to the next iteration

        pbar.close()
        print(f"Finished generating {generated_count} images for class '{class_name}'.")
        # Optional: Clear cache between classes if memory is very tight
        # if torch.cuda.is_available():
        #     torch.cuda.empty_cache()
        #     gc.collect()

    print(f"\nFinished generating images for all specified classes. Saved in {base_folder}")


if __name__ == "__main__":
    # Clear CUDA cache before starting
    if torch.cuda.is_available():
        print("Clearing CUDA cache...")
        torch.cuda.empty_cache()
        gc.collect()
        print("Cache cleared.")

    # Run the image generation for all CIFAR-10 classes with specific modifiers
    make_cifar10_style_images(
        base_folder="./cifar10_synaug_sdxl_class_specific", # Use a distinct folder name
        num_images_per_class=5000,                           # Adjust as needed
        classes=cifar10_classes,                            # Pass the CIFAR-10 class list
        modifier_map=class_specific_modifiers               # Pass the dictionary mapping classes to modifiers
    )

    print("\nScript finished.")