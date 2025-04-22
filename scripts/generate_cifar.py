# !pip install diffusers transformers torch tqdm accelerate ipywidgets Pillow
# !huggingface-cli login # You might need to run this in your terminal if you haven't before

import os
import random
import torch
from diffusers import StableDiffusionXLPipeline
from tqdm import tqdm
import gc

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
    "truck",
]

# --- Class-Specific Concepts (Inspired by PDF's Cat Example) ---
# We adapt the PDF's <BREED> <PREPOSITION> <FURNITURE> idea
# into more general <DESCRIPTOR/TYPE> <CONTEXT/ACTION> concepts
# relevant to each CIFAR-10 class.

class_specific_concepts = {
    "airplane": {
        "descriptor_type": [
            "jet",
            "propeller plane",
            "biplane",
            "passenger airplane",
            "cargo plane",
            "military aircraft",
            "small airplane",
            "large airplane",
            "vintage airplane",
            "modern airplane",
        ],
        "context_action": [
            "flying in the sky",
            "on the runway",
            "taking off",
            "landing",
            "parked at an airport",
            "above the clouds",
            "during sunset",
            "in clear blue sky",
            "performing aerobatics",
            "in formation",
        ],
    },
    "automobile": {
        "descriptor_type": [
            "sedan",
            "suv",
            "convertible",
            "sports car",
            "vintage car",
            "modern car",
            "pickup truck",
            "electric car",
            "red car",
            "blue car",
            "family car",
        ],
        "context_action": [
            "driving on a highway",
            "parked on a city street",
            "on a scenic road",
            "in a garage",
            "at a gas station",
            "in traffic",
            "during rain",
            "at night",
            "covered in snow",
            "off-road",
        ],
    },
    "bird": {
        "descriptor_type": [
            "songbird",
            "bird of prey",
            "eagle",
            "sparrow",
            "robin",
            "blue jay",
            "parrot",
            "hummingbird",
            "owl",
            "colorful bird",
            "small bird",
        ],
        "context_action": [
            "perched on a branch",
            "flying in the sky",
            "in a nest with eggs",
            "on the ground looking for food",
            "drinking water",
            "preening its feathers",
            "in a bird feeder",
            "in a forest",
            "near a lake",
            "silhouette against the sun",
        ],
    },
    # --- PDF Inspired Cat Structure ---
    "cat": {
        "descriptor_type": [
            "tabby cat",
            "black cat",
            "white cat",
            "ginger cat",
            "siamese cat",
            "persian cat",
            "fluffy cat",
            "sleek cat",
            "kitten",
            "old cat",
            "domestic cat",
        ],
        "context_action": [
            "sitting on a sofa",
            "sleeping in a sunbeam",
            "playing with a toy mouse",
            "looking out a window",
            "climbing a tree",
            "drinking milk from a bowl",
            "yawning",
            "curled up in a box",
            "walking on a fence",
            "hiding under a bed",
        ],
    },
    "deer": {
        "descriptor_type": [
            "buck with large antlers",
            "doe",
            "fawn",
            "white-tailed deer",
            "mule deer",
            "majestic deer",
            "group of deer",
            "young deer",
            "alert deer",
        ],
        "context_action": [
            "standing in a misty forest",
            "grazing in a meadow",
            "running across a field",
            "drinking from a stream",
            "peeking through bushes",
            "in the snow",
            "at sunrise",
            "camouflaged in the woods",
            "crossing a road",
            "startled",
        ],
    },
    "dog": {
        "descriptor_type": [
            "golden retriever",
            "german shepherd",
            "labrador",
            "poodle",
            "beagle",
            "bulldog",
            "small dog",
            "large dog",
            "puppy",
            "loyal dog",
            "fluffy dog",
        ],
        "context_action": [
            "playing fetch in a park",
            "running on a beach",
            "sitting by a fireplace",
            "panting after a walk",
            "sleeping on a rug",
            "waiting by the door",
            "riding in a car",
            "getting a bath",
            "wagging its tail",
            "on a leash",
        ],
    },
    "frog": {
        "descriptor_type": [
            "green frog",
            "tree frog",
            "bullfrog",
            "poison dart frog",
            "camouflaged frog",
            "small frog",
            "large frog",
            "spotted frog",
        ],
        "context_action": [
            "sitting on a lily pad",
            "hopping through grass",
            "in a murky pond",
            "climbing a plant",
            "catching an insect",
            "inflating its throat",
            "submerged in water",
            "on a wet rock",
            "in a rainforest",
            "at night",
        ],
    },
    "horse": {
        "descriptor_type": [
            "brown horse",
            "white horse",
            "black horse",
            "stallion",
            "mare",
            "foal",
            "wild horse",
            "racehorse",
            "draft horse",
            "pony",
            "majestic horse",
        ],
        "context_action": [
            "running free in a field",
            "grazing peacefully",
            "standing in a stable",
            "being ridden",
            "pulling a cart",
            "drinking from a river",
            "galloping on a track",
            "jumping over a fence",
            "in the mountains",
            "silhouette at sunset",
        ],
    },
    "ship": {
        "descriptor_type": [
            "sailboat",
            "cruise ship",
            "container ship",
            "battleship",
            "fishing boat",
            "yacht",
            "old wooden ship",
            "modern ship",
            "ferry",
            "tugboat",
            "large ship",
        ],
        "context_action": [
            "sailing on the open ocean",
            "docked in a busy harbor",
            "navigating a river",
            "in stormy seas",
            "anchored near an island",
            "during sunset",
            "passing under a bridge",
            "loaded with cargo",
            "reflecting in the water",
            "in the fog",
        ],
    },
    "truck": {
        "descriptor_type": [
            "pickup truck",
            "semi-truck",
            "dump truck",
            "delivery truck",
            "fire truck",
            "monster truck",
            "old rusty truck",
            "modern truck",
            "red truck",
            "blue truck",
            "heavy-duty truck",
        ],
        "context_action": [
            "driving on a highway",
            "parked at a construction site",
            "loading or unloading goods",
            "on a dirt road",
            "at a truck stop",
            "stuck in mud",
            "carrying logs",
            "hauling a trailer",
            "in city traffic",
            "at night",
        ],
    },
}

# --- General Quality Modifiers ---
quality_modifiers = [
    "photorealistic",
    "high resolution",
    "detailed",
    "clear photo",
    "sharp focus",
    "cinematic lighting",
    "natural light",
    "realistic",
    "professional photography",
    "4k",
    "masterpiece",
    "best quality",
]


def make_cifar10_style_images(
    base_folder: str = "./cifar10_synaug_sdxl_pdf_style",  # Updated folder name
    num_images_per_class: int = 500,
    classes: list = None,
    concept_map: dict = None,  # Takes the concept dictionary
) -> None:
    """
    Generates images for specified classes using SDXL-Turbo pipeline
    with prompting inspired by the PDF's structured approach
    (e.g., "photorealistic photo of a {descriptor_type} {class_name} {context_action}").
    Saves images into class-specific subfolders.

    Args:
        base_folder (str): The root directory to save generated class folders.
        num_images_per_class (int): How many images to generate for each class.
        classes (list): A list of class names (strings).
        concept_map (dict): A dictionary mapping class names to sub-dictionaries
                            containing lists for 'descriptor_type' and 'context_action'.
    """
    if classes is None or len(classes) == 0:
        print("Error: A list of classes must be provided.")
        return
    if concept_map is None or len(concept_map) == 0:
        print("Error: A class-to-concept dictionary must be provided.")
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
            use_safetensors=True,
        )
        print("Pipeline loaded.")
    except Exception as e:
        print(f"Error loading pipeline: {e}")
        print(
            "Please ensure you have the necessary libraries installed, sufficient VRAM, and are logged in (`huggingface-cli login`)."
        )
        return

    # Move to GPU and apply optimizations
    if torch.cuda.is_available():
        print("Moving pipeline to GPU and applying optimizations...")
        pipe = pipe.to("cuda")
        # Attention slicing might not be needed/beneficial for SDXL-Turbo's 1-step inference
        # but can leave it if desired or test without it.
        # pipe.enable_attention_slicing(slice_size='auto')
        print("Optimizations applied (moved to CUDA).")
    else:
        print("Warning: CUDA not available. Running on CPU will be very slow.")
        pipe = pipe.to("cpu")

    # Generate synthetic images for each class
    for class_name in classes:
        class_folder = os.path.join(base_folder, class_name)
        os.makedirs(class_folder, exist_ok=True)
        print(f"\n--- Generating images for class: '{class_name}' ---")
        print(f"Saving to: {class_folder}")

        # --- Get the concepts SPECIFIC to this class ---
        current_concepts = concept_map.get(class_name)
        if (
            not current_concepts
            or "descriptor_type" not in current_concepts
            or "context_action" not in current_concepts
        ):
            print(
                f"Warning: Incomplete or missing concepts for class '{class_name}'. Skipping this class."
            )
            continue  # Skip to the next class

        descriptors = current_concepts["descriptor_type"]
        contexts = current_concepts["context_action"]

        if not descriptors or not contexts:
            print(
                f"Warning: Empty descriptor or context list for class '{class_name}'. Skipping this class."
            )
            continue

        generated_count = 0
        pbar = tqdm(total=num_images_per_class, desc=f"Generating {class_name}")
        while generated_count < num_images_per_class:
            # --- PDF-Inspired Prompting ---
            selected_descriptor = random.choice(descriptors)
            selected_context = random.choice(contexts)
            selected_quality = random.choice(
                quality_modifiers
            )  # Add random quality term

            # Construct the prompt - placing quality modifier first often works well
            prompt = f"{selected_quality} photo of a {selected_descriptor} {class_name} {selected_context}"
            # --- End PDF-Inspired Prompting ---

            try:
                # Generate image using SDXL-Turbo parameters
                image = pipe(
                    prompt=prompt,
                    guidance_scale=0.0,  # Crucial for SDXL-Turbo
                    num_inference_steps=1,  # Crucial for SDXL-Turbo
                    width=512,  # Standard SDXL size, can adjust if needed
                    height=512,
                ).images[0]

                # Define output path - make filename safe
                safe_descriptor = selected_descriptor.replace(" ", "_").replace(
                    "/", "_"
                )
                safe_context = selected_context.replace(" ", "_").replace("/", "_")
                output_filename = f"{class_name}_{generated_count:04d}_{safe_descriptor}_{safe_context}.png"
                # Truncate filename if it gets too long (OS limits)
                max_len = 200  # Adjust if needed
                if len(output_filename) > max_len:
                    output_filename = f"{class_name}_{generated_count:04d}_{safe_descriptor[:30]}_{safe_context[:30]}.png"

                output_path = os.path.join(class_folder, output_filename)

                # Save the image
                image.save(output_path)
                generated_count += 1
                pbar.update(1)

            except torch.cuda.OutOfMemoryError:
                print(
                    f"\nCUDA OutOfMemoryError occurred for class '{class_name}', prompt: '{prompt}'. Skipping image, clearing cache."
                )
                # Clear cache immediately upon OOM
                del image  # Try to explicitly delete the potential large object
                torch.cuda.empty_cache()
                gc.collect()
                # Consider adding a small delay if OOM happens frequently
                # import time
                # time.sleep(1)
                continue  # Skip to the next iteration

            except Exception as e:
                print(
                    f"\nError generating image {generated_count} for class '{class_name}' with prompt: '{prompt}'"
                )
                print(f"Error details: {e}")
                # Optional: Clear cache even on other errors if memory seems correlated
                # if torch.cuda.is_available():
                #     torch.cuda.empty_cache()
                #     gc.collect()
                continue  # Skip to the next iteration
            finally:
                # Clean up reference to image object to potentially help GC
                if "image" in locals():
                    del image

        pbar.close()
        print(f"Finished generating {generated_count} images for class '{class_name}'.")
        # Optional: Clear cache between classes if memory is very tight, even without OOM
        # if torch.cuda.is_available():
        #     print("Clearing CUDA cache between classes...")
        #     torch.cuda.empty_cache()
        #     gc.collect()

    print(
        f"\nFinished generating images for all specified classes. Saved in {base_folder}"
    )
    # Final cache clear
    if torch.cuda.is_available():
        print("Final CUDA cache clear...")
        del pipe  # Explicitly delete pipeline
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    # Clear CUDA cache before starting
    if torch.cuda.is_available():
        print("Initial CUDA cache clear...")
        torch.cuda.empty_cache()
        gc.collect()
        print("Cache cleared.")

    # Run the image generation for all CIFAR-10 classes with specific concepts
    make_cifar10_style_images(
        base_folder="./cifar10_synaug_sdxl_pdf_style_v2",  # Use a distinct folder name
        num_images_per_class=5000,  # Generate 5000 images per class
        classes=cifar10_classes,  # Pass the CIFAR-10 class list
        concept_map=class_specific_concepts,  # Pass the dictionary mapping classes to concepts
    )

    print("\nScript finished.")
