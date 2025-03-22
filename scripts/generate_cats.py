# !pip install diffusers transformers torch tqdm sentencepiece accelerate ipywidgets
# !huggingface-cli login
import os
import random

import torch
from diffusers import FluxPipeline
from tqdm import tqdm
import gc
# import lightning as L

# L.seed_everything(42)


def make_img(folder: str = "./tmp", num_images=10000) -> None:
    # Directory to save generated images
    output_dir = folder
    os.makedirs(output_dir, exist_ok=True)
    # Generate synthetic images
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        torch_dtype=torch.float16,
        use_safetensors=True
    )

    pipe.enable_sequential_cpu_offload()
    pipe.enable_attention_slicing(slice_size=4)
    pipe.enable_vae_tiling()
    for i in tqdm(range(num_images), desc="Generating Images"):
        # Randomly select components for the prompt
        # breed = random.choice(cat_breeds)
        # preposition = random.choice(prepositions)
        # furniture = random.choice(furniture_or_outdoor[preposition])
        angle = random.choice(camera_angles)
        gaze = random.choice(gaze_directions)

        # Construct the prompt
        prompt = (
            f"A realistic photo of a cat {gaze}, {angle}. "
            "The cat has realistic fur textures, intricate details, and sharp features, "
            "with soft lighting and a clear focus. 8k, cinematic, photorealistic"
        )
        image = pipe(
                prompt,
                guidance_scale=3.0,
                num_inference_steps=4,
                max_sequence_length=128,
                width=128,
                height=128,
                # negative_prompt=negative_prompt
        ).images[0]
        output_path = os.path.join(
            output_dir,
            f"{i:05d}_cat_{angle}_{gaze}.png",
        )
        image.save(output_path)


if __name__ == "__main__":
    camera_angles = [
        "taken from above",
        "taken from below",
        "side-view",
        "front-facing",
        "taken from behind",
    ]

    # List of gaze directions
    gaze_directions = [
        # "looking straight ahead",
        # "looking up",
        # "looking down",
        # "looking to the left",
        # "looking to the right",
        # "looking up and to the left",
        # "looking up and to the right",
        # "looking down and to the left",
        # "looking down and to the right",
        # "eyes closed",
        # "looking over its shoulder",
        "walking",
        "running",
        "playing",
        "sleeping",
        "sitting",
        "standing",
        "jumping",
        "crouching",
        "pouncing",
        "stretching",
        "grooming",
        "eating",
    ]

    torch.cuda.empty_cache()
    gc.collect()
    make_img("./flux_tests", 40)
