# !pip install diffusers transformers torch tqdm sentencepiece accelerate ipywidgets
# !huggingface-cli login
import os
import random

import torch
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
from tqdm import tqdm


def make_img(
    folder: str = "./tmp",
    num_inference_steps=40,
    guidance_scale=0.0,
    num_images_per_class=5000,
) -> None:
    output_dir = folder
    os.makedirs(output_dir, exist_ok=True)

    scheduler = EulerDiscreteScheduler.from_pretrained(
        "stabilityai/stable-diffusion-2", subfolder="scheduler"
    )
    pipe = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2", scheduler=scheduler, torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda")

    for cifar_class in cifar_classes:
        for i in tqdm(
            range(num_images_per_class), desc=f"Generating {cifar_class} Images"
        ):
            setting = random.choice(settings[cifar_class])
            style = random.choice(image_styles)
            lighting = random.choice(lighting_conditions)
            background = random.choice(backgrounds)
            angle = random.choice(camera_angles)
            action = random.choice(actions[cifar_class])

            prompt = (
                f"{style}, {angle} of a {cifar_class} {action}, in {setting}, {lighting}. "
                f"The background features {background}. Ultra-detailed, high resolution, photorealistic, 8K."
            )

            try:
                image = pipe(
                    prompt=prompt,
                    width=512,
                    height=512,
                ).images[0]

                output_path = os.path.join(output_dir, f"{cifar_class}_{i:05d}.png")
                image.save(output_path)
            except Exception:
                pass


if __name__ == "__main__":
    cifar_classes = [
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

    settings = {
        "airplane": [
            "on a runway",
            "flying in the sky",
            "inside an airport hangar",
            "during sunset",
        ],
        "automobile": [
            "on a city street",
            "racing on a track",
            "parked in a driveway",
            "covered in snow",
        ],
        "bird": [
            "on a tree branch",
            "flying over a lake",
            "perched on a fence",
            "in a birdhouse",
        ],
        "cat": ["on a sofa", "under a tree", "next to a window", "on a bookshelf"],
        "deer": [
            "in a dense forest",
            "crossing a meadow",
            "drinking from a stream",
            "standing on a hilltop",
        ],
        "dog": [
            "playing in a park",
            "sitting on a porch",
            "running on the beach",
            "inside a house",
        ],
        "frog": [
            "on a lily pad",
            "jumping near a pond",
            "resting on a rock",
            "inside a rainforest",
        ],
        "horse": [
            "galloping in a field",
            "standing near a barn",
            "racing on a track",
            "pulling a carriage",
        ],
        "ship": [
            "sailing in the ocean",
            "docked at a harbor",
            "facing a storm",
            "cruising in calm waters",
        ],
        "truck": [
            "on a highway",
            "parked at a gas station",
            "hauling cargo",
            "covered in mud",
        ],
    }

    actions = {
        "airplane": [
            "soaring through the clouds",
            "taking off",
            "landing",
            "performing aerobatics",
        ],
        "automobile": [
            "drifting around a curve",
            "stuck in traffic",
            "speeding down the road",
            "being washed",
        ],
        "bird": [
            "singing on a branch",
            "flying in the air",
            "eating seeds",
            "chasing an insect",
        ],
        "cat": [
            "sleeping on a couch",
            "chasing a toy",
            "watching birds outside",
            "stretching lazily",
        ],
        "deer": [
            "grazing in a meadow",
            "running through the woods",
            "standing alert",
            "jumping over a log",
        ],
        "dog": [
            "playing with a ball",
            "wagging its tail",
            "barking at something",
            "lying in the sun",
        ],
        "frog": [
            "jumping between rocks",
            "catching a fly",
            "resting on a leaf",
            "croaking near water",
        ],
        "horse": [
            "galloping across a field",
            "pulling a cart",
            "standing majestically",
            "drinking water",
        ],
        "ship": [
            "sailing under the stars",
            "facing huge waves",
            "anchored at port",
            "carrying cargo",
        ],
        "truck": [
            "driving through a muddy road",
            "delivering goods",
            "stuck in traffic",
            "covered in dust",
        ],
    }

    image_styles = [
        "cinematic shot",
        "realistic photograph",
        "hyper-detailed rendering",
        "artistic composition",
    ]
    lighting_conditions = [
        "soft golden light",
        "harsh midday sun",
        "moody blue hour",
        "dramatic backlighting",
    ]
    backgrounds = [
        "a bustling city",
        "a quiet countryside",
        "a futuristic setting",
        "a historic village",
    ]
    camera_angles = [
        "top-down view",
        "low-angle close-up",
        "side perspective",
        "distant wide shot",
    ]

    make_img("./SD20", 40, num_images_per_class=1)
