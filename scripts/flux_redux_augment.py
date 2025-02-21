import random
import torch
from diffusers import FluxPriorReduxPipeline, FluxPipeline
from diffusers.utils import load_image  # for type conversions if needed
from PIL import Image

class FluxReduxAugment:
    """
    A custom torchvision-like transform that applies the Flux Redux augmentation
    with a specified probability. Expects a PIL image as input and returns a PIL image.
    """
    def __init__(
        self,
        guidance_scale: float = 2.5,
        num_inference_steps: int = 50,
        seed: int = 0,
        device: str = "cuda",
        probability: float = 1.0  # probability to apply the augmentation
    ):
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.seed = seed
        self.device = device
        self.probability = probability

        # Load the Flux Redux pipelines once
        self.flux_prior_redux = FluxPriorReduxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Redux-dev",
            torch_dtype=torch.bfloat16
        ).to(self.device)

        self.flux_pipeline = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            text_encoder=None,
            text_encoder_2=None,
            torch_dtype=torch.bfloat16
        ).to(self.device)

    def __call__(self, image: Image.Image) -> Image.Image:
        """
        Apply the Flux Redux augmentation to the input PIL image with the given probability.
        """
        # Decide whether to apply the augmentation.
        if random.random() > self.probability:
            # Skip augmentation; return the original image.
            return image

        # Pass the image through the prior redux pipeline.
        pipe_prior_output = self.flux_prior_redux(image)

        # Create a generator on the correct device and seed it.
        generator = torch.Generator(self.device).manual_seed(self.seed)

        # Generate an augmented image using the Flux pipeline.
        result = self.flux_pipeline(
            guidance_scale=self.guidance_scale,
            num_inference_steps=self.num_inference_steps,
            generator=generator,
            **pipe_prior_output,
        )
        # Return the first image from the result (as a PIL image).
        return result.images[0]
