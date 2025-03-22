import os
import torch
from diffusers import FluxPipeline
import gc

# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.cuda.empty_cache()
gc.collect()

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    torch_dtype=torch.float16,
    use_safetensors=True
)

pipe.enable_sequential_cpu_offload()
pipe.enable_attention_slicing(slice_size=4)
pipe.enable_vae_tiling()

promtps = [
    "A close-up of a cat, looking at the camera, detailed fur.",
    # "A close-up of a cat, looking left, detailed fur.",
    # "A full shot of a cat, detailed fur.",
    # "A front view of a cat, looking at the camera, detailed fur.",
    # "A side view of a cat, looking at the camera, detailed fur.",
    # "A dog in a park",
    # "A bird in a tree",
    # "A deer in a meadow",
    # "A frog near a pond",
    # "A horse in a field",
    # "A ship in the ocean",
    # "A truck on a highway",
    # "An airplane on a runway",
    # "An automobile on a city street"
]
prompt = (
            "A realistic photo taken from behind of a cat, looking straight ahead."
            "The cat has realistic fur textures, intricate details, and sharp features, "
            "8k, cinematic, photorealistic"
        )
negative_prompt = "blurry, low quality, artifacts, noise, distorted, deformed, text, watermark, logo, signature"


for promptss in promtps:
    image = pipe(
        prompt,
        guidance_scale=1.0,
        num_inference_steps=4,
        max_sequence_length=128,
        width=128,
        height=128,
        # negative_prompt=negative_prompt
    ).images[0]
    image.save("flux-schnell.png")