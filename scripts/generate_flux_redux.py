import os
import glob
import random
import torch
from PIL import Image
from flux_redux_augment import FluxReduxAugment
from tqdm import tqdm

def main():
    input_dir = "./notebooks/cats"
    output_dir = "./notebooks/cats_redux"
    os.makedirs(output_dir, exist_ok=True)

    # Look for common image file extensions.
    image_paths = []
    for ext in ('*.jpg', '*.jpeg', '*.png'):
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))
    
    # Sort and limit to first 50 images.
    image_paths = sorted(image_paths)[:50]
    
    if not image_paths:
        print("No cat images found in", input_dir)
        return

    num_aug_per_image = 99  # 50 images * 99 augmentations = 4950 new examples

    # Initialize the augmentation instance with probability 1.0.
    flux_augment = FluxReduxAugment(
        guidance_scale=2.5,
        num_inference_steps=50,
        seed=0,           # initial seed; will be updated for each augmentation
        device="cuda",
        probability=1.0   # always apply the augmentation
    )

    total = 0
    for img_path in tqdm(image_paths):
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error opening image {img_path}: {e}")
            continue

        base_name = os.path.splitext(os.path.basename(img_path))[0]
        img.save(os.path.join(output_dir, f"{base_name}_orig.png"))
        for i in range(num_aug_per_image):
            # Update seed for variation (you can also use a counter or random seed)
            flux_augment.seed = random.randint(0, 10000)
            augmented_img = flux_augment(img)
            out_name = f"{base_name}_aug_{i}.png"
            out_path = os.path.join(output_dir, out_name)
            augmented_img.save(out_path)
            total += 1
            print(f"Saved {out_path}")
    print(f"Total augmented images saved: {total}")

if __name__ == "__main__":
    main()
