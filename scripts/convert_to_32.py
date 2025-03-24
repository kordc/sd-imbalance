import os
from PIL import Image

# Paths
input_dir = "flux_tests"
output_dir = "flux_tests_32_bicubic"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Process images
for filename in os.listdir(input_dir):
    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, filename)

    # Check if the file is an image
    if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff")):
        with Image.open(input_path) as img:
            # Resize image to 32x32 using bicubic interpolation
            img_resized = img.resize((32, 32), Image.BICUBIC)
            img_resized.save(output_path)

print("All images have been resized and saved to", output_dir)
