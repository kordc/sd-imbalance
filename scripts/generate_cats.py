# !pip install diffusers transformers torch tqdm sentencepiece accelerate ipywidgets
# !huggingface-cli login
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import load_image
from diffusers import DiffusionPipeline, StableDiffusion3Pipeline
import torch
import os
from tqdm import tqdm
import random
import itertools

def make_img(folder: str = "./tmp", num_inference_steps=4, guidance_scale=0.0):
    # Directory to save generated images
    output_dir = folder
    os.makedirs(output_dir, exist_ok=True)
    # Generate synthetic images
    num_images = 6000
    for i in tqdm(range(num_images), desc="Generating Images"):
        # Randomly select components for the prompt
        breed = random.choice(cat_breeds)
        preposition = random.choice(prepositions)
        furniture = random.choice(furniture_or_outdoor[preposition])
        angle = random.choice(camera_angles)
        gaze = random.choice(gaze_directions)
        if i == 0:
            angle = "a photo taken from behind"
        
        # Construct the prompt
        prompt = (
            f"{angle} of a {breed} cat {preposition} the {furniture}, {gaze}. "
            "The cat has realistic fur textures, intricate details, and sharp features, "
            "with soft lighting and a clear focus. The image has a shallow depth of field, "
            "emphasizing the cat in fine detail. 8k, cinematic, photorealistic"
        )
        print(prompt)
        try:
            # Generate image
            result = pipe(prompt, num_inference_steps=num_inference_steps,
                          guidance_scale=guidance_scale, width=512, height=512)
            image = result.images[0]  # Get the first image from the list
            
            # Save the generated image
            output_path = os.path.join(output_dir, f"{i:05d}_cat_{breed}_{preposition}_{furniture}_{angle}_{gaze}.png")
            image.save(output_path)
            
        except Exception as e:
            print(f"Failed to generate image {i}: {e}")
    
    print(f"Generated {num_images} images and saved them in {output_dir}.")

if __name__ == "__main__":
    # Expanded list of cat breeds
    cat_breeds = [
        "Abyssinian", "Aegean", "American Bobtail", "American Curl", "American Ringtail", 
        "American Shorthair", "American Wirehair", "Aphrodite Giant", "Arabian Mau", 
        "Asian", "Asian Semi-longhair", "Australian Mist", "Balinese", "Bambino", "Bengal", 
        "Birman", "Bombay", "Brazilian Shorthair", "British Longhair", "British Shorthair", 
        "Burmese", "Burmilla", "California Spangled", "Chantilly-Tiffany", "Chartreux", 
        "Chausie", "Colorpoint Shorthair", "Cornish Rex", "Cymric", "Cyprus", "Devon Rex", 
        "Donskoy", "Dragon Li", "Dwelf", "Egyptian Mau", "European Shorthair", 
        "Exotic Shorthair", "Foldex", "German Rex", "Havana Brown", "Highlander", 
        "Himalayan", "Japanese Bobtail", "Javanese", "Kanaani", "Karelian Bobtail", 
        "Khao Manee", "Kinkalow", "Korat", "Korean Bobtail", "Korn Ja", "Kurilian Bobtail", 
        "Lambkin", "LaPerm", "Lykoi", "Maine Coon", "Manx", "Mekong Bobtail", "Minskin", 
        "Minuet", "Munchkin", "Nebelung", "Neva Masquerade", "Norwegian Forest cat", 
        "Ocicat", "Ojos Azules", "Oriental Bicolor", "Oriental Longhair", 
        "Oriental Shorthair", "Persian", "Peterbald", 
        "Pixie-bob", "Ragamuffin", "Ragdoll", "Raas", "Russian Blue", 
        "Russian White, Russian Black and Russian Tabby", "Sam Sawet", "Savannah", 
        "Scottish Fold", "Selkirk Rex", "Serengeti", "Siamese", "Siberian", 
        "Singapura", "Snowshoe", "Sokoke", "Somali", "Sphynx", "Suphalak", "Thai", 
        "Thai Lilac", "Tonkinese", "Toybob", "Toyger", "Turkish Angora", "Turkish Van", 
        "Turkish Vankedisi", "Ukrainian Levkoy", "York Chocolate"
    ]
    
    print(f"Cat breeds: {len(cat_breeds)}")
    
    # Expanded list of prepositions
    prepositions = ["on", "under", "next to", "beside", "in front of", "behind"]
    
    # Furniture types that fit with the prepositions
    furniture_or_outdoor = {
        "on": [
            # Indoor furniture
            "sofa", "armchair", "dining table", "coffee table", "bed", "bookshelf", "cabinet", 
            "cupboard", "chair", "desk", "ottoman", "recliner", "side table", "nightstand", 
            "TV stand", "TV cabinet", "end table", "love seat", "couch", "lounge chair", "bean bag", 
            "bar stool", "console table", "chest of drawers", "vanity", "shelf",
            # Outdoor/natural elements
            "rock", "tree stump", "fence", "bench", "picnic table", "grass", "car hood", "roof",
            "garden wall", "park bench", "log", "fallen tree", "sand dune", "boulder"
        ],
        "under": [
            # Indoor furniture
            "sofa", "bed", "coffee table", "side table", "desk", "recliner", "nightstand", 
            "chest of drawers", "love seat", "couch", "bookshelf",
            # Outdoor/natural elements
            "tree", "bush", "bridge", "rock", "table", "park bench", "porch", "car", 
            "wooden deck", "fallen tree", "shade", "overhang"
        ],
        "next to": [
            # Indoor furniture
            "sofa", "armchair", "dining table", "coffee table", "bookshelf", "cabinet", "cupboard", 
            "chair", "ottoman", "recliner", "side table", "nightstand", "TV stand", "end table", 
            "love seat", "lounge chair", "console table", "vanity", "window sill",
            # Outdoor/natural elements
            "tree", "bush", "fence", "bench", "log", "rock", "flower bed", "pathway", "pond", 
            "stream", "hill", "park bench", "fire hydrant", "fallen tree"
        ],
        "beside": [
            # Indoor furniture
            "sofa", "armchair", "dining table", "coffee table", "bed", "bookshelf", "chair", 
            "ottoman", "recliner", "side table", "nightstand", "love seat", "couch", "lounge chair", 
            "vanity",
            # Outdoor/natural elements
            "tree", "rock", "bush", "fence", "bench", "stream", "pathway", "log", "garden wall", 
            "pond", "flower bed", "hill", "fallen tree"
        ],
        "in front of": [
            # Indoor furniture
            "sofa", "armchair", "coffee table", "cabinet", "cupboard", "TV stand", "console table", 
            "chest of drawers", "bed", "recliner", "side table", "dining chairs", "love seat",
            # Outdoor/natural elements
            "tree", "rock", "fence", "bush", "bench", "stream", "pathway", "building", 
            "garden wall", "pond", "waterfall", "hill", "statue"
        ],
        "behind": [
            # Indoor furniture
            "sofa", "armchair", "bookshelf", "recliner", "cabinet", "TV stand", "chest of drawers", 
            "love seat",
            # Outdoor/natural elements
            "tree", "bush", "fence", "rock", "log", "bench", "building", "hill", "statue", 
            "garden wall", "shed"
        ],
    }
    camera_angles = [
        "a photo taken from above",
        "a photo taken from below",
        "a side-view photo",
        "a front-facing photo",
        "a photo taken from behind"
    ]
    
    # List of gaze directions
    gaze_directions = [
        "looking straight ahead",
        "looking up",
        "looking down",
        "looking to the left",
        "looking to the right",
        "looking up and to the left",
        "looking up and to the right",
        "looking down and to the left",
        "looking down and to the right",
        "eyes closed",
        "looking over its shoulder",
    ]
    print(f"Prepositions: {len(furniture_or_outdoor)}")
    print(f"Objects: {len(set(list(itertools.chain.from_iterable(furniture_or_outdoor.values()))))}")
    
    from diffusers import FluxPipeline
    pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-large-turbo", torch_dtype=torch.bfloat16)
    pipe = pipe.to("cuda")
    make_img("./generated_cats_stable-diffusion-3.5-large-turbo", 6)
