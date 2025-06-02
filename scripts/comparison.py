import torch
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    StableDiffusion3Pipeline,
    FluxPipeline,
    AutoPipelineForText2Image,  # Można też użyć tego dla uproszczenia
    StableDiffusion3Pipeline,
)
from PIL import Image
import os
import gc

# --- Konfiguracja ---
PROMPT = "high resolution, sharp focus photo of a fluffy ginger cat curled up asleep on a sunlit windowsill"
SEEDS = [42, 123, 456, 789, 1011]  # 5 różnych seedów
NUM_IMAGES_PER_MODEL = len(SEEDS)
OUTPUT_DIR = "model_comparison_images"
HF_TOKEN = None  # Wstaw swój token Hugging Face jeśli jest potrzebny (np. dla SD3)
# lub zaloguj się przez `huggingface-cli login`

# Słownik mapujący nazwy modeli na ich identyfikatory i typy pipeline
# oraz sugerowane parametry generacji
MODEL_CONFIG = {
    "Stable Diffusion XL": {
        "id": "stabilityai/stable-diffusion-xl-base-1.0",
        "pipeline_class": StableDiffusionXLPipeline,
        "params": {
            "num_inference_steps": 30,
            "guidance_scale": 7.0,
        },  # SDXL często działa dobrze z mniejszą liczbą kroków niż SD1.5
        "torch_dtype": torch.float16,
        "variant": "fp16",
    },
    "Stable Diffusion XL Turbo": {
        "id": "stabilityai/sdxl-turbo",
        "pipeline_class": StableDiffusionXLPipeline,
        "params": {
            "num_inference_steps": 1,
            "guidance_scale": 0.0,
        },  # Specyficzne dla SDXL-Turbo
        "torch_dtype": torch.float16,
        "variant": "fp16",
    },
    "Stable Diffusion 2.1": {  # Zastępuje niejasne "SD 3.5 Large Turbo"
        "id": "stabilityai/stable-diffusion-3.5-large-turbo",  # lub stabilityai/stable-diffusion-2-1 dla pełnej precyzji
        "pipeline_class": StableDiffusion3Pipeline,
        "params": {"num_inference_steps": 4, "guidance_scale": 0},
        "torch_dtype": torch.bfloat16,
        "variant": "fp16",
    },
    "Stable Diffusion 3 Medium": {  # Dla "SD 3.5 Medium"
        "id": "stabilityai/stable-diffusion-3-medium-diffusers",  # Model GATED - wymaga tokenu/loginu i akceptacji
        "pipeline_class": StableDiffusion3Pipeline,
        "params": {
            "num_inference_steps": 28,
            "guidance_scale": 7.0,
        },  # Zalecane dla SD3
        "torch_dtype": torch.float16,  # SD3 Medium wspiera bfloat16 i float16
    },
    "FLUX.1 schnell": {
        "id": "black-forest-labs/FLUX.1-schnell",
        "pipeline_class": FluxPipeline,  # AutoPipelineForText2Image może być lepszy do obsługi transformerów
        "params": {
            "num_inference_steps": 4,
            "guidance_scale": 0.0,
        },  # Zalecane dla schnell
        "torch_dtype": torch.bfloat16,  # FLUX często używa bfloat16
    },
    "FLUX.1 dev": {
        "id": "black-forest-labs/FLUX.1-dev",
        "pipeline_class": FluxPipeline,  # AutoPipelineForText2Image może być lepszy
        "params": {
            "num_inference_steps": 50,
            "guidance_scale": 3.5,
        },  # Przykładowe dla dev
        "torch_dtype": torch.bfloat16,
    },
}

# Sprawdzenie dostępności CUDA
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Tworzenie głównego folderu wyjściowego
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Główna pętla generacji ---
for model_name, config in MODEL_CONFIG.items():
    print(f"\n--- Generating images for: {model_name} ---")
    model_output_dir = os.path.join(
        OUTPUT_DIR, model_name.replace(" ", "_").replace(".", "")
    )
    os.makedirs(model_output_dir, exist_ok=True)

    pipe = None
    try:
        print(f"Loading pipeline for {config['id']}...")
        pipeline_kwargs = {
            "torch_dtype": config.get("torch_dtype", torch.float32),
        }
        if config.get("variant"):
            pipeline_kwargs["variant"] = config["variant"]
        if HF_TOKEN and "diffusers" in config["id"]:  # SD3 jest gated
            pipeline_kwargs["token"] = HF_TOKEN

        # Dla modeli FLUX, AutoPipelineForText2Image może lepiej zarządzać wymaganymi transformerami
        if "FLUX" in model_name:
            pipe = AutoPipelineForText2Image.from_pretrained(
                config["id"], **pipeline_kwargs
            )
        else:
            pipe = config["pipeline_class"].from_pretrained(
                config["id"], **pipeline_kwargs
            )

        pipe = pipe.to(device)
        print(f"Pipeline for {model_name} loaded successfully.")

        for i, seed in enumerate(SEEDS):
            print(
                f"Generating image {i + 1}/{NUM_IMAGES_PER_MODEL} with seed {seed}..."
            )

            # Ustawienie generatora dla reprodukowalności
            generator = torch.Generator(device=device).manual_seed(seed)

            # Parametry specyficzne dla modelu
            current_params = config["params"].copy()

            # Generowanie obrazu
            with torch.no_grad():  # Oszczędność pamięci
                if isinstance(
                    pipe, StableDiffusion3Pipeline
                ):  # SD3 ma inne nazwy parametrów
                    image = pipe(
                        prompt=PROMPT,
                        num_inference_steps=current_params.get(
                            "num_inference_steps", 28
                        ),
                        guidance_scale=current_params.get("guidance_scale", 7.0),
                        generator=generator,
                    ).images[0]
                elif isinstance(pipe, FluxPipeline):
                    image = pipe(
                        prompt=PROMPT,
                        num_inference_steps=current_params.get(
                            "num_inference_steps", 8
                        ),
                        guidance_scale=current_params.get(
                            "guidance_scale", 0.0
                        ),  # FLUX często używa 0.0
                        generator=generator,
                    ).images[0]
                else:  # Dla SD 1.5, 2.1, XL, XL-Turbo
                    image = pipe(
                        prompt=PROMPT,
                        num_inference_steps=current_params["num_inference_steps"],
                        guidance_scale=current_params["guidance_scale"],
                        generator=generator,
                    ).images[0]

            # Zapisywanie obrazu
            image_path = os.path.join(model_output_dir, f"seed_{seed}.png")
            image.save(image_path)
            print(f"Image saved to {image_path}")

            # Zwolnienie pamięci GPU (ważne przy wielu modelach)
            if device == "cuda":
                torch.cuda.empty_cache()

    except Exception as e:
        print(f"ERROR generating images for {model_name}: {e}")
        if "permission" in str(e).lower() or "gated" in str(e).lower():
            print(
                "This might be a gated model. Ensure you have accepted the terms on Hugging Face and are logged in or provided a token."
            )
    finally:
        # Zwolnienie pipeline'u z pamięci
        if pipe is not None:
            del pipe
        gc.collect()  # Garbage collection
        if device == "cuda":
            torch.cuda.empty_cache()
        print(f"Finished processing {model_name}. Cleared memory.")


print("\n--- All image generation complete. ---")
