import os
import requests
import torch
from diffusers import StableDiffusionXLInpaintPipeline
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

def download_file(url, destination):
    print(f"Downloading {url} to {destination}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Download complete.")

def build():
    # 1. Download ClipSeg
    print("Downloading ClipSeg model and processor...")
    transformers_cache = "CIDAS/clipseg-rd64-refined"
    CLIPSegProcessor.from_pretrained(transformers_cache)
    CLIPSegForImageSegmentation.from_pretrained(transformers_cache)

    # 2. Download Base SDXL Inpainting (for config cache)
    print("Downloading SDXL Inpainting base (for configs)...")
    model_id = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
    # We download the pretrained model to cache so `from_single_file` can find configs if needed,
    # or just to populate the huggingface cache.
    # Note: from_single_file usually handles configs well, but having the base repo cached helps avoid runtime downloads for generic components.
    StableDiffusionXLInpaintPipeline.from_pretrained(model_id, torch_dtype=torch.float16)

    # 3. Download Custom Checkpoint
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "model.safetensors")
    
    # CivitAI URL
    model_url = "https://civitai.com/api/download/models/1990969?token=be68b983e1cd67210cc903389e929cc0"
    
    if not os.path.exists(checkpoint_path):
        download_file(model_url, checkpoint_path)
    else:
        print(f"Checkpoint already exists at {checkpoint_path}")

if __name__ == "__main__":
    build()
