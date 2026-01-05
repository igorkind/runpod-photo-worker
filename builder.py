import os
import requests
from huggingface_hub import snapshot_download

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
    print("Downloading ClipSeg model and processor (snapshot)...")
    snapshot_download(repo_id="CIDAS/clipseg-rd64-refined")

    # 2. Download Base SDXL Inpainting (for config cache)
    print("Downloading SDXL Inpainting base (snapshot, configs/weights)...")
    # Downloading fp16 safetensors and json configs ensures we have what we need for from_single_file fallback,
    # without loading the huge fp32 model into RAM.
    snapshot_download(
        repo_id="diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        allow_patterns=["*.json", "*.txt"]  # <--- ТОЛЬКО КОНФИГИ!
    )

    # 3. Download Custom Checkpoint
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "Biglove2.safetensors")
    
    # CivitAI URL
    model_url = "https://civitai.com/api/download/models/1990969?token=be68b983e1cd67210cc903389e929cc0"
    
    if not os.path.exists(checkpoint_path):
        download_file(model_url, checkpoint_path)
    else:
        print(f"Checkpoint already exists at {checkpoint_path}")

if __name__ == "__main__":
    build()
