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
    # 1. ClipSeg
    print("Downloading ClipSeg...")
    snapshot_download(repo_id="CIDAS/clipseg-rd64-refined")

    # 2. Base Inpainting (Официальная база - оставляем, она лучшая для масок)
    print("Downloading Base Inpainting...")
    snapshot_download(
        repo_id="diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        allow_patterns=["*.fp16.safetensors", "*.json", "*.txt"]
    )

    os.makedirs("checkpoints", exist_ok=True)

    # 3. 🔥 Big Love XL v4 (Вместо Juggernaut)
    checkpoint_path = "checkpoints/BigLoveXL_v4.safetensors"
    # Ваша ссылка на XL4
    model_url = "https://civitai.com/api/download/models/1990969?token=be68b983e1cd67210cc903389e929cc0"
    
    if not os.path.exists(checkpoint_path):
        print("Downloading Big Love XL v4...")
        download_file(model_url, checkpoint_path)

    # 4. Detail LoRA (Оставляем, она универсальная)
    lora_path = "checkpoints/add-detail-xl.safetensors"
    lora_url = "https://civitai.com/api/download/models/135867?type=Model&format=SafeTensor"
    if not os.path.exists(lora_path):
        download_file(lora_url, lora_path)

if __name__ == "__main__":
    build()