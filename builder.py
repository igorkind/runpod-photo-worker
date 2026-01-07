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
    # 1. ClipSeg (Для создания масок)
    print("Downloading ClipSeg model...")
    snapshot_download(repo_id="CIDAS/clipseg-rd64-refined")

    # 2. (УДАЛЕНО) Официальная модель Inpainting больше не нужна, 
    # так как мы используем Juggernaut как базу.

    # Создаем папку для чекпоинтов
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 3. 🔥 Juggernaut XL v9 (Основная модель)
    checkpoint_path = os.path.join(checkpoint_dir, "JuggernautXL_v9.safetensors")
    
    # Ссылка на CivitAI (с токеном)
    model_url = "https://civitai.com/api/download/models/348913?token=be68b983e1cd67210cc903389e929cc0"
    
    if not os.path.exists(checkpoint_path):
        print("Downloading Juggernaut XL...")
        download_file(model_url, checkpoint_path)
    else:
        print(f"Checkpoint already exists at {checkpoint_path}")

    # 4. Add Detail LoRA (Детализация)
    lora_path = os.path.join(checkpoint_dir, "add-detail-xl.safetensors")
    lora_url = "https://civitai.com/api/download/models/135867?type=Model&format=SafeTensor"
    
    if not os.path.exists(lora_path):
        print("Downloading Detail LoRA...")
        download_file(lora_url, lora_path)
    else:
        print(f"LoRA already exists at {lora_path}")

if __name__ == "__main__":
    build()