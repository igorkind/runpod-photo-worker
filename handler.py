import sys
import runpod
import torch
import requests
import base64
import io
import cv2
import numpy as np
import traceback
import diffusers
import transformers

print(f"DEBUG: Script started. Diffusers: {diffusers.__version__}", file=sys.stderr)

from PIL import Image
# Импортируем конкретные классы для стабильности
from diffusers import StableDiffusionXLInpaintPipeline, StableDiffusionXLImg2ImgPipeline
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

# Глобальные переменные
pipe_base = None   # Официальный Inpainting
pipe_style = None  # Big Love
processor = None
segmentator = None

def init_handler():
    global pipe_base, pipe_style, processor, segmentator
    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 Initializing handler on {device}...")

        # 1. ClipSeg
        print("Loading ClipSeg...")
        processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        segmentator = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(device)

        # 2. Загружаем БАЗУ (Официальный Inpainting)
        print("Loading Base Inpainting Model (Official)...")
        pipe_base = StableDiffusionXLInpaintPipeline.from_pretrained(
            "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        ).to(device)

        # 3. Загружаем СТИЛЬ (Big Love) как Img2Img
        checkpoint_path = "./checkpoints/Biglove2.safetensors"
        print(f"Loading Style Model (Big Love) from {checkpoint_path}...")
        
        pipe_style = StableDiffusionXLImg2ImgPipeline.from_single_file(
            checkpoint_path,
            torch_dtype=torch.float16,
            use_safetensors=True
            # Здесь не нужны флаги ignore_mismatched, так как мы грузим её в родной Img2Img
        ).to(device)
        
        # Соединяем компоненты, чтобы сэкономить память (опционально)
        # pipe_style.text_encoder = pipe_base.text_encoder
        # pipe_style.text_encoder_2 = pipe_base.text_encoder_2
        # pipe_style.vae = pipe_base.vae
        
        print("✅ Initialization complete (Dual Model Mode).")
        
    except Exception as e:
        print(f"🔥 CRITICAL ERROR during init: {e}")
        traceback.print_exc()
        import time
        time.sleep(10)
        raise e

def smart_resize(image, max_side=1024):
    width, height = image.size
    if max(width, height) > max_side:
        scale = max_side / max(width, height)
        new_width = int(width * scale)
        new_height = int(height * scale)
    else:
        new_width, new_height = width, height
    new_width = (new_width // 8) * 8
    new_height = (new_height // 8) * 8
    return image.resize((new_width, new_height), Image.LANCZOS)

def get_mask(image, text_prompts):
    device = segmentator.device
    prompts = [p.strip() for p in text_prompts.split(",")]
    inputs = processor(text=prompts, images=[image] * len(prompts), padding="max_length", return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = segmentator(**inputs)
    preds = outputs.logits.unsqueeze(1)
    combined_mask = torch.sigmoid(preds[0][0])
    for i in range(1, len(prompts)):
        combined_mask = torch.max(combined_mask, torch.sigmoid(preds[i][0]))
    mask_np = combined_mask.cpu().numpy()
    mask_cv = cv2.resize(mask_np, image.size, interpolation=cv2.INTER_CUBIC)
    _, binary_mask = cv2.threshold(mask_cv, 0.3, 255, cv2.THRESH_BINARY)
    return Image.fromarray(binary_mask.astype(np.uint8))

def download_image(url):
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content)).convert("RGB")

def handler(event):
    global pipe_base, pipe_style
    
    job_id = event.get("id", "local_test")
    print(f"🎬 Starting job: {job_id}")

    if pipe_base is None or pipe_style is None:
        return {"status": "failed", "error": "Models not initialized"}

    job_input = event["input"]
    image_url = job_input.get("image_url")
    prompt = job_input.get("prompt")
    negative_prompt = job_input.get("negative_prompt", "blurry, low quality, distortion")
    mask_target = job_input.get("mask_target", "clothes, dress, suit, tshirt")
    
    if not image_url or not prompt:
        return {"status": "failed", "error": "Missing input"}

    try:
        generator = None
        if "seed" in job_input:
             generator = torch.Generator(device="cuda").manual_seed(job_input["seed"])

        print(f"🎨 Processing: {image_url}")
        
        # 1. Подготовка
        original_image = download_image(image_url)
        processing_image = smart_resize(original_image)
        mask_image = get_mask(processing_image, mask_target)
        
        # 2. ЭТАП 1: Inpainting (Base Model)
        # Закрашиваем структуру
        print("🔹 Stage 1: Base Inpainting...")
        inpainted_image = pipe_base(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=processing_image,
            mask_image=mask_image,
            height=processing_image.height,
            width=processing_image.width,
            num_inference_steps=20, # Быстрый проход
            guidance_scale=7.5,
            strength=0.99, # Полная замена
            generator=generator
        ).images[0]
        
        # 3. ЭТАП 2: Refiner (Big Love)
        # Улучшаем стиль и детали
        print("🔸 Stage 2: Big Love Styling...")
        final_image = pipe_style(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=inpainted_image, # Берем результат первого этапа
            num_inference_steps=25,
            strength=0.35, # <--- Важно! Меняем только 35% пикселей (стилизация), сохраняя структуру
            guidance_scale=7.5,
            generator=generator
        ).images[0]

        # 4. Результат
        buffered = io.BytesIO()
        final_image.save(buffered, format="JPEG", quality=95)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        print(f"✅ Job {job_id} success.")
        return {
            "status": "success",
            "job_id": job_id,
            "image": img_str
        }
        
    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()
        return {"status": "failed", "error": str(e)}

init_handler()
runpod.serverless.start({"handler": handler})