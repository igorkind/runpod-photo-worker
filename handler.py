import sys
# Debug-маячок для логов
print("DEBUG: Script started...", file=sys.stderr)

import runpod
import torch
import requests
import base64
import io
import cv2
import numpy as np
import traceback
from PIL import Image
from diffusers import AutoPipelineForInpainting, AutoPipelineForTextToImage
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

# Глобальные переменные
pipe_inpaint = None
pipe_t2i = None
processor = None
segmentator = None

def init_handler():
    global pipe_inpaint, pipe_t2i, processor, segmentator
    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 Initializing handler on {device}...")

        # 1. Загрузка ClipSeg
        print("Loading ClipSeg...")
        processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        segmentator = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(device)

        # 2. Загрузка SDXL Inpainting (из твоего чекпоинта)
        checkpoint_path = "./checkpoints/Biglove2.safetensors"
        print(f"Loading SDXL from {checkpoint_path}...")
        
        # Загружаем Inpainting pipeline
        pipe_inpaint = AutoPipelineForInpainting.from_single_file(
            checkpoint_path,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16"
        ).to(device)
        
        # 3. Создаем T2I пайплайн из первого (общая память)
        print("Creating Text-to-Image Pipeline from shared weights...")
        pipe_t2i = AutoPipelineForTextToImage.from_pipe(pipe_inpaint)
        
        print("✅ Initialization complete.")
        
    except Exception as e:
        print(f"🔥 CRITICAL ERROR during init: {e}")
        traceback.print_exc()
        import time
        time.sleep(10) # Даем время прочитать логи перед падением
        raise e

def smart_resize(image, max_side=1024):
    """Ресайз с сохранением пропорций, кратный 8."""
    width, height = image.size
    if max(width, height) > max_side:
        scale = max_side / max(width, height)
        new_width = int(width * scale)
        new_height = int(height * scale)
    else:
        new_width, new_height = width, height
        
    # Округляем до кратного 8 (требование VAE)
    new_width = (new_width // 8) * 8
    new_height = (new_height // 8) * 8
    
    return image.resize((new_width, new_height), Image.LANCZOS)

def get_mask(image, text_prompts):
    """Генерация маски через ClipSeg."""
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
    # Порог 0.3 для лучшего захвата одежды
    _, binary_mask = cv2.threshold(mask_cv, 0.3, 255, cv2.THRESH_BINARY)
    return Image.fromarray(binary_mask.astype(np.uint8))

def download_image(url):
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content)).convert("RGB")

def handler(event):
    global pipe_inpaint, pipe_t2i
    
    # 1. Получаем ID задачи (для логов и ответа)
    job_id = event.get("id", "local_test")
    print(f"🎬 Starting job: {job_id}")

    # Если инициализация упала
    if pipe_inpaint is None:
        return {"status": "failed", "job_id": job_id, "error": "Model not initialized"}

    job_input = event["input"]
    image_url = job_input.get("image_url")
    prompt = job_input.get("prompt")
    negative_prompt = job_input.get("negative_prompt", "blurry, low quality, distortion")
    
    if not prompt:
        return {"status": "failed", "job_id": job_id, "error": "Missing prompt"}

    try:
        generator = None
        if "seed" in job_input:
             generator = torch.Generator(device="cuda").manual_seed(job_input["seed"])

        # СЦЕНАРИЙ 1: Inpainting (есть ссылка на фото)
        if image_url:
            print(f"🎨 Mode: Inpainting for {image_url}")
            mask_target = job_input.get("mask_target", "clothes, dress, suit, tshirt")
            
            original_image = download_image(image_url)
            # Умный ресайз (сохраняет 9:16 и качество)
            processing_image = smart_resize(original_image)
            
            mask_image = get_mask(processing_image, mask_target)
            
            output_images = pipe_inpaint(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=processing_image,
                mask_image=mask_image,
                height=processing_image.height,
                width=processing_image.width,
                num_inference_steps=job_input.get("steps", 30),
                guidance_scale=job_input.get("guidance_scale", 7.5),
                strength=job_input.get("strength", 0.99), # Высокая сила для полной замены
                generator=generator
            ).images

        # СЦЕНАРИЙ 2: Text-to-Image (нет фото)
        else:
            print("✨ Mode: Text-to-Image")
            width = job_input.get("width", 832)  # По умолчанию портрет
            height = job_input.get("height", 1216)
            
            output_images = pipe_t2i(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=job_input.get("steps", 30),
                guidance_scale=job_input.get("guidance_scale", 7.5),
                generator=generator
            ).images
        
        # Возврат результата
        buffered = io.BytesIO()
        output_images[0].save(buffered, format="JPEG", quality=95)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        # Возвращаем структурированный ответ с ID
        print(f"✅ Job {job_id} completed successfully.")
        return {
            "status": "success",
            "job_id": job_id,
            "image": img_str
        }
        
    except Exception as e:
        print(f"❌ Error in job {job_id}: {e}")
        traceback.print_exc()
        # Возвращаем ошибку с ID
        return {
            "status": "failed",
            "job_id": job_id,
            "error": str(e)
        }

init_handler()
runpod.serverless.start({"handler": handler})