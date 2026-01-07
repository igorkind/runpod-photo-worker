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

print(f"DEBUG: Script v1.16 (Quality Boost). Diffusers: {diffusers.__version__}", file=sys.stderr)

from PIL import Image
from diffusers import StableDiffusionXLInpaintPipeline, StableDiffusionXLImg2ImgPipeline, DPMSolverMultistepScheduler
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

# Глобальные переменные
pipe_base = None
pipe_style = None
processor = None
segmentator = None

def init_handler():
    global pipe_base, pipe_style, processor, segmentator
    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 Initializing handler on {device}...")

        # 1. ClipSeg
        processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        segmentator = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(device)

        # 2. Base Inpainting
        print("Loading Base Model...")
        pipe_base = StableDiffusionXLInpaintPipeline.from_pretrained(
            "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        ).to(device)
        
        # 🔥 ВАЖНО: Ставим DPM++ 2M Karras Scheduler (Для четкости)
        pipe_base.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe_base.scheduler.config, use_karras_sigmas=True
        )

        # 3. Big Love (Style)
        print("Loading Style Model...")
        pipe_style = StableDiffusionXLImg2ImgPipeline.from_single_file(
            "./checkpoints/Biglove2.safetensors",
            torch_dtype=torch.float16,
            use_safetensors=True
        ).to(device)
        
        # Тоже ставим DPM++ Scheduler
        pipe_style.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe_style.scheduler.config, use_karras_sigmas=True
        )
        
        print("✅ Initialization complete (High Quality Mode).")
        
    except Exception as e:
        print(f"🔥 CRITICAL ERROR: {e}")
        traceback.print_exc()
        import time
        time.sleep(10)
        raise e

def smart_resize(image, target_size=1024):
    """
    Умный ресайз: 
    1. Если фото маленькое -> увеличивает (Upscale) до ~1024 по большей стороне.
    2. Если фото огромное -> уменьшает до 1024.
    3. Делает стороны кратными 8.
    """
    width, height = image.size
    aspect_ratio = width / height
    
    # Определяем новую ширину и высоту, стремясь к target_size
    if width > height:
        new_width = target_size
        new_height = int(target_size / aspect_ratio)
    else:
        new_height = target_size
        new_width = int(target_size * aspect_ratio)
        
    # Округляем до 8
    new_width = (new_width // 8) * 8
    new_height = (new_height // 8) * 8
    
    # Используем LANCZOS для качественного изменения размера
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

    job_input = event["input"]
    image_url = job_input.get("image_url")
    prompt = job_input.get("prompt")
    # Добавляем "проверенные" негативные промпты для Big Love
    negative_prompt = job_input.get("negative_prompt", "drawing, painting, illustration, render, 3d, cartoon, anime, low quality, blurry, deformed, ugly, bad anatomy, bad hands, text, watermark")
    
    if not image_url or not prompt:
        return {"status": "failed", "error": "Missing input"}

    try:
        generator = None
        if "seed" in job_input:
             generator = torch.Generator(device="cuda").manual_seed(job_input["seed"])

        print(f"🎨 Processing: {image_url}")
        
        # 1. Подготовка и Upscaling
        original_image = download_image(image_url)
        processing_image = smart_resize(original_image, target_size=1024) # <-- Форсируем 1024px
        print(f"📏 Resized to: {processing_image.size}")
        
        mask_target = job_input.get("mask_target", "clothes, dress, suit, tshirt, swimsuit, lingerie, underwear, bra, panties")
        mask_image = get_mask(processing_image, mask_target)
        
        # 2. ЭТАП 1: Base Inpainting (Структура)
        print("🔹 Stage 1: Base Structure...")
        inpainted_image = pipe_base(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=processing_image,
            mask_image=mask_image,
            height=processing_image.height,
            width=processing_image.width,
            num_inference_steps=25,  # Больше шагов для качества
            guidance_scale=5.0,      # <-- Снижаем CFG для реализма (было 7.5)
            strength=0.99,
            generator=generator
        ).images[0]
        
        # 3. ЭТАП 2: Refiner Big Love (Детали)
        print("🔸 Stage 2: Big Love Finish...")
        final_image = pipe_style(
            prompt=prompt, # Тот же промпт
            negative_prompt=negative_prompt,
            image=inpainted_image,
            num_inference_steps=25,
            strength=0.50,       # Чуть сильнее перерисовываем (было 0.35)
            guidance_scale=5.0,  # <-- Тоже 5.0
            generator=generator
        ).images[0]

        buffered = io.BytesIO()
        final_image.save(buffered, format="JPEG", quality=95)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        return {"status": "success", "image": img_str}
        
    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()
        return {"status": "failed", "error": str(e)}

init_handler()
runpod.serverless.start({"handler": handler})