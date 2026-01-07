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

print(f"DEBUG: Script v2.0 (Smart Masking + Compositing). Diffusers: {diffusers.__version__}", file=sys.stderr)

from PIL import Image, ImageFilter
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

        # 1. ClipSeg (Сегментация)
        processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        segmentator = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(device)

        # 2. Base Inpainting (Официальная модель - для структуры)
        print("Loading Base Model...")
        pipe_base = StableDiffusionXLInpaintPipeline.from_pretrained(
            "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        ).to(device)
        
        # DPM++ Scheduler для четкости
        pipe_base.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe_base.scheduler.config, use_karras_sigmas=True
        )

        # 3. Big Love (Стиль - через Img2Img)
        print("Loading Style Model...")
        pipe_style = StableDiffusionXLImg2ImgPipeline.from_single_file(
            "./checkpoints/Biglove2.safetensors",
            torch_dtype=torch.float16,
            use_safetensors=True
        ).to(device)
        
        pipe_style.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe_style.scheduler.config, use_karras_sigmas=True
        )
        
        print("✅ Initialization complete.")
        
    except Exception as e:
        print(f"🔥 CRITICAL ERROR: {e}")
        traceback.print_exc()
        import time
        time.sleep(10)
        raise e

def smart_resize(image, target_size=1024):
    """Умный ресайз до ~1024px по большей стороне."""
    width, height = image.size
    aspect_ratio = width / height
    
    if width > height:
        new_width = target_size
        new_height = int(target_size / aspect_ratio)
    else:
        new_height = target_size
        new_width = int(target_size * aspect_ratio)
        
    new_width = (new_width // 8) * 8
    new_height = (new_height // 8) * 8
    
    return image.resize((new_width, new_height), Image.LANCZOS)

def get_mask_advanced(image, include_prompts, exclude_prompts):
    """
    Генерирует маску: (Include - Exclude).
    Это позволяет выделить 'одежду', но вычесть 'лицо' и 'руки'.
    """
    device = segmentator.device
    
    # Подготовка текста
    targets = [p.strip() for p in include_prompts.split(",")]
    anti_targets = [p.strip() for p in exclude_prompts.split(",")] if exclude_prompts else []
    
    all_prompts = targets + anti_targets
    
    inputs = processor(text=all_prompts, images=[image] * len(all_prompts), padding="max_length", return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = segmentator(**inputs)
        
    preds = outputs.logits.unsqueeze(1)
    
    # 1. Складываем всё, что нужно включить (Одежда)
    mask_include = torch.sigmoid(preds[0][0])
    for i in range(1, len(targets)):
        mask_include = torch.max(mask_include, torch.sigmoid(preds[i][0]))
        
    # 2. Складываем всё, что нужно исключить (Лицо, руки)
    if anti_targets:
        mask_exclude = torch.sigmoid(preds[len(targets)][0])
        for i in range(len(targets) + 1, len(all_prompts)):
            mask_exclude = torch.max(mask_exclude, torch.sigmoid(preds[i][0]))
        
        # 3. Вычитаем: Одежда МИНУС Лицо
        final_mask_tensor = mask_include - (mask_exclude * 1.2) # Усиливаем исключение
        final_mask_tensor = torch.clamp(final_mask_tensor, 0, 1)
    else:
        final_mask_tensor = mask_include

    mask_np = final_mask_tensor.cpu().numpy()
    mask_cv = cv2.resize(mask_np, image.size, interpolation=cv2.INTER_CUBIC)
    
    # Порог чуть выше (0.35), чтобы не цеплять фон
    _, binary_mask = cv2.threshold(mask_cv, 0.35, 255, cv2.THRESH_BINARY)
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
    negative_prompt = job_input.get("negative_prompt", "drawing, painting, illustration, render, 3d, cartoon, anime, low quality, blurry, deformed, ugly, bad anatomy, bad hands, text, watermark")
    
    if not image_url or not prompt:
        return {"status": "failed", "error": "Missing input"}

    try:
        generator = None
        if "seed" in job_input:
             generator = torch.Generator(device="cuda").manual_seed(job_input["seed"])

        print(f"🎨 Processing: {image_url}")
        
        # 1. Загрузка и подготовка
        original_image = download_image(image_url)
        processing_image = smart_resize(original_image, target_size=1024)
        
        # 2. Умная Маска
        # Ищем одежду, но ЯВНО исключаем лицо и руки
        mask_target = job_input.get("mask_target", "clothes, dress, suit, tshirt, swimsuit, lingerie, underwear, bra, panties, dress, suit, tshirt, outfit")
        mask_exclude = "face, head, hands, skin" 
        
        print(f"🎭 Generating mask: +[{mask_target}] -[{mask_exclude}]")
        mask_image = get_mask_advanced(processing_image, mask_target, mask_exclude)
        
        # Размываем маску для мягких краев (чтобы не было эффекта "аппликации")
        mask_blurred = mask_image.filter(ImageFilter.GaussianBlur(radius=5))

        # 3. ЭТАП 1: Base Inpainting (Заменяем одежду, сохраняя позу)
        print("🔹 Stage 1: Base Structure...")
        # Маска черная для лица -> модель его не трогает
        inpainted_image = pipe_base(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=processing_image,
            mask_image=mask_image, # Жесткая маска для точности
            height=processing_image.height,
            width=processing_image.width,
            num_inference_steps=25,
            guidance_scale=5.0,
            strength=0.99,
            generator=generator
        ).images[0]
        
        # 4. ЭТАП 2: Big Love (Стиль)
        print("🔸 Stage 2: Big Love Styling...")
        # Прогоняем ВСЮ картинку через стайлер
        style_image = pipe_style(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=inpainted_image,
            num_inference_steps=25,
            strength=0.45, # Достаточно сильно, чтобы наложить текстуры ткани
            guidance_scale=5.0,
            generator=generator
        ).images[0]

        # 5. ЭТАП 3: Композитинг (Восстановление лица)
        print("🔧 Stage 3: Face Restoration (Compositing)...")
        # Мы берем style_image там, где была одежда (mask), 
        # и inpainted_image (где лицо нетронуто) там, где маски нет.
        # Используем размытую маску для плавного перехода.
        final_image = Image.composite(style_image, inpainted_image, mask_blurred)

        # Кодирование
        buffered = io.BytesIO()
        final_image.save(buffered, format="JPEG", quality=95)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        print(f"✅ Job {job_id} success.")
        return {"status": "success", "image": img_str}
        
    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()
        return {"status": "failed", "error": str(e)}

init_handler()
runpod.serverless.start({"handler": handler})