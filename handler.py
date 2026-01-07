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

print(f"DEBUG: Script v2.3 (NoFilter + Ultimate Quality). Diffusers: {diffusers.__version__}", file=sys.stderr)

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

        # 2. Base Inpainting (Структура)
        print("Loading Base Model (Inpainting)...")
        pipe_base = StableDiffusionXLInpaintPipeline.from_pretrained(
            "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            # 🔥 ОТКЛЮЧАЕМ ЦЕНЗУРУ
            safety_checker=None,
            requires_safety_checker=False
        ).to(device)
        
        # SDE Scheduler для реалистичной текстуры
        pipe_base.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe_base.scheduler.config, 
            use_karras_sigmas=True,
            algorithm_type="sde-dpmsolver++"
        )

        # 3. Big Love (Стиль) с оптимизацией памяти
        print("Loading Style Model (Big Love)...")
        checkpoint_path = "./checkpoints/Biglove2.safetensors"
        
        pipe_style = StableDiffusionXLImg2ImgPipeline.from_single_file(
            checkpoint_path,
            torch_dtype=torch.float16,
            use_safetensors=True,
            # Переиспользуем компоненты для экономии VRAM
            text_encoder=pipe_base.text_encoder,
            text_encoder_2=pipe_base.text_encoder_2,
            vae=pipe_base.vae,
            tokenizer=pipe_base.tokenizer,
            tokenizer_2=pipe_base.tokenizer_2,
            # 🔥 ОТКЛЮЧАЕМ ЦЕНЗУРУ
            safety_checker=None,
            requires_safety_checker=False
        ).to(device)
        
        pipe_style.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe_style.scheduler.config, 
            use_karras_sigmas=True,
            algorithm_type="sde-dpmsolver++"
        )

        # 4. Подключаем Detail LoRA (если есть)
        lora_path = "./checkpoints/add-detail-xl.safetensors"
        try:
            print("Loading Detail LoRA...")
            pipe_style.load_lora_weights(lora_path)
            pipe_style.fuse_lora(lora_scale=0.6) # Сила детализации 0.6
            print("✅ LoRA fused.")
        except Exception:
            print("⚠️ LoRA not found, skipping (check builder.py).")
        
        print("✅ Initialization complete.")
        
    except Exception as e:
        print(f"🔥 CRITICAL ERROR: {e}")
        traceback.print_exc()
        import time
        time.sleep(10)
        raise e

def smart_resize(image, target_size=1024):
    """Ресайз с сохранением пропорций до ~1024px."""
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
    """Умная маска: включает одежду, вычитает лицо/руки + расширяет края."""
    device = segmentator.device
    
    targets = [p.strip() for p in include_prompts.split(",")]
    anti_targets = [p.strip() for p in exclude_prompts.split(",")] if exclude_prompts else []
    all_prompts = targets + anti_targets
    
    inputs = processor(text=all_prompts, images=[image] * len(all_prompts), padding="max_length", return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = segmentator(**inputs)
    
    preds = outputs.logits.unsqueeze(1)
    
    # Складываем всё, что нужно включить
    mask_include = torch.sigmoid(preds[0][0])
    for i in range(1, len(targets)):
        mask_include = torch.max(mask_include, torch.sigmoid(preds[i][0]))
        
    # Вычитаем то, что нужно исключить
    if anti_targets:
        mask_exclude = torch.sigmoid(preds[len(targets)][0])
        for i in range(len(targets) + 1, len(all_prompts)):
            mask_exclude = torch.max(mask_exclude, torch.sigmoid(preds[i][0]))
        final_mask_tensor = mask_include - (mask_exclude * 1.5)
        final_mask_tensor = torch.clamp(final_mask_tensor, 0, 1)
    else:
        final_mask_tensor = mask_include

    mask_np = final_mask_tensor.cpu().numpy()
    mask_cv = cv2.resize(mask_np, image.size, interpolation=cv2.INTER_CUBIC)
    _, binary_mask = cv2.threshold(mask_cv, 0.35, 255, cv2.THRESH_BINARY)
    
    # Расширяем маску, чтобы захватить швы
    kernel = np.ones((10, 10), np.uint8)
    dilated_mask = cv2.dilate(binary_mask.astype(np.uint8), kernel, iterations=1)
    
    return Image.fromarray(dilated_mask * 255)

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
    negative_prompt = job_input.get("negative_prompt", "")
    
    if not prompt:
        return {"status": "failed", "error": "Missing prompt"}

    try:
        generator = None
        if "seed" in job_input:
             generator = torch.Generator(device="cuda").manual_seed(job_input["seed"])

        # ОПРЕДЕЛЯЕМ РЕЖИМ (Текст или Фото)
        is_t2i = False
        if not image_url:
            is_t2i = True
            print("✨ Mode: Text-to-Image")
        else:
            print(f"🎨 Mode: Inpainting for {image_url}")

        # 1. ПОДГОТОВКА ИЗОБРАЖЕНИЯ И МАСКИ
        if is_t2i:
            # Создаем пустой холст
            width = job_input.get("width", 832) 
            height = job_input.get("height", 1216)
            processing_image = Image.new("RGB", (width, height), (0, 0, 0)) # Черный фон
            
            # Создаем ПОЛНУЮ белую маску (рисуем везде)
            mask_image = Image.new("L", (width, height), 255)
            mask_blurred = mask_image 
        else:
            # Скачиваем и ресайзим фото
            original_image = download_image(image_url)
            processing_image = smart_resize(original_image, target_size=1024)
            
            # Генерируем умную маску
            mask_target = job_input.get("mask_target", "clothes, dress, suit, tshirt, outfit, swimsuit, lingerie")
            mask_exclude = "face, head, hands, skin, hair"
            
            print(f"🎭 Generating smart mask...")
            mask_image = get_mask_advanced(processing_image, mask_target, mask_exclude)
            mask_blurred = mask_image.filter(ImageFilter.GaussianBlur(radius=9))

        # 2. ЭТАП 1: Base Generation (Структура)
        print("🔹 Stage 1: Base Structure...")
        strength_val = 1.0 if is_t2i else 0.99
        
        inpainted_image = pipe_base(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=processing_image,
            mask_image=mask_image, 
            height=processing_image.height,
            width=processing_image.width,
            num_inference_steps=25,
            guidance_scale=5.0,
            strength=strength_val,
            generator=generator
        ).images[0]
        
        # 3. ЭТАП 2: Refiner Big Love (Стиль и Детали)
        print("🔸 Stage 2: Big Love Styling...")
        style_image = pipe_style(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=inpainted_image,
            num_inference_steps=30,
            strength=0.45, 
            guidance_scale=5.0,
            generator=generator
        ).images[0]

        # 4. ФИНАЛ: Композитинг (Только для Inpainting)
        if is_t2i:
            final_image = style_image
        else:
            print("🔧 Stage 3: Compositing (Face Restore)...")
            # Накладываем новую одежду на старое фото (чтобы сохранить лицо оригинала)
            final_image = Image.composite(style_image, inpainted_image, mask_blurred)

        # Кодирование в Base64
        buffered = io.BytesIO()
        final_image.save(buffered, format="JPEG", quality=98, subsampling=0)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        print(f"✅ Job {job_id} success.")
        return {"status": "success", "image": img_str}
        
    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()
        return {"status": "failed", "error": str(e)}

init_handler()
runpod.serverless.start({"handler": handler})