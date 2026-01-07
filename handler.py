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

print(f"DEBUG: Script v2.4 (Fix Masking). Diffusers: {diffusers.__version__}", file=sys.stderr)

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

        # 1. ClipSeg
        processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        segmentator = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(device)

        # 2. Base Inpainting
        print("Loading Base Model...")
        pipe_base = StableDiffusionXLInpaintPipeline.from_pretrained(
            "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            safety_checker=None,
            requires_safety_checker=False
        ).to(device)
        
        pipe_base.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe_base.scheduler.config, 
            use_karras_sigmas=True,
            algorithm_type="sde-dpmsolver++"
        )

        # 3. Big Love
        print("Loading Style Model...")
        checkpoint_path = "./checkpoints/Biglove2.safetensors"
        
        pipe_style = StableDiffusionXLImg2ImgPipeline.from_single_file(
            checkpoint_path,
            torch_dtype=torch.float16,
            use_safetensors=True,
            text_encoder=pipe_base.text_encoder,
            text_encoder_2=pipe_base.text_encoder_2,
            vae=pipe_base.vae,
            tokenizer=pipe_base.tokenizer,
            tokenizer_2=pipe_base.tokenizer_2,
            safety_checker=None,
            requires_safety_checker=False
        ).to(device)
        
        pipe_style.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe_style.scheduler.config, 
            use_karras_sigmas=True,
            algorithm_type="sde-dpmsolver++"
        )

        # 4. LoRA
        lora_path = "./checkpoints/add-detail-xl.safetensors"
        try:
            print("Loading Detail LoRA...")
            pipe_style.load_lora_weights(lora_path)
            pipe_style.fuse_lora(lora_scale=0.6)
            print("✅ LoRA fused.")
        except Exception:
            print("⚠️ LoRA skipped.")
        
        print("✅ Initialization complete.")
        
    except Exception as e:
        print(f"🔥 CRITICAL ERROR: {e}")
        traceback.print_exc()
        import time
        time.sleep(10)
        raise e

def smart_resize(image, target_size=1024):
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
    device = segmentator.device
    
    targets = [p.strip() for p in include_prompts.split(",")]
    anti_targets = [p.strip() for p in exclude_prompts.split(",")] if exclude_prompts else []
    all_prompts = targets + anti_targets
    
    inputs = processor(text=all_prompts, images=[image] * len(all_prompts), padding="max_length", return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = segmentator(**inputs)
    
    preds = outputs.logits.unsqueeze(1)
    
    # 1. ВКЛЮЧАЕМ (Одежда)
    mask_include = torch.sigmoid(preds[0][0])
    for i in range(1, len(targets)):
        mask_include = torch.max(mask_include, torch.sigmoid(preds[i][0]))
        
    # 2. ИСКЛЮЧАЕМ (Лицо)
    if anti_targets:
        mask_exclude = torch.sigmoid(preds[len(targets)][0])
        for i in range(len(targets) + 1, len(all_prompts)):
            mask_exclude = torch.max(mask_exclude, torch.sigmoid(preds[i][0]))
        
        # Исправление: Меньший коэффициент вычитания (1.0 вместо 1.5)
        # Чтобы не стирать одежду, если она рядом с лицом
        final_mask_tensor = mask_include - (mask_exclude * 1.0)
        final_mask_tensor = torch.clamp(final_mask_tensor, 0, 1)
    else:
        final_mask_tensor = mask_include

    mask_np = final_mask_tensor.cpu().numpy()
    mask_cv = cv2.resize(mask_np, image.size, interpolation=cv2.INTER_CUBIC)
    
    # Исправление: Порог 0.2 (был 0.35) - захватит больше одежды
    _, binary_mask = cv2.threshold(mask_cv, 0.20, 255, cv2.THRESH_BINARY)
    
    # Dilate: Расширяем маску на 15 пикселей, чтобы перекрыть швы
    kernel = np.ones((15, 15), np.uint8)
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

        is_t2i = False
        if not image_url:
            is_t2i = True
            print("✨ Mode: Text-to-Image")
        else:
            print(f"🎨 Mode: Inpainting for {image_url}")

        if is_t2i:
            width = job_input.get("width", 832) 
            height = job_input.get("height", 1216)
            processing_image = Image.new("RGB", (width, height), (0, 0, 0))
            mask_image = Image.new("L", (width, height), 255)
            mask_blurred = mask_image 
        else:
            original_image = download_image(image_url)
            processing_image = smart_resize(original_image, target_size=1024)
            
            # ИСПРАВЛЕНИЕ: Убрали 'skin' и 'hair' из исключений! Это важно.
            mask_target = job_input.get("mask_target", "clothes, dress, suit, tshirt, outfit, jacket, coat")
            mask_exclude = "face, head, hands" 
            
            print(f"🎭 Generating mask...")
            mask_image = get_mask_advanced(processing_image, mask_target, mask_exclude)
            
            # Проверка на пустую маску
            if mask_image.getbbox() is None:
                print("⚠️ WARNING: Empty mask detected! Fallback to full image processing.")
                mask_image = Image.new("L", processing_image.size, 255)
            
            mask_blurred = mask_image.filter(ImageFilter.GaussianBlur(radius=15))

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

        if is_t2i:
            final_image = style_image
        else:
            print("🔧 Stage 3: Compositing...")
            # ИСПРАВЛЕНИЕ: Используем processing_image (Оригинал) как фон, чтобы вернуть оригинальное лицо
            final_image = Image.composite(style_image, processing_image, mask_blurred)

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