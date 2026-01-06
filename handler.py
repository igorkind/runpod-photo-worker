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

# Лог для подтверждения обновления версии кода
print(f"DEBUG: Script started v1.13. Diffusers: {diffusers.__version__}", file=sys.stderr)

from PIL import Image
# Используем AutoPipeline - он самый надежный для кастомных моделей
from diffusers import AutoPipelineForInpainting
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

# Глобальные переменные
pipe_inpaint = None
processor = None
segmentator = None

def init_handler():
    global pipe_inpaint, processor, segmentator
    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 Initializing handler on {device}...")

        # 1. ClipSeg
        print("Loading ClipSeg...")
        processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        segmentator = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(device)

        # 2. SDXL Inpainting
        checkpoint_path = "./checkpoints/Biglove2.safetensors"
        print(f"Loading SDXL Inpainting from {checkpoint_path}...")
        
        # Используем AutoPipelineForInpainting
        # Он автоматически сконвертирует веса 4-канальной модели в 9-канальную
        pipe_inpaint = AutoPipelineForInpainting.from_single_file(
            checkpoint_path,
            torch_dtype=torch.float16,
            # Убрали variant="fp16", так как он часто ломает конвертацию весов
            # variant="fp16", 
            use_safetensors=True,
            
            # КРИТИЧЕСКИ ВАЖНЫЕ ФЛАГИ ДЛЯ ЗАГРУЗКИ ОБЫЧНОЙ МОДЕЛИ В INPAINTING:
            ignore_mismatched_sizes=True, 
            low_cpu_mem_usage=False
        ).to(device)
        
        print("✅ Initialization complete (Inpainting Mode).")
        
    except Exception as e:
        print(f"🔥 CRITICAL ERROR during init: {e}")
        traceback.print_exc()
        import time
        time.sleep(10)
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
        
    new_width = (new_width // 8) * 8
    new_height = (new_height // 8) * 8
    
    return image.resize((new_width, new_height), Image.LANCZOS)

def get_mask(image, text_prompts):
    """Генерация маски."""
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
    global pipe_inpaint
    
    job_id = event.get("id", "local_test")
    print(f"🎬 Starting job: {job_id}")

    if pipe_inpaint is None:
        return {"status": "failed", "job_id": job_id, "error": "Model not initialized"}

    job_input = event["input"]
    image_url = job_input.get("image_url")
    prompt = job_input.get("prompt")
    negative_prompt = job_input.get("negative_prompt", "blurry, low quality, distortion")
    
    if not image_url:
        return {"status": "failed", "job_id": job_id, "error": "image_url is required"}
    if not prompt:
        return {"status": "failed", "job_id": job_id, "error": "prompt is required"}

    try:
        generator = None
        if "seed" in job_input:
             generator = torch.Generator(device="cuda").manual_seed(job_input["seed"])

        print(f"🎨 Processing: {image_url}")
        mask_target = job_input.get("mask_target", "clothes, dress, suit, tshirt")
        
        original_image = download_image(image_url)
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
            strength=job_input.get("strength", 0.99),
            generator=generator
        ).images

        buffered = io.BytesIO()
        output_images[0].save(buffered, format="JPEG", quality=95)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        print(f"✅ Job {job_id} success.")
        return {
            "status": "success",
            "job_id": job_id,
            "image": img_str
        }
        
    except Exception as e:
        print(f"❌ Error in job {job_id}: {e}")
        traceback.print_exc()
        return {
            "status": "failed",
            "job_id": job_id,
            "error": str(e)
        }

init_handler()
runpod.serverless.start({"handler": handler})