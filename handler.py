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

print(f"DEBUG: Script v3.9 (Fix: Mask Brightness Level). Diffusers: {diffusers.__version__}", file=sys.stderr)

from PIL import Image, ImageFilter
from diffusers import StableDiffusionXLInpaintPipeline, StableDiffusionXLImg2ImgPipeline, DPMSolverMultistepScheduler
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

# Глобальные переменные
pipe_base = None
pipe_style = None
processor = None
segmentator = None
CHECKPOINT_FILE = "JuggernautXL_v9.safetensors"

def init_handler():
    global pipe_base, pipe_style, processor, segmentator
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 Initializing on {device}...")

        # 1. ClipSeg
        processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        segmentator = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(device)

        # 2. Base Inpainting (Официальная модель)
        print("Loading Base Model (Official Inpainting)...")
        pipe_base = StableDiffusionXLInpaintPipeline.from_pretrained(
            "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            safety_checker=None, 
            requires_safety_checker=False
        ).to(device)
        
        pipe_base.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe_base.scheduler.config, use_karras_sigmas=True, algorithm_type="sde-dpmsolver++"
        )

        # 3. Refiner (Juggernaut)
        print(f"Loading Refiner ({CHECKPOINT_FILE})...")
        checkpoint_path = f"./checkpoints/{CHECKPOINT_FILE}"
        
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
            pipe_style.scheduler.config, use_karras_sigmas=True, algorithm_type="sde-dpmsolver++"
        )

        # 4. LoRA
        lora_path = "./checkpoints/add-detail-xl.safetensors"
        try:
            pipe_base.load_lora_weights(lora_path)
            pipe_base.fuse_lora(lora_scale=0.5)
            pipe_style.load_lora_weights(lora_path)
            pipe_style.fuse_lora(lora_scale=0.5)
            print("✅ LoRA fused.")
        except Exception:
            print("⚠️ LoRA skipped.")
        
        print("✅ Initialization complete.")
    except Exception as e:
        print(f"🔥 CRITICAL ERROR: {e}")
        sys.exit(1)

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

def get_mask_tensor(image, targets, anti_targets):
    device = segmentator.device
    all_prompts = targets + anti_targets
    inputs = processor(text=all_prompts, images=[image] * len(all_prompts), padding="max_length", return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = segmentator(**inputs)
    preds = outputs.logits.unsqueeze(1)
    
    mask_include = torch.sigmoid(preds[0][0])
    for i in range(1, len(targets)):
        mask_include = torch.max(mask_include, torch.sigmoid(preds[i][0]))
        
    if anti_targets:
        mask_exclude = torch.sigmoid(preds[len(targets)][0])
        for i in range(len(targets) + 1, len(all_prompts)):
            mask_exclude = torch.max(mask_exclude, torch.sigmoid(preds[i][0]))
        inverted_exclude = 1.0 - mask_exclude
        final_mask_tensor = mask_include * inverted_exclude
    else:
        final_mask_tensor = mask_include
    return final_mask_tensor

def process_mask_from_tensor(mask_tensor, image_size):
    mask_np = mask_tensor.cpu().numpy()
    mask_cv = cv2.resize(mask_np, image_size, interpolation=cv2.INTER_CUBIC)
    
    # 1. Порог: делаем пиксели либо 0 (черный), либо 255 (белый)
    _, binary_mask = cv2.threshold(mask_cv, 0.15, 255, cv2.THRESH_BINARY)
    
    # 2. Считаем покрытие
    non_zero = cv2.countNonZero(binary_mask)
    coverage = non_zero / (binary_mask.shape[0] * binary_mask.shape[1])
    
    # 3. Расширяем маску
    kernel = np.ones((20, 20), np.uint8)
    dilated_mask = cv2.dilate(binary_mask.astype(np.uint8), kernel, iterations=1)
    
    # 🔥 ИСПРАВЛЕНИЕ: dilated_mask УЖЕ имеет значения 255. 
    # Умножать еще раз на 255 НЕЛЬЗЯ (будет переполнение и черный цвет).
    return Image.fromarray(dilated_mask), coverage

def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG", quality=85)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def download_image(url):
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content)).convert("RGB")

def handler(event):
    global pipe_base, pipe_style
    
    job_id = event.get("id", "local_test")
    job_input = event["input"]
    
    try:
        # --- 1. ПАРСИНГ ПАРАМЕТРОВ ---
        image_url = job_input.get("image_url")
        user_prompt = job_input.get("prompt", "")
        
        mask_target_str = job_input.get("mask_target", "clothes, dress, suit, tshirt, outfit, jacket, coat, underwear, swimsuit, underpants")
        mask_exclude_str = job_input.get("mask_exclude", "face, head, hands")
        mask_blur_radius = int(job_input.get("mask_blur", 15))
        
        steps_base = int(job_input.get("steps_base", 20))
        steps_refiner = int(job_input.get("steps_refiner", 25))
        strength_refiner = float(job_input.get("strength_refiner", 0.40))
        guidance_scale = float(job_input.get("guidance_scale", 7.0))
        
        negative_prompt = job_input.get("negative_prompt", "blurry, low quality, distorted face, ugly hands, cartoon, 3d render")
        
        prompt = f"({user_prompt})++, professional photo, realistic, 8k, highly detailed, soft lighting"
        
        generator = torch.Generator(device="cuda").manual_seed(job_input.get("seed", 42))
        
        # --- 2. ЗАГРУЗКА И МАСКИРОВАНИЕ ---
        if not image_url:
            processing_image = Image.new("RGB", (832, 1216), (0,0,0))
            mask_image = Image.new("L", (832, 1216), 255)
            coverage = 1.0
        else:
            original_image = download_image(image_url)
            processing_image = smart_resize(original_image, target_size=1024)
            
            targets = [t.strip() for t in mask_target_str.split(",")]
            excludes = [t.strip() for t in mask_exclude_str.split(",")]
            
            print(f"🎭 Mask Targets: {targets}")
            print(f"🛡️ Mask Excludes: {excludes}")
            
            mask_tensor = get_mask_tensor(processing_image, targets, excludes)
            mask_image, coverage = process_mask_from_tensor(mask_tensor, processing_image.size)
            print(f"📊 Coverage: {coverage:.2%}")

            # Fallback
            if coverage < 0.10:
                print("⚠️ Coverage low (<10%). Switching to Full Body Mask.")
                mask_tensor = get_mask_tensor(processing_image, ["person", "body"], ["face"])
                mask_image, coverage = process_mask_from_tensor(mask_tensor, processing_image.size)

        mask_blurred = mask_image.filter(ImageFilter.GaussianBlur(radius=mask_blur_radius))

        # --- 3. ГЕНЕРАЦИЯ ---
        
        # Stage 1: Base
        print(f"🔹 Stage 1: Steps={steps_base}, CFG={guidance_scale}...")
        # Для Inpainting strength=1.0 означает полную замену под маской
        inpainted_image = pipe_base(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=processing_image,
            mask_image=mask_image, 
            height=processing_image.height,
            width=processing_image.width,
            num_inference_steps=steps_base,
            guidance_scale=guidance_scale,
            strength=0.99 if image_url else 1.0,
            generator=generator
        ).images[0]
        
        # Stage 2: Refiner
        print(f"🔸 Stage 2: Steps={steps_refiner}, Strength={strength_refiner}...")
        style_image = pipe_style(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=inpainted_image,
            num_inference_steps=steps_refiner,
            strength=strength_refiner, 
            guidance_scale=guidance_scale,
            generator=generator
        ).images[0]

        # Compositing
        if image_url:
            print("🔧 Stage 3: Compositing...")
            final_image = Image.composite(style_image, processing_image, mask_blurred)
        else:
            final_image = style_image

        return {
            "status": "success",
            "images": {
                "1_mask": image_to_base64(mask_image.convert("RGB")),
                "2_stage1": image_to_base64(inpainted_image),
                "3_stage2": image_to_base64(style_image),
                "4_final": image_to_base64(final_image)
            }
        }
        
    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()
        return {"status": "failed", "error": str(e)}

init_handler()
runpod.serverless.start({"handler": handler})