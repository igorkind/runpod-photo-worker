import runpod
import torch
import requests
import base64
import io
import cv2
import numpy as np
from PIL import Image
from diffusers import AutoPipelineForInpainting, AutoPipelineForTextToImage
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

# Global variables for caching models (Cold Start optimization)
pipe_inpaint = None
pipe_t2i = None
processor = None
segmentator = None

def init_handler():
    global pipe_inpaint, pipe_t2i, processor, segmentator
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing handler on {device}...")

    # Load ClipSeg
    print("Loading ClipSeg...")
    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    segmentator = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(device)

    # Load SDXL Inpainting from Custom Checkpoint
    checkpoint_path = "./checkpoints/Biglove2.safetensors"
    print(f"Loading SDXL Pipeline from {checkpoint_path}...")
    
    # 1. Load Inpainting Pipeline
    pipe_inpaint = AutoPipelineForInpainting.from_single_file(
        checkpoint_path,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16"
    )
    
    # 2. Create Text-to-Image Pipeline from Inpainting (shares weights/memory)
    print("Creating Text-to-Image Pipeline from shared weights...")
    pipe_t2i = AutoPipelineForTextToImage.from_pipe(pipe_inpaint)

    # Optimizations
    if device == "cuda":
        # Enable cancellation of CPU offload for one to switch to other if needed, 
        # but standard enable_model_cpu_offload works per-pipeline instance or shared.
        # Since they share components, offloading one usually handles it for the shared components.
        pipe_inpaint.enable_model_cpu_offload() 
        pipe_t2i.enable_model_cpu_offload()
    
    print("Initialization complete.")

def smart_resize(image, max_side=1024):
    """
    Resizes image so that the longest side does not exceed max_side,
    maintaining aspect ratio. Ensures dimensions are multiples of 8.
    """
    width, height = image.size
    
    if max(width, height) > max_side:
        scale = max_side / max(width, height)
        new_width = int(width * scale)
        new_height = int(height * scale)
    else:
        new_width = width
        new_height = height
        
    # Ensure multiples of 8 for VAE
    new_width = (new_width // 8) * 8
    new_height = (new_height // 8) * 8
    
    # Prevent degenerate dimensions
    if new_width < 8: new_width = 8
    if new_height < 8: new_height = 8

    return image.resize((new_width, new_height), Image.LANCZOS)

def get_mask(image, text_prompts):
    """
    Generates a binary mask for the given image and text prompts using ClipSeg.
    """
    device = segmentator.device
    
    # Prepare input
    if isinstance(text_prompts, str):
        prompts = [p.strip() for p in text_prompts.split(",")]
    else:
        prompts = text_prompts
        
    inputs = processor(text=prompts, images=[image] * len(prompts), padding="max_length", return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = segmentator(**inputs)
        
    preds = outputs.logits.unsqueeze(1) # [B, 1, H, W]
    
    # Combined mask
    combined_mask = torch.sigmoid(preds[0][0])
    for i in range(1, len(prompts)):
        combined_mask = torch.max(combined_mask, torch.sigmoid(preds[i][0]))

    # Convert to numpy
    mask_np = combined_mask.cpu().numpy()
    
    # Resize mask to original image size
    mask_cv = cv2.resize(mask_np, image.size, interpolation=cv2.INTER_CUBIC)
    
    # Thresholding to binary mask (0.3)
    _, binary_mask = cv2.threshold(mask_cv, 0.3, 255, cv2.THRESH_BINARY)
    
    # Convert back to PIL Image
    mask_image = Image.fromarray(binary_mask.astype(np.uint8))
    
    return mask_image

def download_image(url):
    response = requests.get(url)
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content)).convert("RGB")

def handler(event):
    global pipe_inpaint, pipe_t2i
    
    job_input = event["input"]
    
    image_url = job_input.get("image_url")
    prompt = job_input.get("prompt")
    negative_prompt = job_input.get("negative_prompt", "blurry, low quality, distortion")
    
    if not prompt:
        return {"error": "Missing prompt"}

    try:
        generator = None
        if "seed" in job_input:
             generator = torch.Generator(device="cpu").manual_seed(job_input["seed"])

        # SCENARIO 1: Inpainting (if image_url is provided)
        if image_url:
            mask_target = job_input.get("mask_target", "clothes, shirt, pants, dress")
            
            # Step 1: Download Image
            print("Downloading image (Mode: Inpainting)...")
            original_image = download_image(image_url)
            
            # Step 2: Smart Resize
            print("Resizing image...")
            processing_image = smart_resize(original_image)
            width, height = processing_image.size
            print(f"Processed size: {width}x{height}")
            
            # Step 3: Generate Mask
            print(f"Generating mask for target: {mask_target}")
            mask_image = get_mask(processing_image, mask_target)
            
            # Step 4: Run Inpainting
            print("Running Inpainting...")
            output_images = pipe_inpaint(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=processing_image,
                mask_image=mask_image,
                height=height,
                width=width,
                num_inference_steps=job_input.get("steps", 30),
                guidance_scale=job_input.get("guidance_scale", 7.5),
                strength=job_input.get("strength", 0.99),
                generator=generator
            ).images

        # SCENARIO 2: Text-to-Image (if no image_url)
        else:
            print("Mode: Text-to-Image...")
            # Default size 1024x1024 for SDXL
            width = job_input.get("width", 1024)
            height = job_input.get("height", 1024)
            
            output_images = pipe_t2i(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=job_input.get("steps", 30),
                guidance_scale=job_input.get("guidance_scale", 7.5),
                generator=generator
            ).images
        
        result_image = output_images[0]
        
        # Convert to Base64
        buffered = io.BytesIO()
        result_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        return img_str
        
    except Exception as e:
        print(f"Error: {e}")
        return {"error": str(e)}

# Initialize models once at startup
init_handler()

# Start the Serverless Worker
runpod.serverless.start({"handler": handler})
