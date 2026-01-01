import runpod
import torch
import requests
import base64
import io
import cv2
import numpy as np
from PIL import Image
from diffusers import StableDiffusionXLInpaintPipeline
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

# Global variables for caching models (Cold Start optimization)
pipe = None
processor = None
segmentator = None

def init_handler():
    global pipe, processor, segmentator
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing handler on {device}...")

    # Load ClipSeg
    print("Loading ClipSeg...")
    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    segmentator = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(device)

    # Load SDXL Inpainting from Custom Checkpoint
    checkpoint_path = "checkpoints/model.safetensors"
    print(f"Loading SDXL Pipeline from {checkpoint_path}...")
    
    pipe = StableDiffusionXLInpaintPipeline.from_single_file(
        checkpoint_path,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16"
    )
    
    # Optimizations
    if device == "cuda":
        pipe.enable_model_cpu_offload() # Saves VRAM by offloading unused submodules
        # pipe.enable_xformers_memory_efficient_attention() # Optional: if xformers is installed/available
    
    print("Initialization complete.")

def get_mask(image, text_prompts):
    """
    Generates a binary mask for the given image and text prompts using ClipSeg.
    """
    device = segmentator.device
    
    # Prepare input
    # text_prompts can be a list or a single string.
    # If it's a comma-separated string, split it.
    if isinstance(text_prompts, str):
        prompts = [p.strip() for p in text_prompts.split(",")]
    else:
        prompts = text_prompts
        
    inputs = processor(text=prompts, images=[image] * len(prompts), padding="max_length", return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = segmentator(**inputs)
        
    # Get prediction
    # If multiple prompts, we might want to sum/combine them or take the user intent. 
    # The prompt implies "clothes, shirt, dress" etc. might be passed.
    # We will aggregate the masks if multiple prompts are predicted.
    
    preds = outputs.logits.unsqueeze(1) # [B, 1, H, W]
    
    # Resize to original image size
    # ClipSeg output is usually small (352x352). We need to resize to image size.
    # Attention: torch.nn.functional.interpolate expects [N, C, H, W]
    original_size = image.size[::-1] # (H, W) -> (W, H) in PIL
    
    # Combined mask
    combined_mask = torch.sigmoid(preds[0][0])
    for i in range(1, len(prompts)):
        combined_mask = torch.max(combined_mask, torch.sigmoid(preds[i][0]))

    # Convert to numpy
    mask_np = combined_mask.cpu().numpy()
    
    # Resize mask to original image size
    mask_cv = cv2.resize(mask_np, image.size, interpolation=cv2.INTER_CUBIC)
    
    # Thresholding to binary mask
    # You might want to tune the threshold (0.4 - 0.5 usually works)
    _, binary_mask = cv2.threshold(mask_cv, 0.4, 255, cv2.THRESH_BINARY)
    
    # Convert back to PIL Image
    mask_image = Image.fromarray(binary_mask.astype(np.uint8))
    
    return mask_image

def download_image(url):
    response = requests.get(url)
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content)).convert("RGB")

def handler(event):
    global pipe
    
    job_input = event["input"]
    
    image_url = job_input.get("image_url")
    prompt = job_input.get("prompt")
    mask_target = job_input.get("mask_target", "clothes, shirt, pants, dress")
    negative_prompt = job_input.get("negative_prompt", "blurry, low quality, distortion")
    
    if not image_url or not prompt:
        return {"error": "Missing image_url or prompt"}

    try:
        # Step 1: Download Image
        print("Downloading image...")
        original_image = download_image(image_url)
        
        # Step 2: Generate Mask
        print(f"Generating mask for target: {mask_target}")
        mask_image = get_mask(original_image, mask_target)
        
        # Step 3: SDXL Inpainting
        print("Running Inpainting...")
        # seed handling if needed
        generator = None
        if "seed" in job_input:
             generator = torch.Generator(device="cpu").manual_seed(job_input["seed"])

        output_images = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=original_image,
            mask_image=mask_image,
            num_inference_steps=job_input.get("steps", 30),
            guidance_scale=job_input.get("guidance_scale", 7.5),
            strength=job_input.get("strength", 0.99), # High strength to fully replace masked area
            generator=generator
        ).images
        
        result_image = output_images[0]
        
        # Convert to Base64
        buffered = io.BytesIO()
        result_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        return {"image_base64": img_str}
        
    except Exception as e:
        print(f"Error: {e}")
        return {"error": str(e)}

# Initialize models once at startup
init_handler()

# Start the Serverless Worker
runpod.serverless.start({"handler": handler})
