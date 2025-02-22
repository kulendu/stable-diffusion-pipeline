import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import torch
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel
from diffusers.utils import load_image
import cv2
import numpy as np
from PIL import Image
from controlnet_aux import OpenposeDetector
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

def initialize_pipeline():
    """
    Initialize inpainting pipeline with pose control
    """
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load ControlNet for pose preservation
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_openpose",
        torch_dtype=torch.float32
    )

    # Initialize inpainting pipeline
    pipeline = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        controlnet=controlnet,
        torch_dtype=torch.float32,
        safety_checker=None
    )

    if device == "mps":
        pipeline = pipeline.to(device)
        pipeline.enable_attention_slicing()
    
    return pipeline

def create_tshirt_mask(image_path):
    """
    Create a mask for the T-shirt area using segmentation
    """
    # Load segmentation model
    processor = AutoImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
    model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")
    
    # Load and process image
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt")
    
    # Get segmentation predictions
    outputs = model(**inputs)
    logits = outputs.logits.squeeze()
    
    # Get upper body clothing mask
    upper_body_class = 4  # Index for upper body clothing
    mask = (logits.argmax(0) == upper_body_class).float()
    mask = mask.numpy().astype(np.uint8) * 255
    
    # Process mask
    mask = cv2.dilate(mask, np.ones((5,5), np.uint8), iterations=2)
    mask = Image.fromarray(mask).convert('RGB')
    mask = mask.resize((512, 512))
    
    return mask

def get_pose(image_path):
    """
    Extract pose information from the image
    """
    pose_detector = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')
    image = load_image(image_path)
    pose = pose_detector(image)
    return pose

def analyze_tshirt(image_path):
    """
    Analyze target T-shirt characteristics
    """
    image = load_image(image_path)
    img_array = np.array(image)
    
    # Color analysis
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    h_mean = np.mean(hsv[:,:,0])
    s_mean = np.mean(hsv[:,:,1])
    v_mean = np.mean(hsv[:,:,2])
    
    # Determine color name
    color_name = "colored"
    if s_mean < 50:
        color_name = "white" if v_mean > 200 else "black" if v_mean < 50 else "gray"
    
    return color_name

def generate_tshirt_swap(pipeline, image_path, tshirt_path):
    """
    Generate image with swapped T-shirt
    """
    # Load and prepare images
    init_image = load_image(image_path).resize((512, 512))
    
    # Get pose and mask
    pose = get_pose(image_path)
    mask = create_tshirt_mask(image_path)
    
    # Analyze target T-shirt
    tshirt_color = analyze_tshirt(tshirt_path)
    
    # Create specific prompt for the T-shirt
    prompt = f"a person wearing a {tshirt_color} t-shirt, same person, same pose, highly detailed, professional photo"
    negative_prompt = "deformed, blurry, bad anatomy, different person, changing pose, multiple people"

    # Generate new image
    output = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=init_image,
        mask_image=mask,
        control_image=pose,
        num_inference_steps=20,
        guidance_scale=7.5,
        controlnet_conditioning_scale=0.8
    ).images[0]
    
    return output, mask, pose

def main():
    try:
        print("Initializing pipeline...")
        pipeline = initialize_pipeline()
        
        person_image_path = "image_2.jpg"
        tshirt_image_path = "image_1.jpg"
        
        print("Processing images and generating T-shirt swap...")
        result, mask, pose = generate_tshirt_swap(pipeline, person_image_path, tshirt_image_path)
        
        # Save results
        result.save("generated_image.png")
        mask.save("debug_mask.png")  # Save mask for debugging
        pose.save("debug_pose.png")  # Save pose for debugging
        
        print("Successfully saved results:")
        print("- Final image: output.png")
        print("- Debug mask: debug_mask.png")
        print("- Debug pose: debug_pose.png")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("\nTroubleshooting steps:")
        print("1. Install additional requirements:")
        print("   pip install transformers controlnet_aux")
        print("2. Verify input images exist and are readable")
        print("3. Check available memory (at least 8GB recommended)")

if __name__ == "__main__":
    main()