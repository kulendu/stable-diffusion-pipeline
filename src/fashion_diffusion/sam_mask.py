import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

from transformers import SamModel, SamProcessor

sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
mask_generator = SamAutomaticMaskGenerator(sam)

image = cv2.imread("image_2.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

print("Masking in Process....")
masks = mask_generator.generate(image)

largest_mask = max(masks, key=lambda x: x['area'])['segmentation']

# Convert to PIL mask format
mask = Image.fromarray((largest_mask * 255).astype(np.uint8))

# Save the mask for debugging
mask.save("mask.jpg")