import cv2
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

IMG_PATH = "/Users/kulendu/Diffuers/generated_images/"
def preprocess(filename):
    # getting the files
    images = []
    for file in os.listdir(filename):
        if file.endswith((".png")):
            images.append(file)
    
    # converting from BGR to RGB
    images_rgb = []
    for img in images:
        img = cv2.imread(os.path.join(IMG_PATH, img))
        if img is not None and len(img.shape) == 3:  
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images_rgb.append(img_rgb)
        else:
            images_rgb.append(img)  

    height, width = 256, 256
    final_img_arr = []

    for img in images_rgb:
        # converting to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # resizing the image to 256 x 256
        image_greyscale = cv2.resize(img, (height, width))
        # normalizing the images - [0,1]
        image_normalized = image_greyscale.astype(np.float32) / 255.0
        final_img_arr.append(image_normalized)

    np.save("processed_image_array.npy", final_img_arr)
    


if __name__ == "__main__":
    preprocess(IMG_PATH)