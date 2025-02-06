import cv2
import os
import torch
import matplotlib.pyplot as plt


IMG_PATH = "/Users/kulendu/Diffuers/generated_images/"
OUTPUT_PATH = "/Users/kulendu/Diffuers/processed_images/"

# getting all the filenames
def img_plt(filename):
    img_files = []
    for file in os.listdir(filename):
        if file.endswith((".png")):
            img_files.append(file)

    print(f"Images in {filename}: {img_files}")

    # visualing the images
    img = plt.imread(os.path.join(IMG_PATH, img_files[0]), cv2.IMREAD_UNCHANGED)
    height, width = img.shape[:2]
    print(f"Height: {height}, width: {width}")

    return img_files

# converting all the images from BGR to RGB colorscale
def bgr2rgb():
    img_files = img_plt(IMG_PATH)
    images_rgb = []
    for img in img_files:
        img = cv2.imread(os.path.join(IMG_PATH, img))
        if img is not None and len(img.shape) == 3:  
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images_rgb.append(img_rgb)
        else:
            images_rgb.append(img)  
    return images_rgb

# normalizing the images (torch tensors) and converting to grayscale
def img_norm():
    images_rgb = bgr2rgb()
    img_tensors = []
    for img in images_rgb:
        if img is not None:
            # converting the normalized image to grayscale
            image_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img_tensor = torch.tensor(image_gray, dtype=torch.float32) / 255.0  
            img_tensors.append(img_tensor)

    torch.save(img_tensors, "image_tensors.pth")
    print(f"Successfully saved the processed tensors!")
    return img_tensors

img_norm()
