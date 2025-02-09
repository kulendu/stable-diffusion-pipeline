import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

data = np.load("/Users/kulendu/Diffuers/processed_image_array.npy")
print(f"Image values: {data}")
print(f"Min value: {data[0].min()} and Max value: {data[0].max()}")

