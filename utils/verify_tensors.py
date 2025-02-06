import torch
import cv2
import matplotlib.pyplot as plt

data = torch.load("/Users/kulendu/Diffuers/image_tensors.pth")
print(f"Image values: {data}")
print(f"Min value: {data[0].min()} and Max value: {data[0].max()}")

