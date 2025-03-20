import torch
from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt

# Load the trained model
model_path = 'runs/train/exp_20250311_med_250/model.pt'
model = YOLO(model_path)

# Load the test image
image_path = 'test_img/test1.jpg'
image = Image.open(image_path).convert('RGB')

# Run the model on the test image
results = model(image)

# Display the results
for result in results:
    result.show()

# Optionally, save the results
for i, result in enumerate(results):
    result.save(f'test_img/test1_result_{i}.jpg')