import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import random

from model import DrowsinessDetectorCNN  # Adjust this if your import path differs

def load_model(model_path):
    model = DrowsinessDetectorCNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model
def process_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Adjust size according to your model's input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

def predict_image(model, image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        return predicted.item()

def process_directories(base_path, model, num_samples=5):
    classes = ['Closed', 'no_yawn', 'Open', 'yawn']  # Update this list based on your folders
    results = {}

    for class_name in classes:
        folder_path = os.path.join(base_path, class_name)
        images = [os.path.join(folder_path, img) for img in os.listdir(folder_path)]
        sampled_images = random.sample(images, min(num_samples, len(images)))  # Choose 5-10 images randomly

        results[class_name] = []
        for image_path in sampled_images:
            image_tensor = process_image(image_path)
            prediction = predict_image(model, image_tensor)
            results[class_name].append((image_path, prediction))

    return results

if __name__ == "__main__":
    model = load_model('models/drowsiness_detector_final_20240511_013958.pth')
    test_dir = 'dataset_new/test'  # Adjust this to your test directory
    results = process_directories(test_dir, model, num_samples=5)
    print(results)
