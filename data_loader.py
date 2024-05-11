# data_loader.py

import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

def load_data(data_dir='./archive/Driver Drowsiness Dataset (DDD)'):
    transformations = transforms.Compose([
        transforms.Resize((128, 128)),  # Resize all images to 128x128
        transforms.ToTensor(),          # Convert images to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalization
        transforms.RandomHorizontalFlip(),  # Random horizontal flip
        transforms.RandomRotation(10)  # Random rotation by 10 degrees
    ])
    dataset = ImageFolder(data_dir, transform=transformations)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    return train_loader, test_loader
