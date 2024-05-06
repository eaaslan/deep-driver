import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import load_data
from model import DrowsinessDetectorCNN

def train_model():
    train_loader, _ = load_data()
    model = DrowsinessDetectorCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 10

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}')

if __name__ == "__main__":
    train_model()
