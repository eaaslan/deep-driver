import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from data_loader import load_data  # Make sure this imports correctly
from model import DrowsinessDetectorCNN
import datetime
import os

def continue_training(model_path, data_path):
    # Setup the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load and prepare the model
    model = DrowsinessDetectorCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.train()

    # Setup loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 20

    # Load data
    train_loader, val_loader = load_data(data_path)

    # For storing metrics
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    # Early stopping and model saving setup
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0

    # Make sure the models directory exists
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        # Training loop
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        # Calculate training metrics
        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # Validation loop
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                v_loss = criterion(outputs, labels)
                val_loss += v_loss.item()
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        # Calculate validation metrics
        val_loss /= len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f'Epoch {epoch + 1}, Train Loss: {train_loss}, Train Accuracy: {train_accuracy}%, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}%')

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_filename = f'{models_dir}/drowsiness_detector_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.pth'
            torch.save(model.state_dict(), model_filename)
            print(f'Model improved and saved to {model_filename}')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered, best model saved at {model_filename}")
                break

    # Optionally, save the last state if early stopping wasn't triggered
    if patience_counter < patience:
        final_model_path = f'{models_dir}/drowsiness_detector_final_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.pth'
        torch.save(model.state_dict(), final_model_path)
        print(f'Training completed without early stopping, final model saved at {final_model_path}')

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Training & Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

if __name__ == "__main__":
    continue_training('drowsiness_detector.pth', 'dataset_new/train')
