import torch
from data_loader import load_data
from model import DrowsinessDetectorCNN

def evaluate_model():
    _, test_loader = load_data()
    model = DrowsinessDetectorCNN()
    model.load_state_dict(torch.load('drowsiness_detector.pth'))
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy on test set: {100 * correct / total}%')

if __name__ == "__main__":
    evaluate_model()
