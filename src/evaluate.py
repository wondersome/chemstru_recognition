# Evaluate the trained model using standard metrics
import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report

def evaluate_model(data_dir, model_path='model.pth'):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)

    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, len(test_dataset.classes))
    model.load_state_dict(torch.load(model_path))
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.numpy())
            y_pred.extend(preds.cpu().numpy())

    print(classification_report(y_true, y_pred, target_names=test_dataset.classes))
