import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from skorch import NeuralNetClassifier
from PIL import Image
import os

# ✅ Define the prediction function
def predict_gemstone_name(img_path):
    # === Step 1: Custom Model Definition ===
    class PretrainedModel(nn.Module):
        def __init__(self, num_classes=39):
            super(PretrainedModel, self).__init__()
            self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            in_features = self.resnet.fc.in_features
            self.resnet.fc = nn.Linear(in_features, num_classes)

        def forward(self, x):
            return self.resnet(x)

    # === Step 2: Transform ===
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # === Step 3: Load class names ===
    dataset_path = 'C:/Users/Lenovo/Desktop/Tarun/Gemscope__/dataset_path/train'


    train_data = datasets.ImageFolder(dataset_path)
    class_names = train_data.classes  # e.g. ['Amethyst', 'Aquamarine', ...]

    # === Step 4: Load model ===
    model_path = 'best_model.pt'

    best_resnet18 = NeuralNetClassifier(
        module=PretrainedModel,
        module__num_classes=39,
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )

    # === Step 5: Load model state dict ===
    checkpoint = torch.load(model_path, map_location='cpu')
    if 'model' in checkpoint:
        checkpoint = checkpoint['model']

    new_state_dict = {}
    for key, value in checkpoint.items():
        if key.startswith('model.'):
            new_key = 'resnet.' + key[len('model.'):]
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value

    best_resnet18.initialize()
    best_resnet18.module_.load_state_dict(new_state_dict)

    # === Step 6: Predict ===
    img = Image.open(img_path).convert('RGB')
    img = transform(img).unsqueeze(0)
    pred_idx = best_resnet18.predict(img)[0]
    return class_names[pred_idx]
