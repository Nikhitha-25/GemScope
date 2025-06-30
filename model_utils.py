# model_utils.py

import torch
import torch.nn as nn
import torchvision.models as models
from skorch import NeuralNetClassifier
from skorch.callbacks import Freezer

class PretrainedModel(nn.Module):
    def __init__(self, output_features):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        num_ftrs = base.fc.in_features
        base.fc = nn.Linear(num_ftrs, output_features)
        self.model = base

    def forward(self, x):
        return self.model(x)

def load_model(checkpoint_path, output_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = NeuralNetClassifier(
        PretrainedModel,
        module__output_features=output_classes,
        criterion=nn.CrossEntropyLoss,
        optimizer=torch.optim.Adam,
        train_split=None,
        callbacks=[('freezer', Freezer(lambda name: not name.startswith('model.fc')))],
        device=device
    )
    net.initialize()
    net.load_params(f_params=checkpoint_path)
    return net
