from data_setup import setup
from utils_py import *
import torch
import torch.nn as nn
from torchvision import models



def build_model():
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    for param in model.parameters():
        param.requires_grad = False

    for param in model.layer4.parameters():
        param.requires_grad = True

    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, 2)
    )

    return model.to(device)
