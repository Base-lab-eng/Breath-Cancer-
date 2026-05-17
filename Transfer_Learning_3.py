from data_setup import setup
from utils_py import *
import torch
import torch.nn as nn
from torchvision import models


def build_model_3():
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

    for param in model.parameters():
        param.requires_grad = False

    for param in model.features[-3:].parameters():
        param.requires_grad = True

    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(1280, 2)
    )

    return model.to(device)