import torch
import numpy as np
from utils_py import device, load_checkpoint, compute_metrics, CLASSES
from data_setup import setup
from HF_Models import build_Vit, build_swin, build_deit


train_loader, val_loader, test_loader, train_labels = setup()

model_type = 'vit'  # 'vit' | 'swin' | 'deit'

if model_type == 'vit':
    model = build_Vit()
elif model_type == 'swin':
    model = build_swin()
elif model_type == 'deit':
    model = build_deit()

load_checkpoint(f"checkpoints/best_model_{model_type}.pth", model)

model.eval()

all_preds  = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())
model.eval()

all_preds  = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())
        

metrics = compute_metrics(all_labels, all_preds)
print(f"\nResults — {model_type.upper()}")
print(f"Accuracy  : {metrics['accuracy']:.4f}")
print(f"F1        : {metrics['f1']:.4f}")
print(f"Precision : {metrics['precision']:.4f}")
print(f"Recall    : {metrics['recall']:.4f}")