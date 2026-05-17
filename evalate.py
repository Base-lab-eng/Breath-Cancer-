from utils_py import *
from data_setup import setup
from model_CNN import CNN
import torch
from Transfer_Learning import build_model
from Transfer_Learning_2 import build_model_2
from Transfer_Learning_3 import build_model_3




# Load data
train_loader, val_loader, test_loader, train_labels = setup()


model_type='mobilenet_v2'  # Change to 'cnn' for the CNN model or 'mobilenet_v2' for the MobileNetV2 model



# Load model
if model_type == 'cnn':
    model = CNN(num_classes=2).to(device)
    load_checkpoint(f"checkpoints/best_model_{model_type}.pth", model)
elif model_type == 'resnet50':
    model = build_model()
    load_checkpoint(f"checkpoints/best_model_{model_type}.pth", model)

elif model_type == 'mobilenet_v2':
    model = build_model_2()
    load_checkpoint(f"checkpoints/best_model_{model_type}.pth", model)

elif model_type == 'EfficientNet_B0':
    model = build_model_3()
    load_checkpoint(f"checkpoints/best_model_{model_type}.pth", model)

    
    
    
    
    
    
model.eval()

# Inference
all_preds  = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

# Results
metrics = compute_metrics(all_labels, all_preds)
print(f"Accuracy  : {metrics['accuracy']:.4f}")
print(f"F1        : {metrics['f1']:.4f}")
print(f"Precision : {metrics['precision']:.4f}")
print(f"Recall    : {metrics['recall']:.4f}")

plot_confusion_matrix(all_labels, all_preds)
#plot_training_curves(train_losses, val_losses, train_accs, val_accs)
