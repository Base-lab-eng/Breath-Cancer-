import torch
import numpy as np
from utils_py import device, save_checkpoint
from data_setup import setup
from HF_Models import *
from engine import train_one_epoch, validate

model_type ='deit'  # 'vit' | 'swin' | 'deit'

if model_type == 'vit':
    model = build_Vit()
elif model_type == 'swin':
    model = build_swin()
elif model_type == 'deit':
    model = build_deit()


train_loader, val_loader, test_loader, train_labels = setup()


class_counts  = np.bincount(train_labels)
weights       = class_counts.max() / class_counts
class_weights = torch.tensor(weights, dtype=torch.float).to(device)
criterion     = torch.nn.CrossEntropyLoss(weight=class_weights)


optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=2e-5,
    weight_decay=0.01
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=50,
    eta_min=1e-7
)



EPOCHS = 50
best_val_loss = float('inf')
train_losses, val_losses = [], []
train_accs,   val_accs   = [], []

for epoch in range(EPOCHS):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss,   val_acc   = validate(model, val_loader, criterion, device)
    scheduler.step()

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_checkpoint(model, optimizer, epoch,
                        f"checkpoints/best_model_{model_type}.pth")

    print(f"Epoch {epoch+1:02d}/{EPOCHS} | "
          f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}")