from utils_py import *
from data_setup import *
from model_CNN import *
from engine import *
from Transfer_Learning import build_model




import torch
import torch.nn as nn


# Device

model = CNN(num_classes=2).to(device)
print(device)


train_loader, val_loader, test_loader ,train_labels=setup()

class_counts = np.bincount(train_labels)
weights = class_counts.max() / class_counts
class_weights = torch.tensor(weights, dtype=torch.float).to(device)



criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)


scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    patience=3,
    factor=0.5

)


# ── Main loop ─────────────────────────────────────────────────
EPOCHS = 50
train_losses, val_losses = [], []
train_accs, val_accs = [], []

best_val_loss = float('inf')


for epoch in range(EPOCHS):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc     = validate(model, val_loader, criterion, device)
    scheduler.step(val_loss)
    

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_checkpoint(model, optimizer, epoch, "checkpoints/best_model.pth")
        

    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")