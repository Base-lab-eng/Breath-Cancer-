# engine_seg.py

import torch
from tqdm import tqdm
from utils_seg import (
    dice_score,
    iou_score,
    save_checkpoint,
)


# ══════════════════════════════════════════════════════════════════════════════
# 1. TRAIN ONE EPOCH
# ══════════════════════════════════════════════════════════════════════════════

def train_one_epoch_seg(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss, dice_sum, iou_sum = 0.0, 0.0, 0.0

    for images, masks in tqdm(train_loader, leave=False, desc="  train"):
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss   = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            preds      = torch.sigmoid(logits)
            total_loss += loss.item()
            dice_sum   += dice_score(preds, masks)
            iou_sum    += iou_score(preds, masks)

    n = len(train_loader)
    return total_loss / n, dice_sum / n, iou_sum / n


# ══════════════════════════════════════════════════════════════════════════════
# 2. VALIDATE
# ══════════════════════════════════════════════════════════════════════════════

def validate_seg(model, val_loader, criterion, device):
    model.eval()
    total_loss, dice_sum, iou_sum = 0.0, 0.0, 0.0

    with torch.no_grad():
        for images, masks in tqdm(val_loader, leave=False, desc="  val  "):
            images, masks = images.to(device), masks.to(device)

            logits = model(images)
            loss   = criterion(logits, masks)
            preds  = torch.sigmoid(logits)

            total_loss += loss.item()
            dice_sum   += dice_score(preds, masks)
            iou_sum    += iou_score(preds, masks)

    n = len(val_loader)
    return total_loss / n, dice_sum / n, iou_sum / n


# ══════════════════════════════════════════════════════════════════════════════
# 3. FIT
# ══════════════════════════════════════════════════════════════════════════════

def fit_seg(model, train_loader, val_loader, optimizer, criterion, device, epochs, model_name="model"):
    history = {
        "train_loss": [], "val_loss": [],
        "train_dice": [], "val_dice": [],
        "train_iou" : [], "val_iou" : [],
    }
    best_dice = 0.0

    for epoch in range(1, epochs + 1):
        train_loss, train_dice, train_iou = train_one_epoch_seg(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_dice, val_iou = validate_seg(
            model, val_loader, criterion, device
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_dice"].append(train_dice)
        history["val_dice"].append(val_dice)
        history["train_iou"].append(train_iou)
        history["val_iou"].append(val_iou)

        print(
            f"Epoch {epoch}/{epochs} "
            f"| Loss {train_loss:.4f}/{val_loss:.4f} "
            f"| Dice {train_dice:.4f}/{val_dice:.4f} "
            f"| IoU  {train_iou:.4f}/{val_iou:.4f}"
        )

        if val_dice > best_dice:
            best_dice = val_dice
            save_checkpoint(model, optimizer, epoch,
                            f"checkpoints/best_seg_{model_name}.pth")
            print(f"  ✓ New best Dice: {best_dice:.4f} — checkpoint saved")

    return history