# train_seg.py

import os
import torch
from utils_seg import combined_loss, device, plot_training_curves_seg
from model_seg import build_segformer, build_unet
from engine_seg import fit_seg
from data_setup_seg import train_loader, val_loader



# ══════════════════════════════════════════════════════════════════════════════
# CONFIG

IMG_SIZE   = 512
BATCH_SIZE = 8
EPOCHS     = 30
LR         = 3e-5
MODEL_TYPE = "segformer"   # "segformer" | "unet"


# ══════════════════════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════════════════════

if MODEL_TYPE == "segformer":
    model = build_segformer(pretrained="nvidia/mit-b2", img_size=IMG_SIZE)
elif MODEL_TYPE == "unet":
    model = build_unet()
else:
    raise ValueError(f"Unknown model type: {MODEL_TYPE}")


# ══════════════════════════════════════════════════════════════════════════════
# OPTIMIZER & CRITERION
# ══════════════════════════════════════════════════════════════════════════════

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR,
    weight_decay=1e-4,
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS, eta_min=1e-7
)

criterion = combined_loss


# ══════════════════════════════════════════════════════════════════════════════
# TRAIN
# ══════════════════════════════════════════════════════════════════════════════

history = fit_seg(
    model,
    train_loader,
    val_loader,
    optimizer=optimizer,
    criterion=criterion,
    device=device,
    epochs=EPOCHS,
    model_name=MODEL_TYPE,
)

scheduler.step()


# ══════════════════════════════════════════════════════════════════════════════
# CURVES
# ══════════════════════════════════════════════════════════════════════════════

plot_training_curves_seg(history, model_name=MODEL_TYPE)