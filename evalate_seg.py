
# evaluate_seg.py

from utils_seg import device, load_checkpoint, plot_predictions, plot_training_curves_seg, evaluate_seg
from engine_seg import *
from model_seg import build_segformer, build_unet
from data_setup_seg import test_loader, test_dataset
from train_seg import MODEL_TYPE, IMG_SIZE, EPOCHS, LR, history


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
# LOAD BEST CHECKPOINT
# ══════════════════════════════════════════════════════════════════════════════

load_checkpoint(f"checkpoints/best_seg_{MODEL_TYPE}.pth", model)


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATE
# ══════════════════════════════════════════════════════════════════════════════

metrics, preds, targets = evaluate_seg(model, test_loader, device)


# ══════════════════════════════════════════════════════════════════════════════
# VISUALISE
# ══════════════════════════════════════════════════════════════════════════════

plot_training_curves_seg(history, model_name=MODEL_TYPE)

plot_predictions(
    model,
    test_dataset,
    device,
    n=4,
    save_path=f"results/segmentation/{MODEL_TYPE}_predictions.png"
)