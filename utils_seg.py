# utils_seg.py

import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import directed_hausdorff


# ══════════════════════════════════════════════════════════════════════════════
# DEVICE
# ══════════════════════════════════════════════════════════════════════════════

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("checkpoints",          exist_ok=True)
os.makedirs("results/segmentation", exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

DATA_PATH = r"breast-ultrasound-images-dataset/Dataset_BUSI_with_GT"


def load_data_seg(data_path=DATA_PATH, classes=("benign", "malignant", "normal")):
    images, masks = [], []

    for cls in classes:
        folder = os.path.join(data_path, cls)
        files  = sorted([f for f in os.listdir(folder) if "mask" not in f and f.endswith(".png")])

        for fname in files:
            stem      = os.path.splitext(fname)[0]
            mask_name = f"{stem}_mask.png"
            img_path  = os.path.join(folder, fname)
            mask_path = os.path.join(folder, mask_name)

            if os.path.exists(mask_path):
                images.append(img_path)
                masks.append(mask_path)
            else:
                print(f"[WARNING] No mask found for {fname}, skipping.")

    print(f"Loaded {len(images)} image-mask pairs.")
    return images, masks


# ══════════════════════════════════════════════════════════════════════════════
# 2. SPLIT
# ══════════════════════════════════════════════════════════════════════════════

def split_data_seg(images, masks, val_size=0.15, test_size=0.15, random_state=42):
    train_imgs, temp_imgs, train_msks, temp_msks = train_test_split(
        images, masks,
        test_size=val_size + test_size,
        random_state=random_state,
    )

    val_imgs, test_imgs, val_msks, test_msks = train_test_split(
        temp_imgs, temp_msks,
        test_size=0.5,
        random_state=random_state,
    )

    print(f"Split → Train: {len(train_imgs)} | Val: {len(val_imgs)} | Test: {len(test_imgs)}")
    return train_imgs, val_imgs, test_imgs, train_msks, val_msks, test_msks


# ══════════════════════════════════════════════════════════════════════════════
# 3. TRANSFORMS
# ══════════════════════════════════════════════════════════════════════════════

def get_transforms_seg(img_size=512):
    img_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    mask_transform = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])

    return img_transform, mask_transform


# ══════════════════════════════════════════════════════════════════════════════
# 4. DATASET
# ══════════════════════════════════════════════════════════════════════════════

class SegDataset(Dataset):
    def __init__(self, images, masks, transform=None, mask_transform=None, augment=False):
        self.images         = images
        self.masks          = masks
        self.transform      = transform
        self.mask_transform = mask_transform
        self.augment        = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        mask  = Image.open(self.masks[idx]).convert("L")

        if self.augment:
            if torch.rand(1).item() > 0.5:
                image = transforms.functional.hflip(image)
                mask  = transforms.functional.hflip(mask)

            if torch.rand(1).item() > 0.5:
                image = transforms.functional.vflip(image)
                mask  = transforms.functional.vflip(mask)

            angle = (torch.rand(1).item() * 40) - 20
            image = transforms.functional.rotate(image, angle)
            mask  = transforms.functional.rotate(mask,  angle)

        if self.transform:
            image = self.transform(image)

        if self.mask_transform:
            mask = self.mask_transform(mask)

        mask = (mask > 0.5).float()

        return image, mask


# ══════════════════════════════════════════════════════════════════════════════
# 5. DATALOADERS
# ══════════════════════════════════════════════════════════════════════════════

def get_dataloaders_seg(train_dataset, val_dataset, test_dataset,
                        batch_size=8, num_workers=0):
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader


# ══════════════════════════════════════════════════════════════════════════════
# 6. LOSS FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def dice_loss(logits, targets, smooth=1e-6):
    probs = torch.sigmoid(logits)
    inter = (probs * targets).sum(dim=(2, 3))
    denom = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
    return (1 - (2 * inter + smooth) / (denom + smooth)).mean()


def focal_loss(logits, targets, alpha=0.8, gamma=2.0):
    bce   = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probs = torch.sigmoid(logits)
    pt    = torch.where(targets == 1, probs, 1 - probs)
    return (alpha * (1 - pt) ** gamma * bce).mean()


def combined_loss(logits, targets, dice_w=0.5, focal_w=0.5):
    return dice_w * dice_loss(logits, targets) + \
           focal_w * focal_loss(logits, targets)


# ══════════════════════════════════════════════════════════════════════════════
# 7. METRICS & EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def dice_score(preds, targets, threshold=0.5, smooth=1e-6):
    preds = (preds > threshold).float()
    inter = (preds * targets).sum(dim=(2, 3))
    denom = preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
    return ((2 * inter + smooth) / (denom + smooth)).mean().item()


def iou_score(preds, targets, threshold=0.5, smooth=1e-6):
    preds = (preds > threshold).float()
    inter = (preds * targets).sum(dim=(2, 3))
    union = preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) - inter
    return ((inter + smooth) / (union + smooth)).mean().item()


def _hausdorff(pred_np, target_np):
    pred_pts   = np.argwhere(pred_np   > 0.5)
    target_pts = np.argwhere(target_np > 0.5)

    if len(pred_pts) == 0 or len(target_pts) == 0:
        return 0.0

    d1 = directed_hausdorff(pred_pts,   target_pts)[0]
    d2 = directed_hausdorff(target_pts, pred_pts  )[0]
    return max(d1, d2)


def compute_metrics_seg(preds, targets):
    dice_list, iou_list, acc_list, haus_list = [], [], [], []

    for p, t in zip(preds, targets):
        p_flat = p.flatten()
        t_flat = t.flatten()

        inter = (p_flat * t_flat).sum()
        denom = p_flat.sum() + t_flat.sum()
        union = denom - inter

        dice_list.append((2 * inter + 1e-6) / (denom + 1e-6))
        iou_list.append((inter + 1e-6) / (union + 1e-6))
        acc_list.append((p_flat == t_flat).mean())
        haus_list.append(_hausdorff(p, t))

    return {
        "dice"          : float(np.mean(dice_list)),
        "iou"           : float(np.mean(iou_list)),
        "pixel_accuracy": float(np.mean(acc_list)),
        "hausdorff"     : float(np.mean(haus_list)),
    }


def evaluate_seg(model, test_loader, device):
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            logits = model(images)
            preds  = (torch.sigmoid(logits) > 0.5).float().cpu().numpy()
            tgts   = masks.numpy()

            for p, t in zip(preds, tgts):
                all_preds.append(p[0])
                all_targets.append(t[0])

    metrics = compute_metrics_seg(all_preds, all_targets)

    print("\n─── Segmentation Test Results ──────────────────────────────")
    for k, v in metrics.items():
        print(f"  {k:<18}: {v:.4f}")
    print("────────────────────────────────────────────────────────────\n")

    return metrics, all_preds, all_targets


# ══════════════════════════════════════════════════════════════════════════════
# 8. CHECKPOINT
# ══════════════════════════════════════════════════════════════════════════════

def save_checkpoint(model, optimizer, epoch, path):
    torch.save({
        "epoch"               : epoch,
        "model_state_dict"    : model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, path)
    print(f"Checkpoint saved → {path}")


def load_checkpoint(path, model, optimizer=None):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print(f"Checkpoint loaded ← {path}  (epoch {checkpoint['epoch']})")
    return checkpoint["epoch"]


# ══════════════════════════════════════════════════════════════════════════════
# 9. VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

def plot_training_curves_seg(history, model_name="model"):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    for ax, metric, title in zip(
        axes,
        [("train_loss", "val_loss"), ("train_dice", "val_dice"), ("train_iou", "val_iou")],
        ["Loss", "Dice", "IoU"],
    ):
        ax.plot(epochs, history[metric[0]], label="Train")
        ax.plot(epochs, history[metric[1]], label="Val")
        ax.set_title(f"{model_name} — {title}")
        ax.set_xlabel("Epoch")
        ax.legend()

    plt.tight_layout()
    plt.savefig(f"results/segmentation/{model_name}_curves.png", dpi=150)
    plt.show()
    print(f"Curves saved → results/segmentation/{model_name}_curves.png")


def plot_predictions(model, dataset, device, n=4, save_path=None):
    model.eval()
    indices = np.random.choice(len(dataset), n, replace=False)

    fig, axes = plt.subplots(n, 3, figsize=(10, n * 3))
    axes[0, 0].set_title("Image",     fontsize=11)
    axes[0, 1].set_title("True Mask", fontsize=11)
    axes[0, 2].set_title("Pred Mask", fontsize=11)

    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    with torch.no_grad():
        for row, idx in enumerate(indices):
            image, mask = dataset[idx]

            logit = model(image.unsqueeze(0).to(device))
            pred  = (torch.sigmoid(logit) > 0.5).float().squeeze().cpu().numpy()

            img_np = image.numpy().transpose(1, 2, 0)
            img_np = np.clip(std * img_np + mean, 0, 1)

            axes[row, 0].imshow(img_np)
            axes[row, 1].imshow(mask.squeeze().numpy(), cmap="gray")
            axes[row, 2].imshow(pred, cmap="gray")

            for ax in axes[row]:
                ax.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Predictions saved → {save_path}")
    plt.show()