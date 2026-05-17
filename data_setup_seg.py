# data_setup_seg.py

from utils_seg import (
    load_data_seg,
    split_data_seg,
    get_transforms_seg,
    SegDataset,
    get_dataloaders_seg,
)

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD
# ══════════════════════════════════════════════════════════════════════════════

images, masks = load_data_seg()

# ══════════════════════════════════════════════════════════════════════════════
# 2. SPLIT
# ══════════════════════════════════════════════════════════════════════════════

(train_images, val_images, test_images,
 train_masks,  val_masks,  test_masks) = split_data_seg(images, masks)

# ══════════════════════════════════════════════════════════════════════════════
# 3. TRANSFORMS
# ══════════════════════════════════════════════════════════════════════════════

img_transform, mask_transform = get_transforms_seg(img_size=512)

# ══════════════════════════════════════════════════════════════════════════════
# 4. DATASETS
# ══════════════════════════════════════════════════════════════════════════════

train_dataset = SegDataset(
    train_images, train_masks,
    transform=img_transform,
    mask_transform=mask_transform,
    augment=True,
)

val_dataset = SegDataset(
    val_images, val_masks,
    transform=img_transform,
    mask_transform=mask_transform,
    augment=False,
)

test_dataset = SegDataset(
    test_images, test_masks,
    transform=img_transform,
    mask_transform=mask_transform,
    augment=False,
)

# ══════════════════════════════════════════════════════════════════════════════
# 5. DATALOADERS
# ══════════════════════════════════════════════════════════════════════════════

train_loader, val_loader, test_loader = get_dataloaders_seg(
    train_dataset, val_dataset, test_dataset,
    batch_size=8,
)

