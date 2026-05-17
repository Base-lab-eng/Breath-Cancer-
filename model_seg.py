# model_seg.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation
from utils_seg import device


# ══════════════════════════════════════════════════════════════════════════════
# 1. SEGFORMER
# ══════════════════════════════════════════════════════════════════════════════

class SegFormerSeg(nn.Module):
    """
    Wraps HuggingFace SegformerForSemanticSegmentation for binary segmentation.

    Freezing strategy
    -----------------
    - Entire backbone frozen        → protects pretrained features
    - Last 2 encoder blocks unfrozen → fine-tunes high-level representations
    - Full decode head unfrozen     → adapts to binary segmentation task

    Forward
    -------
    Input  : [B, 3, H, W]
    Output : logits [B, 1, H, W]  — raw, un-sigmoidised
    """

    def __init__(self, pretrained="nvidia/mit-b2", img_size=512):
        super().__init__()
        self.img_size = img_size

        self.model = SegformerForSemanticSegmentation.from_pretrained(
            pretrained,
            num_labels=2,
            ignore_mismatched_sizes=True,
        )

        # freeze entire backbone
        for param in self.model.segformer.parameters():
            param.requires_grad = False

        # unfreeze last 2 encoder blocks
        for block in self.model.segformer.encoder.block[-2:]:
            for param in block.parameters():
                param.requires_grad = True

        # unfreeze decode head
        for param in self.model.decode_head.parameters():
            param.requires_grad = True

    def forward(self, x):
        out    = self.model(pixel_values=x).logits   # [B, 2, H/4, W/4]
        binary = out[:, :1, :, :]                    # lesion channel only
        return F.interpolate(
            binary,
            size=(self.img_size, self.img_size),
            mode="bilinear",
            align_corners=False,
        )                                            # [B, 1, H, W]


def build_segformer(pretrained="nvidia/mit-b2", img_size=512):
    return SegFormerSeg(pretrained=pretrained, img_size=img_size).to(device)


# ══════════════════════════════════════════════════════════════════════════════
# 2. U-NET  (baseline)
# ══════════════════════════════════════════════════════════════════════════════

class DoubleConv(nn.Module):
    """Two consecutive Conv2d → BatchNorm → ReLU blocks."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    """
    Classic U-Net for binary segmentation.

    Architecture
    ------------
    Encoder    : 4 downsampling stages via MaxPool2d
    Bottleneck : DoubleConv at deepest level
    Decoder    : 4 upsampling stages via ConvTranspose2d + skip connections
    Head       : 1×1 Conv → single channel logits

    Forward
    -------
    Input  : [B, 3, H, W]
    Output : logits [B, 1, H, W]  — raw, un-sigmoidised
    """

    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()

        self.encoder = nn.ModuleList()
        self.pool    = nn.MaxPool2d(2, 2)
        self.upconv  = nn.ModuleList()
        self.decoder = nn.ModuleList()

        # encoder
        ch = in_channels
        for f in features:
            self.encoder.append(DoubleConv(ch, f))
            ch = f

        # bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # decoder
        for f in reversed(features):
            self.upconv.append(nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2))
            self.decoder.append(DoubleConv(f * 2, f))

        # output head
        self.head = nn.Conv2d(features[0], 1, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # encode
        for enc in self.encoder:
            x = enc(x)
            skip_connections.append(x)
            x = self.pool(x)

        # bottleneck
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # decode
        for i, (up, dec) in enumerate(zip(self.upconv, self.decoder)):
            x    = up(x)
            skip = skip_connections[i]
            # handle odd input sizes gracefully
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:],
                                  mode="bilinear", align_corners=False)
            x = dec(torch.cat([skip, x], dim=1))

        return self.head(x)   # [B, 1, H, W]


def build_unet(in_channels=3, features=[64, 128, 256, 512]):
    return UNet(in_channels=in_channels, features=features).to(device)