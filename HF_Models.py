import torch
import torch.nn as nn

from utils_py import device
from transformers import AutoModelForImageClassification



class HFWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(pixel_values=x).logits








def build_Vit():
    model=AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k",
                                                          num_labels=2,
                                                          ignore_mismatched_sizes=True)
    
    
    # freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # unfreeze last 2 transformer blocks
    for block in model.vit.encoder.layer[-2:]:
        for param in block.parameters():
            param.requires_grad = True

    # unfreeze the classification head
    for param in model.classifier.parameters():
        param.requires_grad = True

    return HFWrapper(model).to(device)


def build_swin():
    model = AutoModelForImageClassification.from_pretrained(
        "microsoft/swin-tiny-patch4-window7-224",
        num_labels=2,
        ignore_mismatched_sizes=True
    )
    for param in model.parameters():
        param.requires_grad = False

    # unfreeze last 2 blocks of the last stage
    for block in model.swin.encoder.layers[-1].blocks[-2:]:
        for param in block.parameters():
            param.requires_grad = True

    # unfreeze the classification head
    for param in model.classifier.parameters():
        param.requires_grad = True
    
    
    

    return HFWrapper(model).to(device)


def build_deit():
    model = AutoModelForImageClassification.from_pretrained(
        "facebook/deit-base-distilled-patch16-224",
        num_labels=2,
        ignore_mismatched_sizes=True
    )
    
    
    for param in model.parameters():
        param.requires_grad = False

    # unfreeze last 2 blocks of the last stage
    for block in model.deit.encoder.layer[-2:]:
        for param in block.parameters():
            param.requires_grad = True

    for param in model.cls_classifier.parameters():
        param.requires_grad = True

    for param in model.distillation_classifier.parameters():
        param.requires_grad = True

    return HFWrapper(model).to(device)

                                                            
                                                          
    

