from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from utils import NUM_CLASSES, device


def build_unet_model(
    encoder_name: str = "resnet34",
    encoder_weights: Optional[str] = "imagenet",
    num_classes: int = NUM_CLASSES,
    target_device: str = device,
) -> nn.Module:
    """Builds a U-Net segmentation model and moves it to the target device."""
    import segmentation_models_pytorch as smp

    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=num_classes,
        activation=None,
    )
    return model.to(target_device)


def build_maskrcnn_model(
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    target_device: str = device,
) -> nn.Module:
    """Builds a Mask R-CNN model adapted to the project's class count."""
    import torchvision
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

    weights = "DEFAULT" if pretrained else None
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=weights)

    total_classes = num_classes + 1
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, total_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, total_classes)

    return model.to(target_device)


def get_yolo_checkpoint(
    size: str = "n",
    task: str = "seg",
    major_version: str = "11",
) -> str:
    """Returns the checkpoint filename for the requested YOLO variant."""
    checkpoint_families = {
        ("11", "detect"): {
            "n": "yolo11n.pt",
            "s": "yolo11s.pt",
            "m": "yolo11m.pt",
        },
        ("11", "seg"): {
            "n": "yolo11n-seg.pt",
            "s": "yolo11s-seg.pt",
            "m": "yolo11m-seg.pt",
        },
    }

    key = (str(major_version), task)
    if key not in checkpoint_families:
        raise ValueError(
            f"Unsupported YOLO family/task combination '{major_version}/{task}'. "
            "Supported combinations: 11/detect, 11/seg."
        )

    checkpoint_map = checkpoint_families[key]
    if size not in checkpoint_map:
        raise ValueError(f"Unsupported YOLO size '{size}'. Choose from {sorted(checkpoint_map)}.")
    return checkpoint_map[size]
