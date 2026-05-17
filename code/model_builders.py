from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from utils import NUM_CLASSES, device


def build_segformer_model(
    pretrained_name: str = "nvidia/segformer-b0-finetuned-ade-512-512",
    num_classes: int = NUM_CLASSES,
    target_device: str = device,
) -> nn.Module:
    """Builds a SegFormer segmentation model with the requested class count."""
    from transformers import SegformerForSemanticSegmentation

    model = SegformerForSemanticSegmentation.from_pretrained(
        pretrained_name,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    )
    return model.to(target_device)


class DINOv2Segmentation(nn.Module):
    """Adds a lightweight decoder on top of a DINOv2 backbone for segmentation."""

    def __init__(self, backbone: nn.Module, num_classes: int = NUM_CLASSES) -> None:
        """Stores the backbone and builds the upsampling decoder head."""
        super().__init__()
        self.backbone = backbone
        hidden_size = getattr(backbone.config, "hidden_size", 768)

        self.decode = nn.Sequential(
            nn.ConvTranspose2d(hidden_size, hidden_size // 2, 2, 2),
            nn.BatchNorm2d(hidden_size // 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_size // 2, hidden_size // 4, 2, 2),
            nn.BatchNorm2d(hidden_size // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_size // 4, hidden_size // 8, 2, 2),
            nn.BatchNorm2d(hidden_size // 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_size // 8, num_classes, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts patch features, decodes them, and upsamples logits to image size."""
        batch_size, _, height, width = x.shape
        outputs = self.backbone(x, output_hidden_states=True)
        features = outputs.last_hidden_state[:, 1:, :]

        patch_grid = int(np.sqrt(features.shape[1]))
        features = features.reshape(batch_size, patch_grid, patch_grid, -1).permute(0, 3, 1, 2)
        out = self.decode(features)
        return nn.functional.interpolate(out, size=(height, width), mode="bilinear", align_corners=False)


def build_dinov2_model(
    pretrained_name: str = "facebook/dinov2-base",
    num_classes: int = NUM_CLASSES,
    target_device: str = device,
) -> DINOv2Segmentation:
    """Builds the DINOv2-based segmentation model and moves it to device."""
    from transformers import AutoModel

    backbone = AutoModel.from_pretrained(pretrained_name)
    model = DINOv2Segmentation(backbone, num_classes=num_classes)
    return model.to(target_device)


class SAMDefectHead(nn.Module):
    """Wraps SAMs image encoder with a trainable segmentation head."""

    def __init__(
        self,
        sam_model: nn.Module,
        num_classes: int = NUM_CLASSES,
        freeze_image_encoder: bool = True,
    ) -> None:
        """Optionally freezes SAMs encoder and attaches the custom defect head."""
        super().__init__()
        self.sam = sam_model

        if freeze_image_encoder:
            for parameter in self.sam.image_encoder.parameters():
                parameter.requires_grad = False

        self.head = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, num_classes, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Runs SAMs image encoder and maps its features to class logits."""
        if any(parameter.requires_grad for parameter in self.sam.image_encoder.parameters()):
            features = self.sam.image_encoder(x)
        else:
            with torch.no_grad():
                features = self.sam.image_encoder(x)
        return self.head(features)


def build_sam_head_model(
    checkpoint_path: str,
    model_type: str = "vit_b",
    num_classes: int = NUM_CLASSES,
    freeze_image_encoder: bool = True,
    target_device: str = device,
) -> SAMDefectHead:
    """Loads a SAM checkpoint and attaches the specific segmentation head."""
    from segment_anything import sam_model_registry

    sam_base = sam_model_registry[model_type](checkpoint=checkpoint_path).to(target_device)
    model = SAMDefectHead(
        sam_model=sam_base,
        num_classes=num_classes,
        freeze_image_encoder=freeze_image_encoder,
    )
    return model.to(target_device)


def mask_to_prompt_boxes(mask_channels: np.ndarray, min_pixels: int = 32) -> List[Tuple[int, int, int, int]]:
    """Builds a coarse box prompt from the union of all positive mask channels."""
    boxes: List[Tuple[int, int, int, int]] = []
    binary_mask = (mask_channels.sum(axis=-1) > 0).astype(np.uint8)
    if binary_mask.sum() < min_pixels:
        return boxes

    ys, xs = np.where(binary_mask > 0)
    xmin, xmax = int(xs.min()), int(xs.max())
    ymin, ymax = int(ys.min()), int(ys.max())
    boxes.append((xmin, ymin, xmax, ymax))
    return boxes


def mask_to_prompt_points(mask_channels: np.ndarray, max_points: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """Samples positive point prompts from the union of all positive mask channels."""
    binary_mask = (mask_channels.sum(axis=-1) > 0).astype(np.uint8)
    ys, xs = np.where(binary_mask > 0)
    if len(xs) == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.int32)

    indices = np.linspace(0, len(xs) - 1, num=min(max_points, len(xs)), dtype=int)
    points = np.stack([xs[indices], ys[indices]], axis=1).astype(np.float32)
    labels = np.ones((points.shape[0],), dtype=np.int32)
    return points, labels


def extract_trainable_head_parameters(model: nn.Module) -> Sequence[nn.Parameter]:
    """Returns the parameters that should be optimized for head-only fine-tuning."""
    base_model = getattr(model, "_orig_mod", model)
    if hasattr(base_model, "head"):
        return base_model.head.parameters()
    return [parameter for parameter in base_model.parameters() if parameter.requires_grad]
