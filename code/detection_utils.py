from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from utils import (
    NUM_CLASSES,
    SEED,
    build_mask_from_record,
    build_torch_generator,
    dice_coefficient,
    iou_score,
    load_image_as_rgb,
    seed_worker,
)


def _extract_instances_from_multiclass_mask(mask_channels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Splits a multi-class mask into per-instance masks, labels, and boxes."""
    instance_masks: List[np.ndarray] = []
    labels: List[int] = []
    boxes: List[List[float]] = []

    for class_idx in range(mask_channels.shape[-1]):
        binary_mask = (mask_channels[..., class_idx] > 0).astype(np.uint8)
        if binary_mask.sum() == 0:
            continue

        num_components, component_map = cv2.connectedComponents(binary_mask)
        for component_id in range(1, num_components):
            instance_mask = (component_map == component_id).astype(np.uint8)
            if instance_mask.sum() == 0:
                continue

            ys, xs = np.where(instance_mask > 0)
            xmin, xmax = float(xs.min()), float(xs.max())
            ymin, ymax = float(ys.min()), float(ys.max())
            xmax = max(xmax, xmin + 1.0)
            ymax = max(ymax, ymin + 1.0)

            instance_masks.append(instance_mask)
            labels.append(class_idx + 1)
            boxes.append([xmin, ymin, xmax, ymax])

    if not instance_masks:
        return (
            np.zeros((0, mask_channels.shape[0], mask_channels.shape[1]), dtype=np.uint8),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0, 4), dtype=np.float32),
        )

    return (
        np.stack(instance_masks).astype(np.uint8),
        np.asarray(labels, dtype=np.int64),
        np.asarray(boxes, dtype=np.float32),
    )


def _resize_instance_targets(
    image: np.ndarray,
    masks: np.ndarray,
    target_size: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Resizes an image and its instance masks to a square target size."""
    if target_size is None:
        return image, masks

    resized_image = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    resized_masks = []
    for mask in masks:
        resized_mask = cv2.resize(mask, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
        resized_masks.append((resized_mask > 0).astype(np.uint8))

    if resized_masks:
        return resized_image, np.stack(resized_masks).astype(np.uint8)
    return resized_image, np.zeros((0, target_size, target_size), dtype=np.uint8)


def resize_multiclass_mask(
    mask_channels: np.ndarray,
    target_shape: Tuple[int, int],
) -> np.ndarray:
    """Resizes each class channel without mixing labels across classes."""
    target_height, target_width = target_shape
    if mask_channels.shape[:2] == (target_height, target_width):
        return mask_channels.astype(np.uint8)

    resized_channels: List[np.ndarray] = []
    for class_idx in range(mask_channels.shape[-1]):
        resized_mask = cv2.resize(
            mask_channels[..., class_idx].astype(np.uint8),
            (target_width, target_height),
            interpolation=cv2.INTER_NEAREST,
        )
        resized_channels.append((resized_mask > 0).astype(np.uint8))

    return np.stack(resized_channels, axis=-1).astype(np.uint8)


def _filter_empty_instance_masks(
    instance_masks: np.ndarray,
    labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Drops empty instance masks and keeps labels aligned with the survivors."""
    if len(instance_masks) == 0:
        return (
            np.zeros((0, instance_masks.shape[-2], instance_masks.shape[-1]), dtype=np.uint8),
            np.zeros((0,), dtype=np.int64),
        )

    kept_masks: List[np.ndarray] = []
    kept_labels: List[int] = []
    for mask, label in zip(instance_masks, labels):
        if int(mask.sum()) == 0:
            continue
        kept_masks.append(mask.astype(np.uint8))
        kept_labels.append(int(label))

    if not kept_masks:
        return (
            np.zeros((0, instance_masks.shape[-2], instance_masks.shape[-1]), dtype=np.uint8),
            np.zeros((0,), dtype=np.int64),
        )

    return np.stack(kept_masks).astype(np.uint8), np.asarray(kept_labels, dtype=np.int64)


class UnifiedInstanceDataset(Dataset):
    """Converts unified metadata rows into Mask R-CNN style samples."""

    def __init__(
        self,
        metadata_df,
        image_size: Optional[int] = None,
        return_metadata: bool = False,
    ) -> None:
        """Stores records and output options for instance segmentation."""
        self.records = metadata_df.to_dict("records")
        self.image_size = image_size
        self.return_metadata = return_metadata

    def __len__(self) -> int:
        """Returns the number of metadata records exposed by the dataset."""
        return len(self.records)

    def __getitem__(self, idx: int):
        """Builds one image tensor and one detection target dictionary from a record."""
        record = self.records[idx]
        image = load_image_as_rgb(record["image_path"])
        multiclass_mask = build_mask_from_record(record)
        instance_masks, labels, boxes = _extract_instances_from_multiclass_mask(multiclass_mask)
        image, instance_masks = _resize_instance_targets(image, instance_masks, target_size=self.image_size)
        instance_masks, labels = _filter_empty_instance_masks(instance_masks, labels)

        if len(instance_masks):
            resized_boxes = []
            for mask in instance_masks:
                ys, xs = np.where(mask > 0)
                xmin, xmax = float(xs.min()), float(xs.max())
                ymin, ymax = float(ys.min()), float(ys.max())
                xmax = max(xmax, xmin + 1.0)
                ymax = max(ymax, ymin + 1.0)
                resized_boxes.append([xmin, ymin, xmax, ymax])
            boxes = np.asarray(resized_boxes, dtype=np.float32)
        else:
            boxes = np.zeros((0, 4), dtype=np.float32)

        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "masks": torch.as_tensor(instance_masks, dtype=torch.uint8),
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "area": torch.as_tensor(
                [float(mask.sum()) for mask in instance_masks],
                dtype=torch.float32,
            ),
            "iscrowd": torch.zeros((len(labels),), dtype=torch.int64),
        }

        if self.return_metadata:
            return image_tensor, target, record
        return image_tensor, target


def detection_collate_fn(batch: Sequence[Any]) -> Any:
    """Batches variable-length detection targets while preserving optional metadata."""
    first = batch[0]
    if len(first) == 3:
        images, targets, metadata = zip(*batch)
        return list(images), list(targets), list(metadata)
    images, targets = zip(*batch)
    return list(images), list(targets)


def build_instance_dataloaders(
    train_df,
    val_df,
    image_size: Optional[int],
    batch_size: int,
    num_workers: int = 2,
    return_metadata: bool = False,
    persistent_workers: bool = True,
    prefetch_factor: Optional[int] = 2,
) -> Tuple[UnifiedInstanceDataset, UnifiedInstanceDataset, DataLoader, DataLoader]:
    """Builds train and validation dataloaders for instance segmentation models."""
    train_dataset = UnifiedInstanceDataset(train_df, image_size=image_size, return_metadata=return_metadata)
    val_dataset = UnifiedInstanceDataset(val_df, image_size=image_size, return_metadata=return_metadata)

    generator = build_torch_generator(SEED)
    loader_kwargs: Dict[str, Any] = {}
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        if prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = prefetch_factor

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=detection_collate_fn,
        worker_init_fn=seed_worker,
        generator=generator,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=detection_collate_fn,
        worker_init_fn=seed_worker,
        **loader_kwargs,
    )
    return train_dataset, val_dataset, train_loader, val_loader


def run_maskrcnn_training_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    grad_accumulation_steps: int = 1,
    max_grad_norm: Optional[float] = None,
    show_progress: bool = True,
    progress_desc: str = "maskrcnn_train",
) -> Dict[str, float]:
    """Runs one Mask R-CNN training epoch with optional gradient accumulation."""
    model.train()
    total_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    progress_bar = None
    batch_iterator = enumerate(dataloader)
    if show_progress and tqdm is not None:
        progress_bar = tqdm(batch_iterator, total=len(dataloader), desc=progress_desc, leave=False)
        batch_iterator = progress_bar

    for batch_idx, (images, targets) in batch_iterator:
        images = [image.to(device) for image in images]
        targets = [{key: value.to(device) for key, value in target.items()} for target in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        scaled_loss = losses / grad_accumulation_steps
        scaled_loss.backward()

        should_step = ((batch_idx + 1) % grad_accumulation_steps == 0) or ((batch_idx + 1) == len(dataloader))
        if should_step:
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += losses.item()
        if progress_bar is not None:
            progress_bar.set_postfix(loss=f"{total_loss / (batch_idx + 1):.4f}")

    return {"loss": total_loss / max(1, len(dataloader))}


@torch.no_grad()
def run_maskrcnn_validation_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str,
    show_progress: bool = True,
    progress_desc: str = "maskrcnn_val",
) -> Dict[str, float]:
    """Runs a lightweight validation pass and summarizes predicted instance counts."""
    model.eval()
    total_pred_masks = 0
    total_images = 0

    progress_bar = None
    batch_iterator = dataloader
    if show_progress and tqdm is not None:
        progress_bar = tqdm(dataloader, total=len(dataloader), desc=progress_desc, leave=False)
        batch_iterator = progress_bar

    for images, targets in batch_iterator:
        images = [image.to(device) for image in images]
        outputs = model(images)
        total_images += len(outputs)
        total_pred_masks += sum(int((output["scores"] > 0.5).sum().item()) for output in outputs)

        if progress_bar is not None:
            progress_bar.set_postfix(avg_pred=f"{total_pred_masks / max(1, total_images):.2f}")

    return {
        "avg_pred_instances": total_pred_masks / max(1, total_images),
        "images": total_images,
    }


def detection_output_to_multiclass_mask(
    output: Dict[str, torch.Tensor],
    image_shape: Tuple[int, int],
    score_threshold: float = 0.5,
    num_classes: int = NUM_CLASSES,
) -> np.ndarray:
    """Merges Mask R-CNN outputs into one multi-channel segmentation mask."""
    height, width = image_shape
    pred_mask = np.zeros((height, width, num_classes), dtype=np.uint8)

    if "scores" not in output or "masks" not in output or "labels" not in output:
        return pred_mask

    scores = output["scores"].detach().cpu().numpy()
    masks = output["masks"].detach().cpu().numpy()
    labels = output["labels"].detach().cpu().numpy()

    for score, mask, label in zip(scores, masks, labels):
        if score < score_threshold:
            continue
        class_idx = int(label) - 1
        if class_idx < 0 or class_idx >= num_classes:
            continue
        pred_mask[..., class_idx] = np.maximum(pred_mask[..., class_idx], (mask[0] > 0.5).astype(np.uint8))

    return pred_mask


@torch.no_grad()
def evaluate_maskrcnn_segmentation(
    model: torch.nn.Module,
    dataset: UnifiedInstanceDataset,
    device: str,
    score_threshold: float = 0.5,
    max_samples: Optional[int] = None,
    show_progress: bool = True,
    progress_desc: str = "maskrcnn_seg_eval",
) -> Dict[str, float]:
    """Evaluates Mask R-CNN predictions with Dice and IoU against true masks."""
    was_training = model.training
    model.eval()

    indices = range(len(dataset)) if max_samples is None else range(min(max_samples, len(dataset)))
    dice_scores: List[float] = []
    iou_scores: List[float] = []

    progress_bar = None
    index_iterator = indices
    if show_progress and tqdm is not None:
        total = len(dataset) if max_samples is None else min(max_samples, len(dataset))
        progress_bar = tqdm(indices, total=total, desc=progress_desc, leave=False)
        index_iterator = progress_bar

    for idx in index_iterator:
        sample = dataset[idx]
        if dataset.return_metadata:
            image_tensor, _, record = sample
        else:
            image_tensor, _ = sample
            record = dataset.records[idx]
        output = model([image_tensor.to(device)])[0]
        eval_shape = tuple(int(dim) for dim in image_tensor.shape[-2:])
        true_mask = resize_multiclass_mask(build_mask_from_record(record), eval_shape)
        pred_mask = detection_output_to_multiclass_mask(output, image_shape=true_mask.shape[:2], score_threshold=score_threshold)

        pred_tensor = torch.from_numpy(pred_mask).permute(2, 0, 1).unsqueeze(0).float()
        true_tensor = torch.from_numpy(true_mask).permute(2, 0, 1).unsqueeze(0).float()
        dice_scores.append(dice_coefficient(pred_tensor, true_tensor).item())
        iou_scores.append(iou_score(pred_tensor, true_tensor).item())

        if progress_bar is not None:
            progress_bar.set_postfix(
                dice=f"{float(np.mean(dice_scores)):.4f}",
                iou=f"{float(np.mean(iou_scores)):.4f}",
            )

    if was_training:
        model.train()

    return {
        "dice": float(np.mean(dice_scores)) if dice_scores else 0.0,
        "iou": float(np.mean(iou_scores)) if iou_scores else 0.0,
        "samples": len(dice_scores),
    }


def masks_to_yolo_segments(mask_channels: np.ndarray, min_points: int = 6) -> List[Tuple[int, List[Tuple[float, float]]]]:
    """Converts class masks into normalized polygon segments for YOLO labels."""
    segments: List[Tuple[int, List[Tuple[float, float]]]] = []
    height, width = mask_channels.shape[:2]

    for class_idx in range(mask_channels.shape[-1]):
        binary_mask = (mask_channels[..., class_idx] > 0).astype(np.uint8)
        if binary_mask.sum() == 0:
            continue

        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            contour = contour.squeeze(1)
            if contour.ndim != 2 or contour.shape[0] < 3:
                continue
            if contour.shape[0] * 2 < min_points:
                continue

            normalized_points = [
                (float(x) / width, float(y) / height)
                for x, y in contour
            ]
            segments.append((class_idx, normalized_points))

    return segments


def export_yolo_segmentation_dataset(
    train_df,
    val_df,
    export_root: str | Path,
    class_names: Optional[Sequence[str]] = None,
    overwrite: bool = True,
) -> Dict[str, str]:
    """Exports train and validation splits in YOLO segmentation format."""
    export_root = Path(export_root)
    if overwrite and export_root.exists():
        shutil.rmtree(export_root)

    image_roots = {
        "train": export_root / "images" / "train",
        "val": export_root / "images" / "val",
    }
    label_roots = {
        "train": export_root / "labels" / "train",
        "val": export_root / "labels" / "val",
    }

    for path in [*image_roots.values(), *label_roots.values()]:
        path.mkdir(parents=True, exist_ok=True)

    class_names = list(class_names) if class_names else [f"class_{idx}" for idx in range(1, NUM_CLASSES + 1)]

    def _export_split(dataframe, split_name: str) -> None:
        """Writes one dataframe split as YOLO images plus polygon label files."""
        for record in dataframe.to_dict("records"):
            image = load_image_as_rgb(record["image_path"])
            image_name = f"{record['source']}__{record['ImageId']}"
            image_target_path = image_roots[split_name] / image_name
            cv2.imwrite(str(image_target_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

            mask_channels = build_mask_from_record(record)
            segments = masks_to_yolo_segments(mask_channels)
            label_path = label_roots[split_name] / f"{Path(image_name).stem}.txt"

            lines = []
            for class_id, points in segments:
                flat_points = " ".join(f"{x:.6f} {y:.6f}" for x, y in points)
                lines.append(f"{class_id} {flat_points}")
            label_path.write_text("\n".join(lines), encoding="utf-8")

    _export_split(train_df, "train")
    _export_split(val_df, "val")

    yaml_path = export_root / "dataset.yaml"
    yaml_lines = [
        f"path: {export_root.as_posix()}",
        "train: images/train",
        "val: images/val",
        f"nc: {len(class_names)}",
        f"names: [{', '.join(repr(name) for name in class_names)}]",
    ]
    yaml_path.write_text("\n".join(yaml_lines), encoding="utf-8")

    return {
        "root": str(export_root),
        "yaml_path": str(yaml_path),
        "train_images": str(image_roots["train"]),
        "val_images": str(image_roots["val"]),
    }
