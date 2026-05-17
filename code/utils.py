from __future__ import annotations

import json
import os
import random
import xml.etree.ElementTree as ET
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

PROJECT_DIR = Path(__file__).resolve().parent
SEED = 1335
NUM_CLASSES = 4
CLASS_COLUMNS = [f"Class_{idx}" for idx in range(1, NUM_CLASSES + 1)]
CLASS_PRESENCE_COLUMNS = [f"class_{idx}_present" for idx in range(1, NUM_CLASSES + 1)]

NEU_TO_SEVERSTAL_CLASS = {
    "crazing": 1,
    "inclusion": 2,
    "patches": 3,
    "pitted_surface": 4,
}


KOLEKTOR_POSITIVE_CLASS = 4

device = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = True
DEFAULT_AMP_MODE = "fp16"


def _resolve_dataset_root(env_var: str, candidates: Sequence[Path | str]) -> Optional[Path]:
    """Resolves a dataset root from an environment override or known fallback paths."""
    override = os.getenv(env_var)
    if override:
        override_path = Path(override).expanduser()
        if override_path.exists():
            return override_path

    for candidate in candidates:
        candidate_path = Path(candidate).expanduser()
        if candidate_path.exists():
            return candidate_path

    return None


SEVERSTAL_ROOT = _resolve_dataset_root(
    "SEVERSTAL_DATA_ROOT",
    [
        PROJECT_DIR.parent / "severstal-steel-defect-detection",
        PROJECT_DIR / "severstal-steel-defect-detection",
        "/kaggle/input/competitions/severstal-steel-defect-detection",
        "/kaggle/input/severstal-steel-defect-detection",
    ],
)
NEU_ROOT = _resolve_dataset_root(
    "NEU_DATA_ROOT",
    [
        PROJECT_DIR.parent / "NEU-DET_data",
        PROJECT_DIR / "NEU-DET_data",
        "/kaggle/input/neu-det-data/NEU-DET_data",
        "/kaggle/input/neu-det-data",
    ],
)
KOLEKTOR_ROOT = _resolve_dataset_root(
    "KOLEKTOR_DATA_ROOT",
    [
        PROJECT_DIR.parent / "kolektor_data",
        PROJECT_DIR / "kolektor_data",
        "/kaggle/input/kolektor-data/kolektor_data",
        "/kaggle/input/kolektor-data",
    ],
)

TRAIN_DIR = str(SEVERSTAL_ROOT / "train_images") if SEVERSTAL_ROOT else ""
TRAIN_CSV = str(SEVERSTAL_ROOT / "train.csv") if SEVERSTAL_ROOT else ""
TEST_DIR = str(SEVERSTAL_ROOT / "test_images") if SEVERSTAL_ROOT else ""


def seed_everything(seed: int = SEED) -> None:
    """Seeds Python, NumPy, and PyTorch for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:
    """Seeds one dataloader worker deterministically from the global seed."""
    worker_seed = SEED + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def build_torch_generator(seed: int = SEED) -> torch.Generator:
    """Builds a seeded PyTorch generator for deterministic sampling."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def build_optimization_profile(
    amp_mode: str = DEFAULT_AMP_MODE,
    channels_last: bool = False,
    compile_model: bool = False,
    compile_mode: str = "default",
    gradient_checkpointing: bool = False,
    grad_accumulation_steps: int = 1,
    max_grad_norm: Optional[float] = None,
    ema_decay: Optional[float] = None,
    persistent_workers: bool = True,
    prefetch_factor: Optional[int] = 2,
) -> Dict[str, Any]:
    """Packages runtime and training optimization flags into one dictionary."""
    return {
        "amp_mode": amp_mode,
        "channels_last": channels_last,
        "compile_model": compile_model,
        "compile_mode": compile_mode,
        "gradient_checkpointing": gradient_checkpointing,
        "grad_accumulation_steps": grad_accumulation_steps,
        "max_grad_norm": max_grad_norm,
        "ema_decay": ema_decay,
        "persistent_workers": persistent_workers,
        "prefetch_factor": prefetch_factor,
    }


def _ensure_path(path_like: str | Path, description: str) -> Path:
    """Validates that a required dataset path exists before it is used."""
    path = Path(path_like)
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {description} at '{path}'. "
            "Set the appropriate *_DATA_ROOT environment variable when running on Kaggle."
        )
    return path


def _json_loads(value: Any) -> List[Dict[str, Any]]:
    """Parses annotation JSON fields while treating empty values as no annotations."""
    if isinstance(value, list):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        return json.loads(value)
    return []


def _list_image_files(directory: Path) -> List[Path]:
    """Lists image files from a directory using known image extensions."""
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in suffixes)


def _build_presence_columns(class_ids: Iterable[int]) -> Dict[str, bool]:
    """Builds one boolean presence flag for each defect class."""
    class_id_set = {int(class_id) for class_id in class_ids}
    return {
        f"class_{class_idx}_present": class_idx in class_id_set
        for class_idx in range(1, NUM_CLASSES + 1)
    }


def _primary_class_from_ids(class_ids: Sequence[int]) -> int:
    """Returns the first class id from an annotation list, or 0 if it is empty."""
    return int(class_ids[0]) if class_ids else 0


def _metadata_cache_path(filename: str = "merged_metadata.csv") -> Path:
    """Chooses the cache location used for merged metadata tables."""
    override = os.getenv("MERGED_METADATA_CACHE_PATH")
    if override:
        cache_path = Path(override).expanduser()
    elif Path("/kaggle/working").exists():
        cache_path = Path("/kaggle/working") / filename
    else:
        cache_path = PROJECT_DIR / filename

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    return cache_path


def load_image_as_rgb(image_path: str | Path) -> np.ndarray:
    """Loads a grayscale image and expands it to three RGB-like channels."""
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image '{image_path}'")
    return np.repeat(image[..., None], 3, axis=2)


def get_dataset_roots() -> Dict[str, Optional[str]]:
    """Returns the resolved roots for the supported datasets."""
    return {
        "severstal": str(SEVERSTAL_ROOT) if SEVERSTAL_ROOT else None,
        "neu": str(NEU_ROOT) if NEU_ROOT else None,
        "kolektor": str(KOLEKTOR_ROOT) if KOLEKTOR_ROOT else None,
    }


def build_severstal_metadata(
    csv_path: str | Path = TRAIN_CSV,
    image_dir: str | Path = TRAIN_DIR,
) -> pd.DataFrame:
    """Builds unified metadata rows for the Severstal dataset."""
    csv_path = _ensure_path(csv_path, "Severstal train.csv")
    image_dir = _ensure_path(image_dir, "Severstal train_images")

    df_raw = pd.read_csv(csv_path)
    df_pivot = df_raw.pivot(index="ImageId", columns="ClassId", values="EncodedPixels")
    df_pivot.columns = [f"Class_{idx}" for idx in df_pivot.columns]
    df_pivot = df_pivot.reset_index()

    all_images = pd.DataFrame({"ImageId": [path.name for path in _list_image_files(image_dir)]})
    metadata = all_images.merge(df_pivot, on="ImageId", how="left")

    for column in CLASS_COLUMNS:
        if column not in metadata.columns:
            metadata[column] = np.nan

    metadata["source"] = "severstal"
    metadata["source_split"] = "official_train"
    metadata["record_id"] = metadata["ImageId"].map(lambda image_id: f"severstal::{image_id}")
    metadata["image_path"] = metadata["ImageId"].map(lambda image_id: str(image_dir / image_id))
    metadata["annotation_format"] = "rle"
    metadata["supervision_level"] = "strong"
    metadata["height"] = 256
    metadata["width"] = 1600

    metadata["has_defect"] = metadata[CLASS_COLUMNS].notna().any(axis=1)
    metadata["num_defects"] = metadata[CLASS_COLUMNS].notna().sum(axis=1).astype(int)
    metadata["primary_class_id"] = (
        metadata[CLASS_COLUMNS]
        .notna()
        .idxmax(axis=1)
        .str.replace("Class_", "", regex=False)
        .fillna("0")
        .astype(int)
    )
    metadata.loc[~metadata["has_defect"], "primary_class_id"] = 0

    for class_idx, column in enumerate(CLASS_COLUMNS, start=1):
        metadata[f"class_{class_idx}_present"] = metadata[column].notna()

    return metadata.sort_values("ImageId").reset_index(drop=True)


def _parse_neu_boxes(xml_path: Path) -> Tuple[int, int, List[Dict[str, int]], List[int]]:
    """Parses one NEU XML annotation into boxes and mapped class ids."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    width = int(root.findtext("./size/width", default="200"))
    height = int(root.findtext("./size/height", default="200"))

    boxes: List[Dict[str, int]] = []
    class_ids: List[int] = []

    for obj in root.findall("./object"):
        label = obj.findtext("name", default="").strip()
        if label not in NEU_TO_SEVERSTAL_CLASS:
            continue

        bbox = obj.find("bndbox")
        if bbox is None:
            continue

        class_id = NEU_TO_SEVERSTAL_CLASS[label]
        xmin = int(bbox.findtext("xmin", default="0"))
        ymin = int(bbox.findtext("ymin", default="0"))
        xmax = int(bbox.findtext("xmax", default=str(width - 1)))
        ymax = int(bbox.findtext("ymax", default=str(height - 1)))

        boxes.append(
            {
                "class_id": class_id,
                "source_label": label,
                "xmin": max(0, min(xmin, width - 1)),
                "ymin": max(0, min(ymin, height - 1)),
                "xmax": max(0, min(xmax, width - 1)),
                "ymax": max(0, min(ymax, height - 1)),
            }
        )
        class_ids.append(class_id)

    return width, height, boxes, class_ids


def build_neu_metadata(root_dir: str | Path | None = None) -> pd.DataFrame:
    """Builds unified metadata rows for the NEU-DET dataset."""
    root_dir = _ensure_path(root_dir or NEU_ROOT, "NEU-DET root")
    records: List[Dict[str, Any]] = []

    for source_split in ("train", "validation"):
        image_dir = root_dir / source_split / "images"
        annotation_dir = root_dir / source_split / "annotations"

        for xml_path in sorted(annotation_dir.glob("*.xml")):
            width, height, boxes, class_ids = _parse_neu_boxes(xml_path)
            image_name = xml_path.with_suffix(".jpg").name
            image_path = image_dir / image_name
            if not image_path.exists():
                continue

            presence_columns = _build_presence_columns(class_ids)
            has_defect = bool(class_ids)
            primary_class_id = _primary_class_from_ids(class_ids)
            source_label = boxes[0]["source_label"] if boxes else "unknown"

            records.append(
                {
                    "record_id": f"neu::{source_split}::{image_name}",
                    "source": "neu",
                    "source_split": source_split,
                    "ImageId": image_name,
                    "image_path": str(image_path),
                    "annotation_format": "boxes",
                    "supervision_level": "weak",
                    "height": height,
                    "width": width,
                    "boxes_json": json.dumps(boxes),
                    "mask_path": None,
                    "has_defect": has_defect,
                    "num_defects": len(set(class_ids)),
                    "primary_class_id": primary_class_id,
                    "source_label": source_label,
                    **presence_columns,
                }
            )

    return pd.DataFrame(records).sort_values(["source_split", "ImageId"]).reset_index(drop=True)


def build_kolektor_metadata(root_dir: str | Path | None = None) -> pd.DataFrame:
    """Builds unified metadata rows for the Kolektor dataset."""
    root_dir = _ensure_path(root_dir or KOLEKTOR_ROOT, "Kolektor root")
    records: List[Dict[str, Any]] = []

    for source_split in ("train", "test"):
        split_dir = root_dir / source_split
        image_paths = [
            path for path in _list_image_files(split_dir) if "_GT" not in path.stem
        ]

        for image_path in image_paths:
            mask_path = image_path.with_name(f"{image_path.stem}_GT{image_path.suffix}")
            if not mask_path.exists():
                continue

            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue

            has_defect = bool((mask > 0).any())
            class_ids = [KOLEKTOR_POSITIVE_CLASS] if has_defect else []
            presence_columns = _build_presence_columns(class_ids)

            records.append(
                {
                    "record_id": f"kolektor::{source_split}::{image_path.name}",
                    "source": "kolektor",
                    "source_split": source_split,
                    "ImageId": image_path.name,
                    "image_path": str(image_path),
                    "annotation_format": "binary_mask",
                    "supervision_level": "strong",
                    "height": int(mask.shape[0]),
                    "width": int(mask.shape[1]),
                    "boxes_json": "[]",
                    "mask_path": str(mask_path),
                    "has_defect": has_defect,
                    "num_defects": int(has_defect),
                    "primary_class_id": KOLEKTOR_POSITIVE_CLASS if has_defect else 0,
                    "source_label": "surface_imperfection",
                    **presence_columns,
                }
            )

    return pd.DataFrame(records).sort_values(["source_split", "ImageId"]).reset_index(drop=True)


def build_merged_metadata(
    include_severstal: bool = True,
    include_neu: bool = True,
    include_kolektor: bool = True,
    cache_path: str | Path | None = None,
    refresh_cache: bool = False,
) -> pd.DataFrame:
    """Combines the enabled datasets into one cached metadata table."""
    cache_path = Path(cache_path) if cache_path else _metadata_cache_path()
    if cache_path.exists() and not refresh_cache:
        metadata = pd.read_csv(cache_path)
        for column in CLASS_PRESENCE_COLUMNS:
            if column in metadata.columns:
                metadata[column] = metadata[column].astype(bool)
        if "has_defect" in metadata.columns:
            metadata["has_defect"] = metadata["has_defect"].astype(bool)
        return metadata

    parts: List[pd.DataFrame] = []

    if include_severstal:
        parts.append(build_severstal_metadata())
    if include_neu:
        parts.append(build_neu_metadata())
    if include_kolektor:
        parts.append(build_kolektor_metadata())

    if not parts:
        raise ValueError("At least one dataset must be enabled when building merged metadata.")

    metadata = pd.concat(parts, ignore_index=True, sort=False)

    for column in CLASS_COLUMNS:
        if column not in metadata.columns:
            metadata[column] = np.nan
    if "boxes_json" not in metadata.columns:
        metadata["boxes_json"] = "[]"
    if "mask_path" not in metadata.columns:
        metadata["mask_path"] = None
    if "source_label" not in metadata.columns:
        metadata["source_label"] = "unknown"

    metadata["height"] = metadata["height"].astype(int)
    metadata["width"] = metadata["width"].astype(int)
    metadata["num_defects"] = metadata["num_defects"].fillna(0).astype(int)
    metadata["primary_class_id"] = metadata["primary_class_id"].fillna(0).astype(int)
    metadata["has_defect"] = metadata["has_defect"].fillna(False).astype(bool)

    for column in CLASS_PRESENCE_COLUMNS:
        metadata[column] = metadata[column].fillna(False).astype(bool)

    metadata["source_class_key"] = metadata.apply(
        lambda row: f"{row['source']}__{row['primary_class_id']}",
        axis=1,
    )

    metadata.to_csv(cache_path, index=False)
    return metadata


def _choose_stratify_labels(dataframe: pd.DataFrame, test_size: float) -> Optional[pd.Series]:
    """Chooses the richest viable label set for stratified splitting."""
    candidate_columns = []
    if "source_class_key" in dataframe.columns:
        candidate_columns.append(dataframe["source_class_key"].astype(str))
    if "primary_class_id" in dataframe.columns and "has_defect" in dataframe.columns:
        candidate_columns.append(
            dataframe["primary_class_id"].astype(str) + "__" + dataframe["has_defect"].astype(int).astype(str)
        )
    if "has_defect" in dataframe.columns:
        candidate_columns.append(dataframe["has_defect"].astype(int).astype(str))

    for candidate in candidate_columns:
        value_counts = candidate.value_counts()
        if value_counts.empty:
            continue
        if value_counts.min() < 2:
            continue
        if int(round(len(dataframe) * test_size)) < candidate.nunique():
            continue
        return candidate

    return None


def _split_dataframe(
    dataframe: pd.DataFrame,
    test_size: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Splits a dataframe into train/test parts with safe stratification fallbacks."""
    if len(dataframe) == 0 or test_size <= 0:
        return dataframe.copy().reset_index(drop=True), dataframe.iloc[0:0].copy().reset_index(drop=True)
    if len(dataframe) == 1:
        return dataframe.copy().reset_index(drop=True), dataframe.iloc[0:0].copy().reset_index(drop=True)

    stratify = _choose_stratify_labels(dataframe, test_size=test_size)
    train_df, test_df = train_test_split(
        dataframe,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def get_protocol_splits(
    random_state: int = SEED,
    cache_path: str | Path | None = None,
    refresh_cache: bool = False,
    severstal_val_size: float = 0.10,
    severstal_test_size: float = 0.10,
    neu_val_size_from_train: float = 0.10,
    kolektor_val_size_from_train: float = 0.10,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Creates the project's train, validation, and final test protocol."""
    metadata = build_merged_metadata(cache_path=cache_path, refresh_cache=refresh_cache).copy()
    metadata["protocol_split"] = "unused"

    # Severstal: no public labels for test_images, so we carve out val + final test from official_train.
    severstal_df = metadata[metadata["source"] == "severstal"].copy()
    severstal_train_pool, severstal_holdout = _split_dataframe(
        severstal_df,
        test_size=(severstal_val_size + severstal_test_size),
        random_state=random_state,
    )
    if len(severstal_holdout) > 0 and (severstal_val_size + severstal_test_size) > 0:
        severstal_val, severstal_test = _split_dataframe(
            severstal_holdout,
            test_size=severstal_test_size / (severstal_val_size + severstal_test_size),
            random_state=random_state,
        )
    else:
        severstal_val = severstal_holdout.iloc[0:0].copy()
        severstal_test = severstal_holdout.iloc[0:0].copy()

    metadata.loc[metadata["record_id"].isin(severstal_train_pool["record_id"]), "protocol_split"] = "train"
    metadata.loc[metadata["record_id"].isin(severstal_val["record_id"]), "protocol_split"] = "val"
    metadata.loc[metadata["record_id"].isin(severstal_test["record_id"]), "protocol_split"] = "test_final"

    # NEU: keep official validation as final test, carve a small val split from train for tuning.
    neu_train_df = metadata[(metadata["source"] == "neu") & (metadata["source_split"] == "train")].copy()
    neu_train, neu_val = _split_dataframe(
        neu_train_df,
        test_size=neu_val_size_from_train,
        random_state=random_state,
    )
    neu_test = metadata[(metadata["source"] == "neu") & (metadata["source_split"] == "validation")].copy()

    metadata.loc[metadata["record_id"].isin(neu_train["record_id"]), "protocol_split"] = "train"
    metadata.loc[metadata["record_id"].isin(neu_val["record_id"]), "protocol_split"] = "val"
    metadata.loc[metadata["record_id"].isin(neu_test["record_id"]), "protocol_split"] = "test_final"

    # Kolektor: keep official test as final test, carve a small val split from official train.
    kolektor_train_df = metadata[(metadata["source"] == "kolektor") & (metadata["source_split"] == "train")].copy()
    kolektor_train, kolektor_val = _split_dataframe(
        kolektor_train_df,
        test_size=kolektor_val_size_from_train,
        random_state=random_state,
    )
    kolektor_test = metadata[(metadata["source"] == "kolektor") & (metadata["source_split"] == "test")].copy()

    metadata.loc[metadata["record_id"].isin(kolektor_train["record_id"]), "protocol_split"] = "train"
    metadata.loc[metadata["record_id"].isin(kolektor_val["record_id"]), "protocol_split"] = "val"
    metadata.loc[metadata["record_id"].isin(kolektor_test["record_id"]), "protocol_split"] = "test_final"

    train_df = metadata[metadata["protocol_split"] == "train"].reset_index(drop=True)
    val_df = metadata[metadata["protocol_split"] == "val"].reset_index(drop=True)
    test_df = metadata[metadata["protocol_split"] == "test_final"].reset_index(drop=True)

    return metadata.reset_index(drop=True), train_df, val_df, test_df


def rle_decode(mask_rle: Any, shape: Tuple[int, int] = (256, 1600)) -> np.ndarray:
    """Decodes a Severstal run-length encoded mask into a binary image."""
    if pd.isna(mask_rle) or mask_rle is None or mask_rle == "":
        return np.zeros(shape, dtype=np.uint8)

    s = str(mask_rle).split()
    starts = np.asarray(s[0::2], dtype=int) - 1
    lengths = np.asarray(s[1::2], dtype=int)
    ends = starts + lengths
    image = np.zeros(shape[0] * shape[1], dtype=np.uint8)

    for low, high in zip(starts, ends):
        image[low:high] = 1

    return image.reshape(shape, order="F")


def build_mask_from_record(record: Dict[str, Any]) -> np.ndarray:
    """Builds a multi-class mask from one unified metadata record."""
    height = int(record["height"])
    width = int(record["width"])
    masks = np.zeros((height, width, NUM_CLASSES), dtype=np.uint8)

    annotation_format = record["annotation_format"]
    if annotation_format == "rle":
        for class_idx in range(1, NUM_CLASSES + 1):
            masks[..., class_idx - 1] = rle_decode(record.get(f"Class_{class_idx}"), shape=(height, width))
        return masks

    if annotation_format == "boxes":
        for box in _json_loads(record.get("boxes_json")):
            class_id = int(box["class_id"])
            xmin = int(box["xmin"])
            ymin = int(box["ymin"])
            xmax = int(box["xmax"])
            ymax = int(box["ymax"])
            masks[ymin : ymax + 1, xmin : xmax + 1, class_id - 1] = 1
        return masks

    if annotation_format == "binary_mask":
        if not record.get("mask_path") or not record.get("has_defect", False):
            return masks

        binary_mask = cv2.imread(str(record["mask_path"]), cv2.IMREAD_GRAYSCALE)
        if binary_mask is None:
            return masks

        target_class = int(record.get("primary_class_id") or KOLEKTOR_POSITIVE_CLASS)
        masks[..., target_class - 1] = (binary_mask > 0).astype(np.uint8)
        return masks

    raise ValueError(f"Unsupported annotation format: {annotation_format}")


class SteelDatasetSegmentation(Dataset):
    """Creates segmentation samples from unified metadata for pixel-wise models."""

    def __init__(
        self,
        image_ids: Optional[Sequence[str]] = None,
        df: Optional[pd.DataFrame] = None,
        image_dir: Optional[str | Path] = None,
        transform: Optional[A.Compose] = None,
        image_size: int = 512,
        metadata_df: Optional[pd.DataFrame] = None,
        return_metadata: bool = False,
    ) -> None:
        """Stores records, transforms, and output settings for segmentation training."""
        self.transform = transform
        self.image_size = image_size
        self.return_metadata = return_metadata

        if metadata_df is not None:
            working_df = metadata_df.copy()
        else:
            if df is None:
                raise ValueError("Provide either metadata_df or the legacy df/image_ids inputs.")

            working_df = df.copy()
            if image_ids is not None:
                image_id_set = set(image_ids)
                working_df = working_df[working_df["ImageId"].isin(image_id_set)].copy()

            if "image_path" not in working_df.columns:
                if image_dir is None:
                    raise ValueError("image_dir is required when the dataframe does not include image_path.")
                image_dir = Path(image_dir)
                working_df["image_path"] = working_df["ImageId"].map(lambda image_id: str(image_dir / image_id))

            if "annotation_format" not in working_df.columns:
                working_df["annotation_format"] = "rle"
            if "height" not in working_df.columns:
                working_df["height"] = 256
            if "width" not in working_df.columns:
                working_df["width"] = 1600
            if "has_defect" not in working_df.columns:
                working_df["has_defect"] = working_df[CLASS_COLUMNS].notna().any(axis=1)
            if "record_id" not in working_df.columns:
                working_df["record_id"] = working_df["ImageId"].map(lambda image_id: f"legacy::{image_id}")

            for class_idx, column in enumerate(CLASS_COLUMNS, start=1):
                if column not in working_df.columns:
                    working_df[column] = np.nan
                presence_column = f"class_{class_idx}_present"
                if presence_column not in working_df.columns:
                    working_df[presence_column] = working_df[column].notna()

            if "num_defects" not in working_df.columns:
                working_df["num_defects"] = working_df[CLASS_COLUMNS].notna().sum(axis=1).astype(int)
            if "primary_class_id" not in working_df.columns:
                working_df["primary_class_id"] = (
                    working_df[CLASS_COLUMNS]
                    .notna()
                    .idxmax(axis=1)
                    .str.replace("Class_", "", regex=False)
                    .fillna("0")
                    .astype(int)
                )
                working_df.loc[~working_df["has_defect"], "primary_class_id"] = 0
            if "boxes_json" not in working_df.columns:
                working_df["boxes_json"] = "[]"
            if "mask_path" not in working_df.columns:
                working_df["mask_path"] = None

        self.records = working_df.to_dict("records")

    def __len__(self) -> int:
        """Returns the number of segmentation records available."""
        return len(self.records)

    def __getitem__(self, idx: int):
        """Loads one image/mask pair and applies the configured transform."""
        record = self.records[idx]
        image = load_image_as_rgb(record["image_path"])
        masks = build_mask_from_record(record)

        if self.transform:
            transformed = self.transform(image=image, mask=masks)
            image = transformed["image"]
            masks = transformed["mask"]
            if isinstance(masks, torch.Tensor):
                masks = masks.permute(2, 0, 1).float()
            else:
                masks = torch.from_numpy(masks).permute(2, 0, 1).float()
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float()
            masks = torch.from_numpy(masks).permute(2, 0, 1).float()

        if self.return_metadata:
            return image, masks, record

        return image, masks


def segmentation_collate_fn(batch: Sequence[Any]) -> Any:
    """Stacks segmentation batches and preserves optional per-sample metadata."""
    first_item = batch[0]
    if len(first_item) == 3:
        images, masks, metadata = zip(*batch)
        return torch.stack(images), torch.stack(masks), list(metadata)

    images, masks = zip(*batch)
    return torch.stack(images), torch.stack(masks)


def build_segmentation_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    img_size: int,
    batch_size: int,
    num_workers: int = 2,
    normalize: str = "imagenet",
    return_metadata: bool = False,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: Optional[int] = 2,
    drop_last: bool = False,
) -> Tuple[SteelDatasetSegmentation, SteelDatasetSegmentation, DataLoader, DataLoader]:
    """Builds train and validation dataloaders for segmentation experiments."""
    train_dataset = SteelDatasetSegmentation(
        metadata_df=train_df,
        transform=get_train_transform(img_size, normalize=normalize),
        image_size=img_size,
        return_metadata=return_metadata,
    )
    val_dataset = SteelDatasetSegmentation(
        metadata_df=val_df,
        transform=get_val_transform(img_size, normalize=normalize),
        image_size=img_size,
        return_metadata=return_metadata,
    )

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
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=generator,
        collate_fn=segmentation_collate_fn if return_metadata else None,
        drop_last=drop_last,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        collate_fn=segmentation_collate_fn if return_metadata else None,
        **loader_kwargs,
    )

    return train_dataset, val_dataset, train_loader, val_loader


def get_train_transform(img_size: int, normalize: str = "imagenet") -> A.Compose:
    """Builds the training augmentation pipeline for segmentation experiments."""
    mean, std = _normalization_stats(normalize)
    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomRotate90(p=0.2),
            A.RandomBrightnessContrast(p=0.25),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )


def get_val_transform(img_size: int, normalize: str = "imagenet") -> A.Compose:
    """Builds the validation preprocessing pipeline without random augmentations."""
    mean, std = _normalization_stats(normalize)
    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )


def _normalization_stats(normalize: str) -> Tuple[List[float], List[float]]:
    """Returns channel normalization statistics for the requested input mode."""
    if normalize == "grayscale":
        return [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def dice_coefficient(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """Computes the mean Dice score after thresholding predicted probabilities."""
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    return ((2.0 * intersection + smooth) / (union + smooth)).mean()


def iou_score(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """Computes the mean IoU score after thresholding predicted probabilities."""
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - intersection
    return ((intersection + smooth) / (union + smooth)).mean()


class DiceBCELoss(nn.Module):
    """Combines Dice loss and BCE loss for multi-label segmentation."""

    def __init__(self, dice_weight: float = 0.7) -> None:
        """Configures the Dice and BCE tradeoff used during training."""
        super().__init__()
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Computes the combined Dice and BCE training loss."""
        pred_sigmoid = torch.sigmoid(pred)
        dice_loss = 1 - dice_coefficient(pred_sigmoid, target)
        bce_loss = self.bce(pred, target)
        return self.dice_weight * dice_loss + (1 - self.dice_weight) * bce_loss



def forward_segmentation_model(
    model: nn.Module,
    images: torch.Tensor,
    target_size: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:
    """Runs a segmentation model and normalizes its output tensor shape."""
    try:
        outputs = model(pixel_values=images)
    except TypeError:
        outputs = model(images)

    logits = outputs.logits if hasattr(outputs, "logits") else outputs
    if target_size and logits.shape[-2:] != target_size:
        logits = nn.functional.interpolate(logits, size=target_size, mode="bilinear", align_corners=False)
    return logits


def _unpack_batch(batch: Any) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, Any]]]:
    """Normalizes batch tuples with or without attached metadata."""
    if isinstance(batch, (list, tuple)) and len(batch) == 3:
        images, masks, metadata = batch
        return images, masks, metadata
    if isinstance(batch, (list, tuple)) and len(batch) == 2:
        images, masks = batch
        return images, masks, None
    raise ValueError("Expected batch to contain (images, masks) or (images, masks, metadata).")


def run_training_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[GradScaler] = None,
    scheduler: Optional[Any] = None,
    scheduler_step_on_batch: bool = False,
    amp_mode: str = DEFAULT_AMP_MODE,
    grad_accumulation_steps: int = 1,
    max_grad_norm: Optional[float] = None,
    use_channels_last: bool = False,
    ema: Optional[ModelEMA] = None,
    show_progress: bool = True,
    progress_desc: str = "train",
) -> Dict[str, float]:
    """Runs one training epoch for a semantic segmentation model."""
    model.train()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    optimizer.zero_grad(set_to_none=True)

    progress_bar = None
    batch_iterator = enumerate(dataloader)
    if show_progress and tqdm is not None:
        progress_bar = tqdm(batch_iterator, total=len(dataloader), desc=progress_desc, leave=False)
        batch_iterator = progress_bar

    for batch_idx, batch in batch_iterator:
        images, masks, _ = _unpack_batch(batch)
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        images = maybe_to_channels_last(images, use_channels_last=use_channels_last)

        autocast_context = get_autocast_context(amp_mode)
        if scaler is not None:
            with autocast_context:
                logits = forward_segmentation_model(model, images, target_size=tuple(masks.shape[-2:]))
                loss = criterion(logits, masks)
                loss = loss / grad_accumulation_steps
            scaler.scale(loss).backward()
        else:
            with autocast_context:
                logits = forward_segmentation_model(model, images, target_size=tuple(masks.shape[-2:]))
                loss = criterion(logits, masks)
                loss = loss / grad_accumulation_steps
            loss.backward()

        should_step = ((batch_idx + 1) % grad_accumulation_steps == 0) or ((batch_idx + 1) == len(dataloader))
        if should_step and scaler is not None:
            if max_grad_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if ema is not None:
                ema.update(model)
        elif should_step:
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if ema is not None:
                ema.update(model)

        if scheduler is not None and scheduler_step_on_batch and should_step:
            scheduler.step()

        preds = torch.sigmoid(logits.detach())
        total_loss += (loss.item() * grad_accumulation_steps)
        total_dice += dice_coefficient(preds, masks).item()
        total_iou += iou_score(preds, masks).item()

        if progress_bar is not None:
            seen_batches = batch_idx + 1
            progress_bar.set_postfix(
                loss=f"{total_loss / seen_batches:.4f}",
                dice=f"{total_dice / seen_batches:.4f}",
                iou=f"{total_iou / seen_batches:.4f}",
            )

    num_batches = max(1, len(dataloader))
    return {
        "loss": total_loss / num_batches,
        "dice": total_dice / num_batches,
        "iou": total_iou / num_batches,
    }


def run_validation_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    scaler: Optional[GradScaler] = None,
    amp_mode: str = DEFAULT_AMP_MODE,
    use_channels_last: bool = False,
    show_progress: bool = True,
    progress_desc: str = "val",
) -> Dict[str, float]:
    """Runs one validation epoch and aggregates loss and segmentation metrics."""
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0

    with torch.no_grad():
        progress_bar = None
        batch_iterator = enumerate(dataloader)
        if show_progress and tqdm is not None:
            progress_bar = tqdm(batch_iterator, total=len(dataloader), desc=progress_desc, leave=False)
            batch_iterator = progress_bar

        for batch_idx, batch in batch_iterator:
            images, masks, _ = _unpack_batch(batch)
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            images = maybe_to_channels_last(images, use_channels_last=use_channels_last)

            autocast_context = get_autocast_context(amp_mode)
            if scaler is not None:
                with autocast_context:
                    logits = forward_segmentation_model(model, images, target_size=tuple(masks.shape[-2:]))
                    loss = criterion(logits, masks)
            else:
                with autocast_context:
                    logits = forward_segmentation_model(model, images, target_size=tuple(masks.shape[-2:]))
                    loss = criterion(logits, masks)

            preds = torch.sigmoid(logits)
            total_loss += loss.item()
            total_dice += dice_coefficient(preds, masks).item()
            total_iou += iou_score(preds, masks).item()

            if progress_bar is not None:
                seen_batches = batch_idx + 1
                progress_bar.set_postfix(
                    loss=f"{total_loss / seen_batches:.4f}",
                    dice=f"{total_dice / seen_batches:.4f}",
                    iou=f"{total_iou / seen_batches:.4f}",
                )

    num_batches = max(1, len(dataloader))
    return {
        "loss": total_loss / num_batches,
        "dice": total_dice / num_batches,
        "iou": total_iou / num_batches,
    }


def fit_segmentation_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    scheduler: Optional[Any] = None,
    scaler: Optional[GradScaler] = None,
    patience: int = 5,
    checkpoint_path: Optional[str | Path] = None,
    scheduler_step_on_batch: bool = False,
    model_name: str = "model",
    amp_mode: str = DEFAULT_AMP_MODE,
    grad_accumulation_steps: int = 1,
    max_grad_norm: Optional[float] = None,
    use_channels_last: bool = False,
    ema_decay: Optional[float] = None,
    show_progress: bool = True,
) -> Dict[str, List[float]]:
    """Trains a segmentation model with checkpointing, EMA, and early stopping."""
    history = {
        "epoch": [],
        "train_loss": [],
        "train_dice": [],
        "train_iou": [],
        "val_loss": [],
        "val_dice": [],
        "val_iou": [],
    }
    best_val_dice = float("-inf")
    patience_counter = 0
    ema = ModelEMA(model, decay=ema_decay) if ema_decay is not None else None

    for epoch in range(epochs):
        train_metrics = run_training_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            scheduler_step_on_batch=scheduler_step_on_batch,
            amp_mode=amp_mode,
            grad_accumulation_steps=grad_accumulation_steps,
            max_grad_norm=max_grad_norm,
            use_channels_last=use_channels_last,
            ema=ema,
            show_progress=show_progress,
            progress_desc=f"{model_name} train {epoch + 1}/{epochs}",
        )
        val_metrics = run_validation_epoch(
            model=ema.ema_model if ema is not None else model,
            dataloader=val_loader,
            criterion=criterion,
            scaler=scaler,
            amp_mode=amp_mode,
            use_channels_last=use_channels_last,
            show_progress=show_progress,
            progress_desc=f"{model_name} val {epoch + 1}/{epochs}",
        )

        if scheduler is not None and not scheduler_step_on_batch:
            scheduler.step()

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_metrics["loss"])
        history["train_dice"].append(train_metrics["dice"])
        history["train_iou"].append(train_metrics["iou"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_dice"].append(val_metrics["dice"])
        history["val_iou"].append(val_metrics["iou"])

        print(
            f"{model_name} epoch {epoch + 1}/{epochs} | "
            f"train_loss={train_metrics['loss']:.4f} train_dice={train_metrics['dice']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_dice={val_metrics['dice']:.4f} val_iou={val_metrics['iou']:.4f}"
        )

        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            patience_counter = 0
            if checkpoint_path is not None:
                checkpoint = {
                    "model_state_dict": unwrap_model(model).state_dict(),
                    "ema_state_dict": ema.state_dict() if ema is not None else None,
                    "history": history,
                    "amp_mode": amp_mode,
                    "grad_accumulation_steps": grad_accumulation_steps,
                    "max_grad_norm": max_grad_norm,
                    "use_channels_last": use_channels_last,
                    "ema_decay": ema_decay,
                }
                torch.save(checkpoint, str(checkpoint_path))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"{model_name} early stopping at epoch {epoch + 1}")
                break

    return history


def create_optimizer(
    params_or_model: Any,
    name: str = "adamw",
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    momentum: float = 0.9,
) -> torch.optim.Optimizer:
    """Creates the requested optimizer for a model or parameter iterable."""
    params = params_or_model.parameters() if hasattr(params_or_model, "parameters") else params_or_model
    optimizer_name = name.lower()

    if optimizer_name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if optimizer_name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if optimizer_name == "sgd":
        return torch.optim.SGD(
            params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )

    raise ValueError(f"Unsupported optimizer: {name}")


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    name: str = "cosine",
    epochs: int = 10,
    steps_per_epoch: Optional[int] = None,
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """Creates the requested learning-rate scheduler."""
    scheduler_name = name.lower()

    if scheduler_name == "none":
        return None
    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

    raise ValueError(f"Unsupported scheduler: {name}")


def create_scaler(enabled: Optional[bool] = None, amp_mode: str = DEFAULT_AMP_MODE) -> Optional[GradScaler]:
    """Creates a GradScaler when fp16 mixed precision is active."""
    use_amp = USE_AMP if enabled is None else enabled
    if use_amp and device == "cuda" and amp_mode.lower() == "fp16":
        return GradScaler()
    return None


def get_autocast_context(amp_mode: str = DEFAULT_AMP_MODE):
    """Returns the autocast context that matches the selected AMP mode."""
    amp_mode = amp_mode.lower()
    if device != "cuda" or amp_mode in {"off", "false", "none"}:
        return nullcontext()
    if amp_mode == "bf16":
        return autocast(dtype=torch.bfloat16)
    return autocast(dtype=torch.float16)


def enable_gradient_checkpointing(model: nn.Module) -> List[str]:
    """Enables gradient checkpointing on supported submodules when available."""
    enabled_modules: List[str] = []
    candidates = {
        "model": model,
        "backbone": getattr(model, "backbone", None),
        "sam": getattr(model, "sam", None),
        "image_encoder": getattr(getattr(model, "sam", None), "image_encoder", None),
    }

    for name, module in candidates.items():
        if module is None:
            continue
        if hasattr(module, "gradient_checkpointing_enable"):
            module.gradient_checkpointing_enable()
            enabled_modules.append(name)
        elif hasattr(module, "enable_gradient_checkpointing"):
            module.enable_gradient_checkpointing()
            enabled_modules.append(name)

    return enabled_modules


def prepare_model_for_optimizations(
    model: nn.Module,
    channels_last: bool = False,
    compile_model: bool = False,
    compile_mode: str = "default",
    gradient_checkpointing: bool = False,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Applies channels-last, checkpointing, and compile options to a model."""
    optimization_info = {
        "channels_last": channels_last,
        "compile_model": compile_model,
        "compile_mode": compile_mode,
        "gradient_checkpointing": gradient_checkpointing,
        "gradient_checkpointing_modules": [],
    }

    if channels_last:
        model = model.to(memory_format=torch.channels_last)

    if gradient_checkpointing:
        optimization_info["gradient_checkpointing_modules"] = enable_gradient_checkpointing(model)

    if compile_model and hasattr(torch, "compile"):
        model = torch.compile(model, mode=compile_mode)
    elif compile_model:
        optimization_info["compile_model"] = False

    return model, optimization_info


class ModelEMA:
    """Tracks an exponential moving average copy of a model."""

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        """Initializes the frozen EMA copy from the current model weights."""
        self.decay = decay
        self.ema_model = deepcopy(unwrap_model(model)).eval()
        for parameter in self.ema_model.parameters():
            parameter.requires_grad_(False)

    def update(self, model: nn.Module) -> None:
        """Updates the EMA weights from the latest model parameters."""
        with torch.no_grad():
            model_state = unwrap_model(model).state_dict()
            ema_state = self.ema_model.state_dict()
            for key, value in ema_state.items():
                if key not in model_state:
                    continue
                source = model_state[key].detach()
                if not torch.is_floating_point(source):
                    value.copy_(source)
                else:
                    value.mul_(self.decay).add_(source, alpha=(1.0 - self.decay))

    def state_dict(self) -> Dict[str, Any]:
        """Returns the EMA weights for checkpoint serialization."""
        return self.ema_model.state_dict()

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Loads previously saved EMA weights into the EMA copy."""
        self.ema_model.load_state_dict(state_dict)


def maybe_to_channels_last(images: torch.Tensor, use_channels_last: bool = False) -> torch.Tensor:
    """Converts image batches to channels-last layout when requested."""
    if use_channels_last and images.ndim == 4:
        return images.contiguous(memory_format=torch.channels_last)
    return images


def unwrap_model(model: nn.Module) -> nn.Module:
    """Returns the original model when torch.compile wraps it."""
    return getattr(model, "_orig_mod", model)


def quantize_model_for_inference(model: nn.Module, linear_only: bool = True) -> nn.Module:
    """Builds a CPU quantized copy of a trained model for inference benchmarks."""
    quantized_model = deepcopy(unwrap_model(model)).cpu().eval()
    layer_types = {nn.Linear} if linear_only else {nn.Linear, nn.Conv2d}
    return torch.quantization.quantize_dynamic(quantized_model, layer_types, dtype=torch.qint8)


def load_segmentation_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    prefer_ema: bool = True,
) -> Dict[str, Any]:
    """Loads a checkpoint into a segmentation model, preferring EMA weights."""
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    state_dict = checkpoint
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        if prefer_ema and checkpoint.get("ema_state_dict") is not None:
            state_dict = checkpoint["ema_state_dict"]
        else:
            state_dict = checkpoint["model_state_dict"]

    unwrap_model(model).load_state_dict(state_dict)
    return checkpoint if isinstance(checkpoint, dict) else {"raw_state_dict": True}


def count_trainable_parameters(model: nn.Module) -> int:
    """Counts the parameters that still require gradients."""
    return sum(parameter.numel() for parameter in unwrap_model(model).parameters() if parameter.requires_grad)


def benchmark_model(
    model: nn.Module,
    input_shape: Tuple[int, int, int, int],
    runs: int = 20,
    warmup: int = 5,
    benchmark_device: Optional[str] = None,
) -> Dict[str, float]:
    """Measures latency, throughput, and peak memory for inference."""
    benchmark_device = benchmark_device or device
    model = model.to(benchmark_device).eval()
    sample = torch.randn(*input_shape, device=benchmark_device)

    if benchmark_device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(sample)
        if benchmark_device == "cuda":
            torch.cuda.synchronize()

        timings_ms: List[float] = []
        for _ in range(runs):
            if benchmark_device == "cuda":
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                _ = model(sample)
                end.record()
                torch.cuda.synchronize()
                timings_ms.append(float(start.elapsed_time(end)))
            else:
                import time

                start = time.perf_counter()
                _ = model(sample)
                timings_ms.append((time.perf_counter() - start) * 1000.0)

    peak_memory_mb = 0.0
    if benchmark_device == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    return {
        "mean_latency_ms": float(np.mean(timings_ms)),
        "std_latency_ms": float(np.std(timings_ms)),
        "throughput_img_s": float((input_shape[0] * 1000.0) / np.mean(timings_ms)),
        "peak_memory_mb": float(peak_memory_mb),
    }


def summarize_merged_metadata(metadata: pd.DataFrame) -> pd.DataFrame:
    """Aggregates dataset-level counts from the merged metadata table."""
    summary = (
        metadata.groupby(["source", "source_split"], dropna=False)
        .agg(
            images=("record_id", "count"),
            defect_images=("has_defect", "sum"),
            clean_images=("has_defect", lambda series: int((~series).sum())),
            weak_labels=("supervision_level", lambda series: int((series == "weak").sum())),
        )
        .reset_index()
    )

    for class_idx in range(1, NUM_CLASSES + 1):
        class_counts = (
            metadata.groupby(["source", "source_split"], dropna=False)[f"class_{class_idx}_present"]
            .sum()
            .reset_index(name=f"class_{class_idx}_images")
        )
        summary = summary.merge(class_counts, on=["source", "source_split"], how="left")

    return summary


def summarize_class_balance(metadata: pd.DataFrame) -> pd.DataFrame:
    """Summarizes how often each class appears in the metadata."""
    rows: List[Dict[str, Any]] = []
    total_images = len(metadata)

    for class_idx in range(1, NUM_CLASSES + 1):
        count = int(metadata[f"class_{class_idx}_present"].sum())
        rows.append(
            {
                "class_id": class_idx,
                "images_with_class": count,
                "share_pct": (100.0 * count / total_images) if total_images else 0.0,
            }
        )

    return pd.DataFrame(rows)


def summarize_protocol_splits(metadata: pd.DataFrame) -> pd.DataFrame:
    """Aggregates counts for each source inside the protocol splits."""
    summary = (
        metadata.groupby(["protocol_split", "source"], dropna=False)
        .agg(
            images=("record_id", "count"),
            defect_images=("has_defect", "sum"),
            clean_images=("has_defect", lambda series: int((~series).sum())),
        )
        .reset_index()
        .assign(
            protocol_rank=lambda df: df["protocol_split"].map(
                {"train": 0, "val": 1, "test_final": 2, "unused": 3}
            ).fillna(99)
        )
        .sort_values(["protocol_rank", "source"])
        .drop(columns=["protocol_rank"])
        .reset_index(drop=True)
    )
    return summary


def compute_mask_area_table(metadata: pd.DataFrame, max_samples: Optional[int] = None) -> pd.DataFrame:
    """Computes per-record and per-class mask coverage statistics."""
    working_df = metadata if max_samples is None else metadata.sample(min(max_samples, len(metadata)), random_state=SEED)

    rows: List[Dict[str, Any]] = []
    for record in working_df.to_dict("records"):
        masks = build_mask_from_record(record)
        total_pixels = masks.shape[0] * masks.shape[1]
        for class_idx in range(NUM_CLASSES):
            rows.append(
                {
                    "record_id": record["record_id"],
                    "source": record["source"],
                    "class_id": class_idx + 1,
                    "pixels": int(masks[..., class_idx].sum()),
                    "share_pct": 100.0 * float(masks[..., class_idx].sum()) / total_pixels,
                }
            )

    return pd.DataFrame(rows)


def overlay_multiclass_mask(
    image: np.ndarray,
    mask_channels: np.ndarray,
    alpha: float = 0.45,
    colors: Optional[Dict[int, Tuple[int, int, int]]] = None,
) -> np.ndarray:
    """Overlays colored class masks on top of an input image."""
    if colors is None:
        colors = {
            1: (255, 99, 71),
            2: (65, 105, 225),
            3: (50, 205, 50),
            4: (255, 215, 0),
        }

    overlay = image.copy()
    for class_idx in range(mask_channels.shape[-1]):
        mask = mask_channels[..., class_idx].astype(bool)
        color = np.array(colors[class_idx + 1], dtype=np.uint8)
        overlay[mask] = ((1 - alpha) * overlay[mask] + alpha * color).astype(np.uint8)

    return overlay


def denormalize_image_tensor(image_tensor: torch.Tensor, normalize: str = "imagenet") -> np.ndarray:
    """Converts a normalized tensor back to a displayable uint8 image."""
    mean, std = _normalization_stats(normalize)
    image = image_tensor.detach().cpu().permute(1, 2, 0).numpy()
    image = (image * np.array(std)) + np.array(mean)
    image = np.clip(image, 0.0, 1.0)
    image = (image * 255).astype(np.uint8)
    return image


def collect_prediction_examples(
    model: nn.Module,
    dataset: SteelDatasetSegmentation,
    num_samples: int = 4,
    threshold: float = 0.5,
    normalize: str = "imagenet",
) -> List[Dict[str, Any]]:
    """Collects a small balanced set of prediction examples for visualization."""
    import random
    
    was_training = model.training
    model.eval()

    examples: List[Dict[str, Any]] = []
    
    num_defects_wanted = int(num_samples * 0.75) 
    num_clean_wanted = num_samples - num_defects_wanted 
    
    defects_collected = 0
    clean_collected = 0

    all_indices = list(range(len(dataset)))
    random.shuffle(all_indices)

    with torch.no_grad():
        for idx in all_indices:
            sample = dataset[idx]
            if len(sample) == 3:
                image_tensor, true_mask_tensor, metadata = sample
            else:
                image_tensor, true_mask_tensor = sample
                metadata = dataset.records[idx]

            has_defect = true_mask_tensor.sum() > 0

            if has_defect and defects_collected < num_defects_wanted:
                defects_collected += 1
            elif not has_defect and clean_collected < num_clean_wanted:
                clean_collected += 1
            else:
                continue 

            logits = forward_segmentation_model(
                model,
                image_tensor.unsqueeze(0).to(device),
                target_size=tuple(true_mask_tensor.shape[-2:]),
            )
            pred_mask = (torch.sigmoid(logits).squeeze(0).cpu().numpy() > threshold).astype(np.uint8)

            examples.append(
                {
                    "index": int(idx),
                    "metadata": metadata,
                    "image": denormalize_image_tensor(image_tensor, normalize=normalize),
                    "true_mask": true_mask_tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8),
                    "pred_mask": np.transpose(pred_mask, (1, 2, 0)),
                }
            )
            
            if defects_collected == num_defects_wanted and clean_collected == num_clean_wanted:
                break

    if was_training:
        model.train()

    return examples


def plot_training_history(history: Dict[str, List[float]], title_prefix: str = "Model"):
    """Plots loss and validation metrics from a training history dictionary."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].plot(history["train_loss"], label="Train Loss")
    axes[0].plot(history["val_loss"], label="Val Loss")
    axes[0].set_title(f"{title_prefix} Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(history["train_dice"], label="Train Dice", color="tab:green")
    axes[1].plot(history["val_dice"], label="Val Dice", color="tab:blue")
    axes[1].plot(history["val_iou"], label="Val IoU", color="tab:orange")
    axes[1].set_title(f"{title_prefix} Metrics")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    return fig


def visualize_prediction_examples(
    examples: Sequence[Dict[str, Any]],
    max_examples: int = 4,
):
    """Displays images alongside ground-truth and predicted overlays."""
    import matplotlib.pyplot as plt

    shown_examples = list(examples[:max_examples])
    if not shown_examples:
        raise ValueError("No prediction examples available.")

    fig, axes = plt.subplots(len(shown_examples), 3, figsize=(14, 4 * len(shown_examples)))
    if len(shown_examples) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_idx, example in enumerate(shown_examples):
        image = example["image"]
        
        true_overlay = overlay_multiclass_mask(image, example["true_mask"])
        pred_overlay = overlay_multiclass_mask(image, example["pred_mask"])

        axes[row_idx, 0].imshow(image, cmap="gray")
        axes[row_idx, 0].set_title(f"Image | {example['metadata']['source']}")
        axes[row_idx, 1].imshow(true_overlay)
        axes[row_idx, 1].set_title("Ground Truth")
        axes[row_idx, 2].imshow(pred_overlay)
        axes[row_idx, 2].set_title("Prediction")

        for col_idx in range(3):
            axes[row_idx, col_idx].axis("off")

    plt.tight_layout()
    return fig


def build_experiment_result_row(
    model_name: str,
    history: Dict[str, List[float]],
    benchmark: Optional[Dict[str, float]] = None,
    trainable_params: Optional[int] = None,
    optimizer_name: Optional[str] = None,
    scheduler_name: Optional[str] = None,
    loss_name: Optional[str] = None,
    optimization_profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Builds one comparison-table row from training and benchmark results."""
    if not history.get("val_dice"):
        raise ValueError("History is empty. Run training before building an experiment row.")

    best_idx = int(np.argmax(history["val_dice"]))
    row = {
        "model": model_name,
        "epochs_ran": len(history["epoch"]),
        "best_epoch": int(history["epoch"][best_idx]),
        "best_val_dice": float(history["val_dice"][best_idx]),
        "best_val_iou": float(history["val_iou"][best_idx]),
        "best_val_loss": float(history["val_loss"][best_idx]),
        "final_train_loss": float(history["train_loss"][-1]),
        "final_val_loss": float(history["val_loss"][-1]),
        "optimizer": optimizer_name,
        "scheduler": scheduler_name,
        "loss": loss_name,
        "trainable_params": int(trainable_params) if trainable_params is not None else None,
    }

    if benchmark:
        row.update(benchmark)
        if benchmark.get("mean_latency_ms"):
            row["dice_per_ms"] = row["best_val_dice"] / benchmark["mean_latency_ms"]
        if benchmark.get("peak_memory_mb"):
            row["dice_per_gb"] = row["best_val_dice"] / max(benchmark["peak_memory_mb"] / 1024.0, 1e-6)

    if optimization_profile:
        for key, value in optimization_profile.items():
            row[f"opt_{key}"] = value

    return row
