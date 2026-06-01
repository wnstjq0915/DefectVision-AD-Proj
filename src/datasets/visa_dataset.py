from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from src.datasets.mvtec_dataset import IMAGE_EXTENSIONS
from src.preprocessing.transforms import ImagePreprocessor

NORMAL_NAMES = {"good", "normal", "ok", "negative", "0"}
IMAGE_COLUMN_CANDIDATES = ("image", "image_path", "img_path", "filename", "file")
MASK_COLUMN_CANDIDATES = ("mask", "mask_path", "gt", "ground_truth")
LABEL_COLUMN_CANDIDATES = ("label", "is_anomaly", "anomaly")
SPLIT_COLUMN_CANDIDATES = ("split", "subset")
CATEGORY_COLUMN_CANDIDATES = ("category", "class", "object")


@dataclass(frozen=True)
class VisAItem:
    image_path: Path
    category: str
    defect_type: str
    label: int
    mask_path: Path | None


def _first_column(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    lower_to_original = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate in lower_to_original:
            return lower_to_original[candidate]
    return None


def _is_normal(value: Any) -> bool:
    return str(value).strip().lower() in NORMAL_NAMES


def _resolve_path(root: Path, value: Any) -> Path | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def _infer_mask_path(root: Path, image_path: Path) -> Path | None:
    replacements = [
        ("Images", "Masks"),
        ("images", "masks"),
        ("image", "mask"),
        ("Data", "Annotations"),
    ]
    for source, target in replacements:
        candidate = Path(str(image_path).replace(source, target))
        if candidate.exists() and candidate.suffix.lower() in IMAGE_EXTENSIONS:
            return candidate
        png_candidate = candidate.with_suffix(".png")
        if png_candidate.exists():
            return png_candidate

    for mask_root_name in ("masks", "Masks", "ground_truth", "Annotations"):
        mask_root = root / mask_root_name
        if mask_root.exists():
            matches = sorted(mask_root.rglob(f"{image_path.stem}*"))
            matches = [path for path in matches if path.suffix.lower() in IMAGE_EXTENSIONS]
            if matches:
                return matches[0]
    return None


class VisADataset(Dataset):
    """Flexible VisA dataset loader for CSV-based or directory-based layouts."""

    def __init__(
        self,
        root: str | Path,
        category: str | None = None,
        split: str = "train",
        image_size: int = 256,
        preprocessing: dict[str, Any] | None = None,
        split_csv: str | Path | None = None,
    ) -> None:
        self.root = Path(root)
        self.category = category
        self.split = split.lower()
        self.split_csv = Path(split_csv) if split_csv else None
        self.preprocessing_config = preprocessing or {}
        self.preprocessor = ImagePreprocessor(
            image_size=image_size,
            mean=self.preprocessing_config.get("normalize_mean", (0.485, 0.456, 0.406)),
            std=self.preprocessing_config.get("normalize_std", (0.229, 0.224, 0.225)),
            grayscale=bool(self.preprocessing_config.get("grayscale", False)),
            gaussian_blur=bool(self.preprocessing_config.get("gaussian_blur", False)),
            canny=bool(self.preprocessing_config.get("canny", False)),
            hist_equalize=bool(self.preprocessing_config.get("hist_equalize", False)),
            augment=self.split == "train" and bool(self.preprocessing_config.get("augment", False)),
        )
        self.items = self._collect_items()
        if not self.items:
            raise FileNotFoundError(
                f"No VisA images found for split={self.split!r}, root={self.root}, category={self.category!r}"
            )

    def _collect_items(self) -> list[VisAItem]:
        if self.split_csv:
            return self._collect_from_csv()
        return self._collect_from_directories()

    def _collect_from_csv(self) -> list[VisAItem]:
        import pandas as pd

        csv_path = self.split_csv if self.split_csv.is_absolute() else self.root / self.split_csv
        frame = pd.read_csv(csv_path)
        columns = list(frame.columns)
        image_col = _first_column(columns, IMAGE_COLUMN_CANDIDATES)
        mask_col = _first_column(columns, MASK_COLUMN_CANDIDATES)
        label_col = _first_column(columns, LABEL_COLUMN_CANDIDATES)
        split_col = _first_column(columns, SPLIT_COLUMN_CANDIDATES)
        category_col = _first_column(columns, CATEGORY_COLUMN_CANDIDATES)

        if image_col is None:
            raise ValueError(f"VisA split CSV needs an image path column: {csv_path}")

        items: list[VisAItem] = []
        for row in frame.to_dict("records"):
            if split_col and str(row[split_col]).strip().lower() != self.split:
                continue
            row_category = str(row.get(category_col, self.category or "unknown"))
            if self.category and row_category != self.category:
                continue

            image_path = _resolve_path(self.root, row[image_col])
            if image_path is None or not image_path.exists():
                continue

            mask_path = _resolve_path(self.root, row.get(mask_col)) if mask_col else _infer_mask_path(self.root, image_path)
            label_value = row.get(label_col, "normal")
            label = 0 if _is_normal(label_value) else int(bool(label_value))
            defect_type = "good" if label == 0 else str(label_value)
            items.append(VisAItem(image_path, row_category, defect_type, label, mask_path if mask_path and mask_path.exists() else None))
        return items

    def _collect_from_directories(self) -> list[VisAItem]:
        search_root = self.root / self.category if self.category and (self.root / self.category).exists() else self.root
        image_paths = sorted(path for path in search_root.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
        image_paths = [
            path
            for path in image_paths
            if not any(part.lower() in {"mask", "masks", "ground_truth", "annotations"} for part in path.parts)
        ]

        items: list[VisAItem] = []
        for image_path in image_paths:
            parts = {part.lower() for part in image_path.parts}
            if self.split in {"train", "test", "val", "valid"} and self.split not in parts:
                continue

            category = self.category or image_path.parent.name
            defect_type = image_path.parent.name
            label = 0 if defect_type.lower() in NORMAL_NAMES else 1
            mask_path = _infer_mask_path(self.root, image_path)
            items.append(VisAItem(image_path, category, defect_type, label, mask_path))
        return items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.items[index]
        image = self.preprocessor(item.image_path)
        mask = self.preprocessor.mask(item.mask_path) if item.mask_path else self.preprocessor.empty_mask()
        return {
            "image": image,
            "mask": mask,
            "label": torch.tensor(item.label, dtype=torch.long),
            "path": str(item.image_path),
            "category": item.category,
            "defect_type": item.defect_type,
        }
