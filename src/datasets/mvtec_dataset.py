from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from src.preprocessing.transforms import ImagePreprocessor

IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}


@dataclass(frozen=True)
class MVTecItem:
    image_path: Path
    category: str
    defect_type: str
    label: int
    mask_path: Path | None


def _image_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(path for path in directory.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)


def _category_roots(root: Path, category: str | None) -> list[tuple[str, Path]]:
    if category and category.lower() != "all":
        category_root = root / category
        if category_root.exists():
            return [(category, category_root)]
        if (root / "train").exists() or (root / "test").exists():
            return [(category, root)]
        raise FileNotFoundError(f"MVTec category not found: {category_root}")

    if (root / "train").exists() or (root / "test").exists():
        return [(root.name, root)]

    categories = [(path.name, path) for path in sorted(root.iterdir()) if path.is_dir()]
    categories = [(name, path) for name, path in categories if (path / "train").exists() or (path / "test").exists()]
    if not categories:
        raise FileNotFoundError(f"No MVTec categories found under: {root}")
    return categories


def _infer_mask_path(category_root: Path, defect_type: str, image_path: Path) -> Path | None:
    if defect_type == "good":
        return None

    ground_truth_dir = category_root / "ground_truth" / defect_type
    if not ground_truth_dir.exists():
        return None

    candidates = [
        ground_truth_dir / f"{image_path.stem}_mask{image_path.suffix}",
        ground_truth_dir / f"{image_path.stem}_mask.png",
        ground_truth_dir / f"{image_path.stem}.png",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    matches = sorted(path for path in ground_truth_dir.glob(f"{image_path.stem}*") if path.suffix.lower() in IMAGE_EXTENSIONS)
    return matches[0] if matches else None


class MVTecADDataset(Dataset):
    """Dataset loader for the standard MVTec AD directory layout."""

    def __init__(
        self,
        root: str | Path,
        category: str | None = None,
        split: str = "train",
        image_size: int = 256,
        preprocessing: dict[str, Any] | None = None,
    ) -> None:
        self.root = Path(root)
        self.category = category
        self.split = split.lower()
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
                f"No MVTec images found for split={self.split!r}, root={self.root}, category={self.category!r}"
            )

    def _collect_items(self) -> list[MVTecItem]:
        items: list[MVTecItem] = []
        for category_name, category_root in _category_roots(self.root, self.category):
            split_root = category_root / self.split
            if not split_root.exists():
                continue
            for defect_dir in sorted(path for path in split_root.iterdir() if path.is_dir()):
                defect_type = defect_dir.name
                label = 0 if defect_type == "good" else 1
                for image_path in _image_files(defect_dir):
                    items.append(
                        MVTecItem(
                            image_path=image_path,
                            category=category_name,
                            defect_type=defect_type,
                            label=label,
                            mask_path=_infer_mask_path(category_root, defect_type, image_path),
                        )
                    )
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
