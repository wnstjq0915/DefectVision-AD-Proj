from __future__ import annotations

from typing import Any

from src.datasets.mvtec_dataset import MVTecADDataset

__all__ = ["MVTecADDataset", "build_dataset"]


def build_dataset(config: dict[str, Any], split: str):
    """Build a dataset from a project config."""
    dataset_config = config.get("dataset", config)
    preprocessing_config = config.get("preprocessing", dataset_config.get("preprocessing", {}))
    name = dataset_config.get("name", "mvtec").lower()

    common_kwargs = {
        "root": dataset_config.get("root", "data/raw/mvtec"),
        "category": dataset_config.get("category"),
        "split": split,
        "image_size": dataset_config.get("image_size", 256),
        "preprocessing": preprocessing_config,
    }

    if name in {"mvtec", "mvtec_ad", "mvtec-ad"}:
        return MVTecADDataset(**common_kwargs)
    if name in {"visa", "visual_anomaly"}:
        from src.datasets.visa_dataset import VisADataset

        return VisADataset(
            **common_kwargs,
            split_csv=dataset_config.get("split_csv"),
        )

    raise ValueError(f"Unsupported dataset name: {name}")
