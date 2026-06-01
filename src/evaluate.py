from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.datasets import build_dataset
from src.inference import load_predictor


def compute_metrics(labels: list[int], scores: list[float], threshold: float) -> dict[str, float | None]:
    y_true = np.asarray(labels, dtype=np.int64)
    y_score = np.asarray(scores, dtype=np.float32)
    y_pred = (y_score >= threshold).astype(np.int64)

    total = max(int(y_true.size), 1)
    true_positive = int(((y_true == 1) & (y_pred == 1)).sum())
    false_positive = int(((y_true == 0) & (y_pred == 1)).sum())
    false_negative = int(((y_true == 1) & (y_pred == 0)).sum())
    correct = int((y_true == y_pred).sum())
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    metrics: dict[str, float | None] = {
        "accuracy": correct / total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auroc": None,
    }
    if len(np.unique(y_true)) > 1:
        try:
            from sklearn.metrics import roc_auc_score

            metrics["auroc"] = float(roc_auc_score(y_true, y_score))
        except Exception:
            metrics["auroc"] = None
    return metrics


def pixel_auroc(masks: list[np.ndarray], maps: list[np.ndarray]) -> float | None:
    if not masks:
        return None
    y_true = np.concatenate([mask.reshape(-1) for mask in masks]).astype(np.uint8)
    y_score = np.concatenate([score_map.reshape(-1) for score_map in maps]).astype(np.float32)
    if len(np.unique(y_true)) < 2:
        return None
    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return None


def evaluate(config: dict[str, Any], checkpoint: str | Path, split: str, output_path: str | Path) -> dict[str, Any]:
    predictor = load_predictor(checkpoint, device=config.get("experiment", {}).get("device", "auto"))
    dataset = build_dataset(config, split=split)
    dataset_config = config.get("dataset", {})
    dataloader = DataLoader(
        dataset,
        batch_size=int(dataset_config.get("batch_size", 8)),
        shuffle=False,
        num_workers=int(dataset_config.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )

    labels: list[int] = []
    scores: list[float] = []
    masks: list[np.ndarray] = []
    maps: list[np.ndarray] = []

    for batch in tqdm(dataloader, desc="evaluate", leave=False):
        predictions = predictor.predict_batch(batch["image"])
        scores.extend(predictions["score"].tolist())
        labels.extend(int(value) for value in batch["label"].cpu().tolist())

        anomaly_maps = np.array(predictions["anomaly_map"].cpu().tolist(), dtype=np.float32)
        batch_masks = np.array(batch["mask"].cpu().tolist(), dtype=np.float32)
        for mask, anomaly_map in zip(batch_masks, anomaly_maps):
            if mask.max() > 0:
                masks.append(mask[0])
                maps.append(anomaly_map[0])

    metrics = compute_metrics(labels, scores, predictor.threshold)
    metrics["pixel_auroc"] = pixel_auroc(masks, maps)
    result = {
        "split": split,
        "num_images": len(labels),
        "threshold": predictor.threshold,
        "metrics": metrics,
    }

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DefectVision-AD checkpoints.")
    parser.add_argument("--config", default="configs/model_autoencoder.yaml")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/autoencoder.pt")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", default="outputs/metrics/evaluation.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = evaluate(load_config(args.config), args.checkpoint, args.split, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
