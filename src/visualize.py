from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from src.preprocessing.transforms import load_image


def to_numpy_map(anomaly_map: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(anomaly_map, torch.Tensor):
        anomaly_map = np.array(anomaly_map.detach().cpu().tolist(), dtype=np.float32)
    anomaly_map = np.asarray(anomaly_map)
    if anomaly_map.ndim == 4:
        anomaly_map = anomaly_map[0, 0]
    elif anomaly_map.ndim == 3:
        anomaly_map = anomaly_map[0] if anomaly_map.shape[0] == 1 else anomaly_map[:, :, 0]
    return anomaly_map.astype(np.float32)


def normalize_map(anomaly_map: np.ndarray | torch.Tensor) -> np.ndarray:
    score_map = to_numpy_map(anomaly_map)
    min_value = float(score_map.min())
    max_value = float(score_map.max())
    if max_value - min_value < 1e-12:
        return np.zeros_like(score_map, dtype=np.float32)
    return (score_map - min_value) / (max_value - min_value)


def make_heatmap(anomaly_map: np.ndarray | torch.Tensor, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    normalized = normalize_map(anomaly_map)
    heatmap = cv2.applyColorMap((normalized * 255).astype(np.uint8), colormap)
    return cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)


def overlay_heatmap(
    image: str | Path | Image.Image | np.ndarray,
    anomaly_map: np.ndarray | torch.Tensor,
    alpha: float = 0.45,
) -> np.ndarray:
    base_image = np.asarray(load_image(image).resize(make_heatmap(anomaly_map).shape[1::-1])).astype(np.uint8)
    heatmap = make_heatmap(anomaly_map)
    return cv2.addWeighted(base_image, 1.0 - alpha, heatmap, alpha, 0)


def make_binary_mask(anomaly_map: np.ndarray | torch.Tensor, threshold: float) -> np.ndarray:
    score_map = to_numpy_map(anomaly_map)
    return (score_map >= threshold).astype(np.uint8) * 255


def save_image(array: np.ndarray, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array.astype(np.uint8)).save(output_path)
    return output_path


def save_prediction_panel(
    image: str | Path | Image.Image | np.ndarray,
    anomaly_map: np.ndarray | torch.Tensor,
    mask: np.ndarray | torch.Tensor | None,
    output_path: str | Path,
    title: str | None = None,
) -> Path:
    import matplotlib.pyplot as plt

    base_image = np.asarray(load_image(image).resize(make_heatmap(anomaly_map).shape[1::-1]))
    heatmap = make_heatmap(anomaly_map)
    overlay = overlay_heatmap(base_image, anomaly_map)

    columns = 4 if mask is not None else 3
    fig, axes = plt.subplots(1, columns, figsize=(4 * columns, 4))
    if columns == 1:
        axes = [axes]
    axes[0].imshow(base_image)
    axes[0].set_title("Image")
    axes[1].imshow(heatmap)
    axes[1].set_title("Anomaly map")
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    if mask is not None:
        axes[3].imshow(to_numpy_map(mask), cmap="gray")
        axes[3].set_title("Mask")

    for axis in axes:
        axis.axis("off")
    if title:
        fig.suptitle(title)
    fig.tight_layout()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output
