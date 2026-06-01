from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import torch

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.models import ConvAutoEncoder, PatchCoreWrapper, reconstruction_error_map
from src.preprocessing.transforms import ImagePreprocessor, load_image
from src.visualize import make_binary_mask, make_heatmap, overlay_heatmap, save_image

IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}


class AutoEncoderPredictor:
    def __init__(self, checkpoint: dict[str, Any], device: torch.device, image_size: int | None = None) -> None:
        self.checkpoint = checkpoint
        self.config = checkpoint.get("config", {})
        model_config = self.config.get("model", {})
        dataset_config = self.config.get("dataset", {})
        preprocessing_config = self.config.get("preprocessing", {})
        self.threshold = float(checkpoint.get("threshold", 0.0))
        self.device = device

        self.model = ConvAutoEncoder(
            in_channels=int(model_config.get("in_channels", 3)),
            base_channels=int(model_config.get("base_channels", 32)),
            latent_channels=int(model_config.get("latent_channels", 256)),
        ).to(device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

        self.preprocessor = ImagePreprocessor(
            image_size=image_size or dataset_config.get("image_size", 256),
            mean=preprocessing_config.get("normalize_mean", (0.485, 0.456, 0.406)),
            std=preprocessing_config.get("normalize_std", (0.229, 0.224, 0.225)),
            grayscale=bool(preprocessing_config.get("grayscale", False)),
            gaussian_blur=bool(preprocessing_config.get("gaussian_blur", False)),
            canny=bool(preprocessing_config.get("canny", False)),
            hist_equalize=bool(preprocessing_config.get("hist_equalize", False)),
        )

    @torch.no_grad()
    def predict_batch(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        images = images.to(self.device)
        reconstructions = self.model(images)
        anomaly_map = reconstruction_error_map(images, reconstructions)
        scores = anomaly_map.flatten(1).mean(dim=1)
        return {"score": scores.detach().cpu(), "anomaly_map": anomaly_map.detach().cpu()}

    def predict_image(self, image: str | Path) -> dict[str, Any]:
        tensor = self.preprocessor(image).unsqueeze(0)
        prediction = self.predict_batch(tensor)
        score = float(prediction["score"][0])
        anomaly_map = prediction["anomaly_map"][0]
        return {
            "score": score,
            "label": "anomaly" if score >= self.threshold else "normal",
            "threshold": self.threshold,
            "anomaly_map": anomaly_map,
        }


class PatchCorePredictor:
    def __init__(self, checkpoint: dict[str, Any], device: torch.device, image_size: int | None = None) -> None:
        self.checkpoint = checkpoint
        self.config = checkpoint.get("config", {})
        dataset_config = self.config.get("dataset", {})
        preprocessing_config = self.config.get("preprocessing", {})
        self.threshold = float(checkpoint.get("threshold", 0.0))
        self.device = device
        self.model = PatchCoreWrapper.load(checkpoint)
        self.preprocessor = ImagePreprocessor(
            image_size=image_size or dataset_config.get("image_size", 256),
            mean=preprocessing_config.get("normalize_mean", (0.485, 0.456, 0.406)),
            std=preprocessing_config.get("normalize_std", (0.229, 0.224, 0.225)),
            grayscale=bool(preprocessing_config.get("grayscale", False)),
            gaussian_blur=bool(preprocessing_config.get("gaussian_blur", False)),
            canny=bool(preprocessing_config.get("canny", False)),
            hist_equalize=bool(preprocessing_config.get("hist_equalize", False)),
        )

    @torch.no_grad()
    def predict_batch(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.model.predict(images, self.device)

    def predict_image(self, image: str | Path) -> dict[str, Any]:
        tensor = self.preprocessor(image).unsqueeze(0)
        prediction = self.predict_batch(tensor)
        score = float(prediction["score"][0])
        anomaly_map = prediction["anomaly_map"][0]
        return {
            "score": score,
            "label": "anomaly" if score >= self.threshold else "normal",
            "threshold": self.threshold,
            "anomaly_map": anomaly_map,
        }


def resolve_device(device: str | torch.device = "auto") -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def load_predictor(checkpoint_path: str | Path, device: str | torch.device = "auto", image_size: int | None = None):
    resolved_device = resolve_device(device)
    checkpoint = torch.load(checkpoint_path, map_location=resolved_device)
    model_type = checkpoint.get("model_type", "autoencoder")
    if model_type == "autoencoder":
        return AutoEncoderPredictor(checkpoint, resolved_device, image_size=image_size)
    if model_type == "patchcore":
        return PatchCorePredictor(checkpoint, resolved_device, image_size=image_size)
    raise ValueError(f"Unsupported checkpoint model_type: {model_type}")


def iter_images(path: str | Path) -> list[Path]:
    input_path = Path(path)
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(child for child in input_path.rglob("*") if child.suffix.lower() in IMAGE_EXTENSIONS)
    raise FileNotFoundError(f"Input path not found: {input_path}")


def run_inference(
    checkpoint: str | Path,
    input_path: str | Path,
    output_dir: str | Path,
    device: str = "auto",
    image_size: int | None = None,
) -> list[dict[str, Any]]:
    predictor = load_predictor(checkpoint, device=device, image_size=image_size)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for image_path in iter_images(input_path):
        result = predictor.predict_image(image_path)
        stem = image_path.stem
        original = load_image(image_path)
        heatmap = make_heatmap(result["anomaly_map"])
        overlay = overlay_heatmap(original, result["anomaly_map"])
        mask = make_binary_mask(result["anomaly_map"], result["threshold"])

        heatmap_path = save_image(heatmap, output_root / f"{stem}_heatmap.png")
        overlay_path = save_image(overlay, output_root / f"{stem}_overlay.png")
        mask_path = save_image(mask, output_root / f"{stem}_mask.png")

        row = {
            "path": str(image_path),
            "score": result["score"],
            "threshold": result["threshold"],
            "prediction": result["label"],
            "heatmap_path": str(heatmap_path),
            "overlay_path": str(overlay_path),
            "mask_path": str(mask_path),
        }
        rows.append(row)

    with (output_root / "predictions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()) if rows else ["path"])
        writer.writeheader()
        writer.writerows(rows)
    with (output_root / "predictions.json").open("w", encoding="utf-8") as file:
        json.dump(rows, file, indent=2)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DefectVision-AD inference.")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/autoencoder.pt")
    parser.add_argument("--input", required=True, help="Image file or directory.")
    parser.add_argument("--output-dir", default="outputs/heatmaps")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--image-size", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = run_inference(
        checkpoint=args.checkpoint,
        input_path=args.input,
        output_dir=args.output_dir,
        device=args.device,
        image_size=args.image_size,
    )
    for row in rows:
        print(f"{row['prediction']} score={row['score']:.6f} path={row['path']}")


if __name__ == "__main__":
    main()
