from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import ensure_parent, load_config
from src.datasets import build_dataset
from src.models import ConvAutoEncoder, PatchCoreWrapper, reconstruction_error_map


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(name: str | None) -> torch.device:
    if name in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def make_dataloader(config: dict[str, Any], split: str, shuffle: bool) -> DataLoader:
    dataset = build_dataset(config, split=split)
    dataset_config = config.get("dataset", {})
    return DataLoader(
        dataset,
        batch_size=int(dataset_config.get("batch_size", 8)),
        shuffle=shuffle,
        num_workers=int(dataset_config.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )


def image_scores(model: nn.Module, dataloader: DataLoader, device: torch.device) -> list[float]:
    model.eval()
    scores: list[float] = []
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            reconstructions = model(images)
            error_map = reconstruction_error_map(images, reconstructions)
            scores.extend(error_map.flatten(1).mean(dim=1).detach().cpu().tolist())
    return scores


def train_autoencoder(config: dict[str, Any], device: torch.device) -> Path:
    train_loader = make_dataloader(config, split="train", shuffle=True)
    threshold_loader = make_dataloader(config, split="train", shuffle=False)

    model_config = config.get("model", {})
    train_config = config.get("train", {})
    output_config = config.get("outputs", {})

    model = ConvAutoEncoder(
        in_channels=int(model_config.get("in_channels", 3)),
        base_channels=int(model_config.get("base_channels", 32)),
        latent_channels=int(model_config.get("latent_channels", 256)),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_config.get("learning_rate", 1e-3)),
        weight_decay=float(train_config.get("weight_decay", 1e-5)),
    )
    criterion = nn.MSELoss()
    epochs = int(train_config.get("epochs", 20))

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", leave=False)
        for batch in progress:
            images = batch["image"].to(device)
            optimizer.zero_grad(set_to_none=True)
            reconstructions = model(images)
            loss = criterion(reconstructions, images)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * images.size(0)
            progress.set_postfix(loss=f"{loss.item():.5f}")

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"epoch={epoch} train_loss={epoch_loss:.6f}")

    train_scores = image_scores(model, threshold_loader, device)
    percentile = float(train_config.get("threshold_percentile", 95))
    threshold = float(np.percentile(train_scores, percentile))

    checkpoint_path = ensure_parent(output_config.get("checkpoint", "outputs/checkpoints/autoencoder.pt"))
    torch.save(
        {
            "model_type": "autoencoder",
            "model_state": model.state_dict(),
            "threshold": threshold,
            "train_scores": train_scores,
            "config": config,
        },
        checkpoint_path,
    )
    print(f"saved_checkpoint={checkpoint_path}")
    print(f"threshold_p{percentile:g}={threshold:.8f}")
    return checkpoint_path


def train_patchcore(config: dict[str, Any], device: torch.device) -> Path:
    train_loader = make_dataloader(config, split="train", shuffle=False)
    model_config = config.get("model", {})
    output_config = config.get("outputs", {})
    train_config = config.get("train", {})

    model = PatchCoreWrapper(
        backbone=model_config.get("backbone", "resnet18"),
        pretrained=bool(model_config.get("pretrained", True)),
        max_memory_patches=int(model_config.get("max_memory_patches", 50000)),
    )
    model.fit(train_loader, device)

    train_scores: list[float] = []
    for batch in tqdm(train_loader, desc="threshold", leave=False):
        predictions = model.predict(batch["image"], device)
        train_scores.extend(predictions["score"].tolist())

    percentile = float(train_config.get("threshold_percentile", 95))
    threshold = float(np.percentile(train_scores, percentile))
    checkpoint_path = ensure_parent(output_config.get("checkpoint", "outputs/checkpoints/patchcore.pt"))
    model.save(checkpoint_path, threshold, config)
    print(f"saved_checkpoint={checkpoint_path}")
    print(f"threshold_p{percentile:g}={threshold:.8f}")
    return checkpoint_path


def apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    dataset_config = config.setdefault("dataset", {})
    if args.data_root:
        dataset_config["root"] = args.data_root
    if args.category:
        dataset_config["category"] = args.category
    if args.epochs is not None:
        config.setdefault("train", {})["epochs"] = args.epochs
    if args.checkpoint:
        config.setdefault("outputs", {})["checkpoint"] = args.checkpoint
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DefectVision-AD models.")
    parser.add_argument("--config", default="configs/model_autoencoder.yaml", help="YAML config path.")
    parser.add_argument("--data-root", help="Override dataset root.")
    parser.add_argument("--category", help="Override dataset category.")
    parser.add_argument("--epochs", type=int, help="Override epoch count for AutoEncoder.")
    parser.add_argument("--checkpoint", help="Override checkpoint output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = apply_cli_overrides(load_config(args.config), args)
    seed = int(config.get("experiment", {}).get("seed", 42))
    set_seed(seed)
    device = get_device(config.get("experiment", {}).get("device", "auto"))
    model_type = config.get("model", {}).get("type", "autoencoder").lower()

    if model_type == "autoencoder":
        train_autoencoder(config, device)
    elif model_type == "patchcore":
        train_patchcore(config, device)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


if __name__ == "__main__":
    main()
