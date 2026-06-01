from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


class PatchCoreWrapper:
    """Small PatchCore-style feature memory bank.

    This wrapper keeps the project self-contained. If torchvision pretrained
    weights are available, use them; otherwise the code still runs with an
    untrained ResNet feature extractor for development smoke tests.
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        max_memory_patches: int = 50000,
    ) -> None:
        self.backbone = backbone
        self.pretrained = pretrained
        self.max_memory_patches = max_memory_patches
        self.extractor = self._build_extractor(backbone, pretrained)
        self.memory_bank: torch.Tensor | None = None

    @staticmethod
    def _build_extractor(backbone: str, pretrained: bool) -> nn.Module:
        if backbone != "resnet18":
            raise ValueError("Only resnet18 is supported by the lightweight PatchCore wrapper.")

        try:
            from torchvision.models import ResNet18_Weights, resnet18

            weights = ResNet18_Weights.DEFAULT if pretrained else None
            model = resnet18(weights=weights)
            return nn.Sequential(
                model.conv1,
                model.bn1,
                model.relu,
                model.maxpool,
                model.layer1,
                model.layer2,
            ).eval()
        except Exception:
            return nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ).eval()

    @torch.no_grad()
    def fit(self, dataloader, device: torch.device) -> None:
        self.extractor.to(device)
        patches: list[torch.Tensor] = []
        for batch in dataloader:
            images = batch["image"].to(device)
            features = self._extract(images)
            patches.append(features.cpu())

        if not patches:
            raise ValueError("No patches collected for PatchCore memory bank.")

        memory_bank = torch.cat(patches, dim=0)
        if len(memory_bank) > self.max_memory_patches:
            indices = torch.randperm(len(memory_bank))[: self.max_memory_patches]
            memory_bank = memory_bank[indices]
        self.memory_bank = memory_bank.contiguous()

    @torch.no_grad()
    def predict(self, images: torch.Tensor, device: torch.device, chunk_size: int = 4096) -> dict[str, torch.Tensor]:
        if self.memory_bank is None:
            raise RuntimeError("PatchCore memory bank is empty. Call fit() or load() first.")

        self.extractor.to(device)
        images = images.to(device)
        feature_map = self.extractor(images)
        batch_size, channels, feat_h, feat_w = feature_map.shape
        features = F.normalize(feature_map, dim=1)
        patches = features.permute(0, 2, 3, 1).reshape(-1, channels)

        memory_bank = self.memory_bank.to(device)
        min_distances: list[torch.Tensor] = []
        for chunk in patches.split(chunk_size):
            distances = torch.cdist(chunk, memory_bank)
            min_distances.append(distances.min(dim=1).values)

        patch_scores = torch.cat(min_distances, dim=0).view(batch_size, 1, feat_h, feat_w)
        anomaly_map = F.interpolate(patch_scores, size=images.shape[-2:], mode="bilinear", align_corners=False)
        scores = anomaly_map.flatten(1).max(dim=1).values
        return {"score": scores.detach().cpu(), "anomaly_map": anomaly_map.detach().cpu()}

    def _extract(self, images: torch.Tensor) -> torch.Tensor:
        feature_map = self.extractor(images)
        feature_map = F.normalize(feature_map, dim=1)
        return feature_map.permute(0, 2, 3, 1).reshape(-1, feature_map.shape[1])

    def save(self, path: str | Path, threshold: float, config: dict[str, Any]) -> None:
        if self.memory_bank is None:
            raise RuntimeError("Cannot save PatchCore model before fitting.")
        torch.save(
            {
                "model_type": "patchcore",
                "backbone": self.backbone,
                "pretrained": self.pretrained,
                "max_memory_patches": self.max_memory_patches,
                "memory_bank": self.memory_bank,
                "threshold": threshold,
                "config": config,
            },
            path,
        )

    @classmethod
    def load(cls, checkpoint: dict[str, Any]) -> "PatchCoreWrapper":
        model = cls(
            backbone=checkpoint.get("backbone", "resnet18"),
            pretrained=bool(checkpoint.get("pretrained", False)),
            max_memory_patches=int(checkpoint.get("max_memory_patches", 50000)),
        )
        model.memory_bank = checkpoint["memory_bank"].float().contiguous()
        return model
