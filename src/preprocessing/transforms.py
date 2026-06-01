from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _resample_bilinear() -> int:
    return getattr(getattr(Image, "Resampling", Image), "BILINEAR")


def _resample_nearest() -> int:
    return getattr(getattr(Image, "Resampling", Image), "NEAREST")


def as_size(size: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(size, int):
        return size, size
    if len(size) != 2:
        raise ValueError("Image size must be an int or a 2-item sequence.")
    return int(size[0]), int(size[1])


def load_image(image: str | Path | Image.Image | np.ndarray) -> Image.Image:
    """Load an image-like object as RGB PIL image."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, (str, Path)):
        return Image.open(image).convert("RGB")
    if isinstance(image, np.ndarray):
        array = image
        if array.ndim == 2:
            array = cv2.cvtColor(array, cv2.COLOR_GRAY2RGB)
        if array.ndim != 3 or array.shape[2] not in (3, 4):
            raise ValueError(f"Unsupported image array shape: {array.shape}")
        if array.shape[2] == 4:
            array = array[:, :, :3]
        return Image.fromarray(array.astype(np.uint8)).convert("RGB")
    raise TypeError(f"Unsupported image input type: {type(image)!r}")


def apply_cv_preprocessing(
    image: Image.Image,
    *,
    grayscale: bool = False,
    gaussian_blur: bool = False,
    canny: bool = False,
    hist_equalize: bool = False,
) -> Image.Image:
    """Apply optional OpenCV preprocessing and return an RGB PIL image."""
    array = np.asarray(image.convert("RGB"))

    if grayscale or canny:
        gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
        if hist_equalize:
            gray = cv2.equalizeHist(gray)
        if gaussian_blur:
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
        if canny:
            gray = cv2.Canny(gray, 80, 160)
        array = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    else:
        if hist_equalize:
            ycrcb = cv2.cvtColor(array, cv2.COLOR_RGB2YCrCb)
            ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
            array = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
        if gaussian_blur:
            array = cv2.GaussianBlur(array, (5, 5), 0)

    return Image.fromarray(array)


def tensor_to_uint8(
    tensor: torch.Tensor,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
) -> np.ndarray:
    """Convert a normalized CHW tensor to uint8 HWC RGB."""
    if tensor.ndim == 4:
        tensor = tensor[0]
    if tensor.ndim != 3:
        raise ValueError(f"Expected CHW tensor, got shape {tuple(tensor.shape)}")

    mean_tensor = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    std_tensor = torch.tensor(std, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    image = (tensor.detach() * std_tensor + mean_tensor).clamp(0, 1)
    image_uint8 = (image * 255.0).round().to(torch.uint8).permute(1, 2, 0).cpu()
    return np.array(image_uint8.tolist(), dtype=np.uint8)


def pil_to_chw_tensor(image: Image.Image) -> torch.Tensor:
    """Convert an RGB PIL image to a float CHW tensor without torch.from_numpy."""
    rgb = image.convert("RGB")
    width, height = rgb.size
    data = bytearray(rgb.tobytes())
    if hasattr(torch, "frombuffer"):
        tensor = torch.frombuffer(data, dtype=torch.uint8)
    else:
        tensor = torch.ByteTensor(torch.ByteStorage.from_buffer(data))
    return tensor.view(height, width, 3).permute(2, 0, 1).float().div(255.0).contiguous()


def pil_mask_to_tensor(mask: Image.Image) -> torch.Tensor:
    gray = mask.convert("L")
    width, height = gray.size
    data = bytearray(gray.tobytes())
    if hasattr(torch, "frombuffer"):
        tensor = torch.frombuffer(data, dtype=torch.uint8)
    else:
        tensor = torch.ByteTensor(torch.ByteStorage.from_buffer(data))
    return tensor.view(height, width).gt(0).float().unsqueeze(0).contiguous()


@dataclass(slots=True)
class ImagePreprocessor:
    image_size: int | Sequence[int] = 256
    mean: Sequence[float] = IMAGENET_MEAN
    std: Sequence[float] = IMAGENET_STD
    grayscale: bool = False
    gaussian_blur: bool = False
    canny: bool = False
    hist_equalize: bool = False
    augment: bool = False

    def __call__(self, image: str | Path | Image.Image | np.ndarray) -> torch.Tensor:
        pil_image = load_image(image)
        height, width = as_size(self.image_size)
        pil_image = pil_image.resize((width, height), _resample_bilinear())

        if self.augment:
            pil_image = self._augment(pil_image)

        pil_image = apply_cv_preprocessing(
            pil_image,
            grayscale=self.grayscale,
            gaussian_blur=self.gaussian_blur,
            canny=self.canny,
            hist_equalize=self.hist_equalize,
        )

        tensor = pil_to_chw_tensor(pil_image)
        mean = torch.tensor(self.mean, dtype=tensor.dtype).view(-1, 1, 1)
        std = torch.tensor(self.std, dtype=tensor.dtype).view(-1, 1, 1)
        return ((tensor - mean) / std).contiguous()

    def mask(self, mask: str | Path | Image.Image | np.ndarray) -> torch.Tensor:
        pil_mask = load_image(mask).convert("L")
        height, width = as_size(self.image_size)
        pil_mask = pil_mask.resize((width, height), _resample_nearest())
        return pil_mask_to_tensor(pil_mask)

    def empty_mask(self) -> torch.Tensor:
        height, width = as_size(self.image_size)
        return torch.zeros((1, height, width), dtype=torch.float32)

    def to_uint8(self, tensor: torch.Tensor) -> np.ndarray:
        return tensor_to_uint8(tensor, self.mean, self.std)

    @staticmethod
    def _augment(image: Image.Image) -> Image.Image:
        if random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        return image
