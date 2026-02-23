from __future__ import annotations

import numpy as np

LUMA_WEIGHTS_REC709 = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    rgb = np.clip(rgb, 0.0, 1.0)
    return np.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4,
    ).astype(np.float32, copy=False)


def linear_to_srgb(rgb_linear: np.ndarray) -> np.ndarray:
    rgb_linear = np.clip(rgb_linear, 0.0, 1.0)
    return np.where(
        rgb_linear <= 0.0031308,
        rgb_linear * 12.92,
        1.055 * np.power(rgb_linear, 1.0 / 2.4) - 0.055,
    ).astype(np.float32, copy=False)


def rec709_luminance(rgb_linear: np.ndarray) -> np.ndarray:
    return np.tensordot(rgb_linear, LUMA_WEIGHTS_REC709, axes=([2], [0])).astype(np.float32, copy=False)


def robust_sample(image: np.ndarray, *, max_pixels: int = 900_000) -> np.ndarray:
    height, width = image.shape[:2]
    total = height * width
    if total <= max_pixels:
        return image
    stride = int(np.ceil(np.sqrt(total / max_pixels)))
    return image[::stride, ::stride]


def smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    if edge1 <= edge0:
        return np.zeros_like(x, dtype=np.float32)
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32, copy=False)
