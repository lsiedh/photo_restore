from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .color_math import linear_to_srgb, rec709_luminance, srgb_to_linear
from .models import ImageContext


@dataclass(frozen=True)
class FlatFieldConfig:
    mode: str = "on"
    radius_spec: str = "320px"
    strength: float = 0.35
    gain_min: float = 0.90
    gain_max: float = 1.12
    downsample_max_dim: int = 1024


class FlatFieldStage:
    name = "flatfield-normalization"

    def __init__(self, config: FlatFieldConfig) -> None:
        self.config = config

    def process(self, context: ImageContext) -> ImageContext:
        if context.image_f32 is None:
            raise RuntimeError("Flat-field stage expected normalized image data.")

        if self.config.mode == "off":
            context.metadata.update(
                {
                    "flatfield_applied": False,
                    "flatfield_skipped_reason": "flatfield-off",
                    "flatfield_radius_spec": str(self.config.radius_spec),
                }
            )
            return context

        corrected, metadata = apply_flatfield(context.image_f32, self.config)
        context.image_f32 = corrected
        context.metadata.update(metadata)
        return context


def apply_flatfield(image_srgb: np.ndarray, config: FlatFieldConfig) -> tuple[np.ndarray, dict[str, Any]]:
    source = np.clip(image_srgb, 0.0, 1.0).astype(np.float32, copy=False)
    linear = srgb_to_linear(source)
    luminance_before = rec709_luminance(linear)

    radius_px = resolve_radius_px(
        spec=config.radius_spec,
        shape=luminance_before.shape,
        default_px=320.0,
    )
    if radius_px < 8.0:
        metadata = {
            "flatfield_applied": False,
            "flatfield_skipped_reason": "flatfield-radius-too-small",
            "flatfield_radius_spec": str(config.radius_spec),
            "flatfield_radius_px_effective": float(radius_px),
        }
        return source, metadata

    ds = choose_downsample_factor(luminance_before.shape, max_dim=max(128, int(config.downsample_max_dim)))
    luminance_small = luminance_before[::ds, ::ds]
    sigma_small = max(1.0, float(radius_px) / max(3.0 * ds, 1.0))
    field_small = gaussian_blur_2d(luminance_small, sigma=sigma_small)
    field = upsample_nearest(field_small, target_shape=luminance_before.shape)
    field = np.maximum(field, 1e-6)

    target = float(np.median(field))
    raw_gain = target / field
    clamped_gain = np.clip(raw_gain, float(config.gain_min), float(config.gain_max))
    strength = float(np.clip(config.strength, 0.0, 1.0))
    gain = 1.0 + (strength * (clamped_gain - 1.0))

    corrected_linear = np.clip(linear * gain[:, :, None], 0.0, 1.0)
    corrected_srgb = linear_to_srgb(corrected_linear)
    luminance_after = rec709_luminance(corrected_linear)

    metadata: dict[str, Any] = {
        "flatfield_applied": True,
        "flatfield_radius_spec": str(config.radius_spec),
        "flatfield_radius_px_effective": float(radius_px),
        "flatfield_downsample_factor": int(ds),
        "flatfield_gain_min": float(np.min(gain)),
        "flatfield_gain_max": float(np.max(gain)),
        "flatfield_gain_mean": float(np.mean(gain)),
        "flatfield_strength": strength,
        "flatfield_luminance_mean_before": float(np.mean(luminance_before)),
        "flatfield_luminance_mean_after": float(np.mean(luminance_after)),
    }
    return corrected_srgb.astype(np.float32, copy=False), metadata


def resolve_radius_px(*, spec: str | float | int, shape: tuple[int, int], default_px: float) -> float:
    h, w = shape
    min_dim = float(max(1, min(h, w)))
    if isinstance(spec, (int, float)):
        return float(max(1.0, spec))

    text = str(spec).strip().lower()
    if not text:
        return float(default_px)
    if text.endswith("px"):
        try:
            return float(max(1.0, float(text[:-2].strip())))
        except Exception:
            return float(default_px)
    if text.endswith("%"):
        try:
            pct = float(text[:-1].strip())
        except Exception:
            return float(default_px)
        pct = float(np.clip(pct, 0.1, 100.0))
        return float(max(1.0, min_dim * (pct / 100.0)))

    try:
        return float(max(1.0, float(text)))
    except Exception:
        return float(default_px)


def choose_downsample_factor(shape: tuple[int, int], *, max_dim: int) -> int:
    h, w = shape
    longest = max(h, w)
    if longest <= max_dim:
        return 1
    return int(np.ceil(longest / max(max_dim, 1)))


def upsample_nearest(image: np.ndarray, *, target_shape: tuple[int, int]) -> np.ndarray:
    h, w = target_shape
    if image.shape == (h, w):
        return image.astype(np.float32, copy=False)

    y_idx = np.minimum(((np.arange(h) * image.shape[0]) // max(h, 1)).astype(np.int64), image.shape[0] - 1)
    x_idx = np.minimum(((np.arange(w) * image.shape[1]) // max(w, 1)).astype(np.int64), image.shape[1] - 1)
    out = image[y_idx[:, None], x_idx[None, :]]
    return out.astype(np.float32, copy=False)


def gaussian_blur_2d(image: np.ndarray, *, sigma: float) -> np.ndarray:
    sigma = float(max(0.0, sigma))
    if sigma < 1e-6:
        return image.astype(np.float32, copy=False)
    kernel = gaussian_kernel_1d(sigma)
    tmp = convolve_reflect_1d(image.astype(np.float32, copy=False), kernel, axis=1)
    return convolve_reflect_1d(tmp, kernel, axis=0)


def gaussian_kernel_1d(sigma: float) -> np.ndarray:
    radius = max(1, int(round(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(x * x) / (2.0 * sigma * sigma))
    kernel /= np.sum(kernel)
    return kernel.astype(np.float32, copy=False)


def convolve_reflect_1d(image: np.ndarray, kernel: np.ndarray, *, axis: int) -> np.ndarray:
    radius = kernel.size // 2
    if axis == 0:
        padded = np.pad(image, ((radius, radius), (0, 0)), mode="reflect")
        out = np.zeros_like(image, dtype=np.float32)
        for i, w in enumerate(kernel):
            out += float(w) * padded[i : i + image.shape[0], :]
        return out

    padded = np.pad(image, ((0, 0), (radius, radius)), mode="reflect")
    out = np.zeros_like(image, dtype=np.float32)
    for i, w in enumerate(kernel):
        out += float(w) * padded[:, i : i + image.shape[1]]
    return out
