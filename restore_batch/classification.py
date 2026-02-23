from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .color_math import rec709_luminance, robust_sample, srgb_to_linear
from .models import ImageContext


@dataclass(frozen=True)
class ClassificationConfig:
    threshold: float = 0.0012
    max_pixels: int = 900_000
    sepia_gate_enabled: bool = True
    sepia_hue_circular_var_max: float = 0.01
    sepia_channel_corr_min: float = 0.995
    sepia_sat_mean_max: float = 0.22
    sepia_sat_p90_max: float = 0.35


class ImageTypeClassificationStage:
    name = "classify-image-type"

    def __init__(self, config: ClassificationConfig) -> None:
        self.config = config

    def process(self, context: ImageContext) -> ImageContext:
        if context.image_f32 is None:
            raise RuntimeError("Classification stage expected normalized image data.")

        rgb = context.image_f32
        sample = robust_sample(rgb, max_pixels=self.config.max_pixels)
        linear = srgb_to_linear(sample)

        luma = rec709_luminance(linear)
        luma_low = float(np.percentile(luma, 1.0))
        luma_high = float(np.percentile(luma, 99.0))
        sum_rgb = np.sum(linear, axis=2)
        valid = (sum_rgb > 1e-6) & (luma >= luma_low) & (luma <= luma_high)

        if int(np.count_nonzero(valid)) < 128:
            valid = sum_rgb > 1e-6

        chrom_r = np.divide(linear[:, :, 0], sum_rgb, out=np.zeros_like(sum_rgb), where=valid)
        chrom_g = np.divide(linear[:, :, 1], sum_rgb, out=np.zeros_like(sum_rgb), where=valid)
        r_values = chrom_r[valid]
        g_values = chrom_g[valid]

        if r_values.size == 0:
            chroma_variance = 0.0
            chroma_iqr = 0.0
        else:
            chroma_variance = float(np.var(r_values) + np.var(g_values))
            chroma_distance = np.sqrt(
                (r_values - float(np.mean(r_values))) ** 2 + (g_values - float(np.mean(g_values))) ** 2
            )
            chroma_iqr = float(np.percentile(chroma_distance, 75.0) - np.percentile(chroma_distance, 25.0))
        score = chroma_variance

        sat_mean, sat_p90, hue_circular_var, corr_min = monochrome_cast_proxies(sample)
        sepia_gate_enabled = bool(self.config.sepia_gate_enabled)
        sepia_like = (
            sepia_gate_enabled
            and (hue_circular_var <= float(max(0.0, self.config.sepia_hue_circular_var_max)))
            and (corr_min >= float(np.clip(self.config.sepia_channel_corr_min, 0.0, 1.0)))
            and (sat_mean <= float(max(0.0, self.config.sepia_sat_mean_max)))
            and (sat_p90 <= float(max(0.0, self.config.sepia_sat_p90_max)))
        )

        near_gray_by_variance = score < self.config.threshold
        if near_gray_by_variance or sepia_like:
            classification = "near-grayscale"
            if near_gray_by_variance:
                classification_method = "chromaticity-variance"
            else:
                classification_method = "chromaticity-variance+sepia-monochrome-gate"
        else:
            classification = "true-color"
            classification_method = "chromaticity-variance"

        mean_rgb = np.mean(sample, axis=(0, 1)).astype(np.float64)

        metadata: dict[str, Any] = {
            "image_type": classification,
            "is_color_image": classification == "true-color",
            "classification_method": classification_method,
            "classification_threshold": float(self.config.threshold),
            "classification_score": float(score),
            "classification_chroma_variance": chroma_variance,
            "classification_chroma_iqr": chroma_iqr,
            "classification_sepia_gate_enabled": sepia_gate_enabled,
            "classification_sepia_like": bool(sepia_like),
            "classification_sepia_hue_circular_var": float(hue_circular_var),
            "classification_sepia_hue_circular_var_max": float(max(0.0, self.config.sepia_hue_circular_var_max)),
            "classification_sepia_channel_corr_min": float(corr_min),
            "classification_sepia_channel_corr_min_threshold": float(np.clip(self.config.sepia_channel_corr_min, 0.0, 1.0)),
            "classification_sepia_sat_mean": float(sat_mean),
            "classification_sepia_sat_mean_max": float(max(0.0, self.config.sepia_sat_mean_max)),
            "classification_sepia_sat_p90": float(sat_p90),
            "classification_sepia_sat_p90_max": float(max(0.0, self.config.sepia_sat_p90_max)),
            "classification_sample_pixels": int(sample.shape[0] * sample.shape[1]),
            "stats_mean_rgb": [float(mean_rgb[0]), float(mean_rgb[1]), float(mean_rgb[2])],
        }
        context.metadata.update(metadata)
        return context


def monochrome_cast_proxies(sample_srgb: np.ndarray) -> tuple[float, float, float, float]:
    rgb = np.clip(sample_srgb, 0.0, 1.0).astype(np.float32, copy=False)
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]

    rgb_max = np.max(rgb, axis=2)
    rgb_min = np.min(rgb, axis=2)
    delta = rgb_max - rgb_min
    saturation = np.divide(
        delta,
        np.maximum(rgb_max, 1e-8),
        out=np.zeros_like(delta),
        where=rgb_max > 1e-8,
    )
    sat_mean = float(np.mean(saturation))
    sat_p90 = float(np.percentile(saturation, 90.0))

    hue = np.zeros_like(rgb_max, dtype=np.float32)
    valid = delta > 1e-6
    r_idx = (rgb_max == r) & valid
    g_idx = (rgb_max == g) & valid
    b_idx = (rgb_max == b) & valid
    hue[r_idx] = np.mod((g[r_idx] - b[r_idx]) / np.maximum(delta[r_idx], 1e-8), 6.0)
    hue[g_idx] = ((b[g_idx] - r[g_idx]) / np.maximum(delta[g_idx], 1e-8)) + 2.0
    hue[b_idx] = ((r[b_idx] - g[b_idx]) / np.maximum(delta[b_idx], 1e-8)) + 4.0
    hue /= 6.0

    # Circular hue variance weighted by saturation above noise floor.
    hue_w = np.clip(saturation - 0.05, 0.0, None)
    hue_valid = hue_w > 0.0
    if int(np.count_nonzero(hue_valid)) == 0:
        hue_circular_var = 1.0
    else:
        angle = 2.0 * np.pi * hue[hue_valid]
        weight = hue_w[hue_valid]
        cos_sum = float(np.sum(weight * np.cos(angle)))
        sin_sum = float(np.sum(weight * np.sin(angle)))
        w_sum = float(np.sum(weight))
        resultant = np.sqrt((cos_sum * cos_sum) + (sin_sum * sin_sum)) / max(w_sum, 1e-8)
        hue_circular_var = float(np.clip(1.0 - resultant, 0.0, 1.0))

    # Monochrome casts keep channels highly correlated with luminance.
    y = ((0.299 * r) + (0.587 * g) + (0.114 * b)).reshape(-1)
    rr = safe_corrcoef(y, r.reshape(-1))
    rg = safe_corrcoef(y, g.reshape(-1))
    rb = safe_corrcoef(y, b.reshape(-1))
    corr_min = float(min(rr, rg, rb))
    return sat_mean, sat_p90, hue_circular_var, corr_min


def safe_corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    a_std = float(np.std(a))
    b_std = float(np.std(b))
    if a_std < 1e-12 or b_std < 1e-12:
        return 1.0
    c = float(np.corrcoef(a, b)[0, 1])
    if not np.isfinite(c):
        return 0.0
    return float(np.clip(c, -1.0, 1.0))
