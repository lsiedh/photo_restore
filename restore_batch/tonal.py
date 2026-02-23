from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .color_math import linear_to_srgb, rec709_luminance, smoothstep, srgb_to_linear
from .models import ImageContext


@dataclass(frozen=True)
class TonalNormalizationConfig:
    black_percentile: float = 0.5
    white_percentile: float = 99.5
    border_trim_fraction: float = 0.04
    strength: float = 0.65
    contrast_strength: float = 0.25
    near_grayscale_strength: float = 1.0
    near_grayscale_contrast_strength: float = 0.30
    near_grayscale_white_percentile: float = 99.2
    near_grayscale_min_effective_strength: float = 0.50
    dynamic_range_good: float = 0.82
    clip_fraction_threshold: float = 0.001
    near_grayscale_neutralize: bool = True
    max_chroma_boost: float = 0.12
    max_guardrail_iterations: int = 8
    min_strength_floor: float = 0.10
    fallback_percentile_expand: float = 0.50


class TonalNormalizationStage:
    name = "global-tonal-normalization"

    def __init__(self, config: TonalNormalizationConfig) -> None:
        self.config = config

    def process(self, context: ImageContext) -> ImageContext:
        if context.image_f32 is None:
            raise RuntimeError("Tonal stage expected image data.")

        image_type = str(context.metadata.get("image_type", "true-color"))
        is_color = image_type == "true-color"
        corrected, metadata = apply_global_tonal_normalization(
            image_srgb=context.image_f32,
            config=self.config,
            is_color_image=is_color,
        )
        context.image_f32 = corrected
        context.metadata.update(metadata)
        return context


def apply_global_tonal_normalization(
    *,
    image_srgb: np.ndarray,
    config: TonalNormalizationConfig,
    is_color_image: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    linear = srgb_to_linear(image_srgb)
    luminance_before = rec709_luminance(linear)
    stats_region = center_trim_region(luminance_before, trim_fraction=config.border_trim_fraction)

    p1 = float(np.percentile(stats_region, 1.0))
    p99 = float(np.percentile(stats_region, 99.0))
    dynamic_range = max(0.0, p99 - p1)
    black_percentile = float(config.black_percentile)
    white_percentile = float(config.white_percentile)

    if is_color_image:
        workflow = "color-wb-luminance"
        luminance_mapping = "adaptive-luminance-remap"
        base_strength = float(np.clip(config.strength, 0.0, 1.0))
        contrast_strength = float(np.clip(config.contrast_strength, 0.0, 1.0))
    else:
        workflow = "near-grayscale-percentile-remap"
        luminance_mapping = "full-range-remap"
        base_strength = float(np.clip(config.near_grayscale_strength, 0.0, 1.0))
        contrast_strength = float(np.clip(config.near_grayscale_contrast_strength, 0.0, 1.0))
        white_percentile = float(config.near_grayscale_white_percentile)

    black_point, white_point = robust_black_white_points(
        stats_region,
        black_percentile=black_percentile,
        white_percentile=white_percentile,
    )

    clip_threshold = max(0.0, float(config.clip_fraction_threshold))
    effective_strength = adaptive_strength(
        base_strength=base_strength,
        dynamic_range=dynamic_range,
        dynamic_range_good=float(np.clip(config.dynamic_range_good, 0.1, 1.0)),
    )
    near_grayscale_min_effective_strength = 0.0
    near_grayscale_strength_floor_applied = False
    if not is_color_image:
        near_grayscale_min_effective_strength = float(np.clip(config.near_grayscale_min_effective_strength, 0.0, 1.0))
        floor_target = min(base_strength, near_grayscale_min_effective_strength)
        if effective_strength < floor_target:
            effective_strength = floor_target
            near_grayscale_strength_floor_applied = True

    guardrail_iterations = 0
    guardrail_triggers: list[str] = []
    hard_guardrail_applied = False
    fallback_used = False
    fallback_mode: str | None = None
    floor_strength_attempted = 0.0
    fallback_black_percentile = None
    fallback_white_percentile = None
    output_linear = linear.copy()
    target_luma = luminance_before.copy()
    clip_low = 0.0
    clip_high = 0.0

    while True:
        target_luma = luminance_target(
            luminance=luminance_before,
            black_point=black_point,
            white_point=white_point,
            strength=effective_strength,
            contrast_strength=contrast_strength,
        )
        output_linear = recombine_from_luminance(
            source_linear=linear,
            source_luminance=luminance_before,
            target_luminance=target_luma,
            is_color_image=is_color_image,
            near_grayscale_neutralize=config.near_grayscale_neutralize,
            max_chroma_boost=config.max_chroma_boost,
        )
        clip_low = float(np.mean(output_linear < 0.0))
        clip_high = float(np.mean(output_linear > 1.0))
        if (clip_low + clip_high) <= clip_threshold:
            break
        if guardrail_iterations >= config.max_guardrail_iterations:
            break
        if "clip-fraction-limit" not in guardrail_triggers:
            guardrail_triggers.append("clip-fraction-limit")
        effective_strength *= 0.72
        guardrail_iterations += 1

    if (clip_low + clip_high) > clip_threshold:
        floor_strength_attempted = float(np.clip(min(base_strength, config.min_strength_floor), 0.0, 1.0))
        if floor_strength_attempted > 1e-6:
            fallback_black_percentile, fallback_white_percentile = widened_percentiles(
                black_percentile=black_percentile,
                white_percentile=white_percentile,
                expand_amount=float(config.fallback_percentile_expand),
            )
            fallback_black_point, fallback_white_point = robust_black_white_points(
                stats_region,
                black_percentile=fallback_black_percentile,
                white_percentile=fallback_white_percentile,
            )
            fallback_target_luma = luminance_target(
                luminance=luminance_before,
                black_point=fallback_black_point,
                white_point=fallback_white_point,
                strength=floor_strength_attempted,
                contrast_strength=contrast_strength,
            )
            fallback_output_linear = recombine_from_luminance(
                source_linear=linear,
                source_luminance=luminance_before,
                target_luminance=fallback_target_luma,
                is_color_image=is_color_image,
                near_grayscale_neutralize=config.near_grayscale_neutralize,
                max_chroma_boost=config.max_chroma_boost,
            )
            fallback_clip_low = float(np.mean(fallback_output_linear < 0.0))
            fallback_clip_high = float(np.mean(fallback_output_linear > 1.0))
            if (fallback_clip_low + fallback_clip_high) <= clip_threshold:
                fallback_used = True
                fallback_mode = "min-strength-widened-percentiles"
                effective_strength = floor_strength_attempted
                black_point = fallback_black_point
                white_point = fallback_white_point
                target_luma = fallback_target_luma
                output_linear = fallback_output_linear
                clip_low = fallback_clip_low
                clip_high = fallback_clip_high

    if (clip_low + clip_high) > clip_threshold:
        hard_guardrail_applied = True
        if "hard-guardrail-disable" not in guardrail_triggers:
            guardrail_triggers.append("hard-guardrail-disable")
        effective_strength = 0.0
        target_luma = luminance_before.copy()
        output_linear = recombine_from_luminance(
            source_linear=linear,
            source_luminance=luminance_before,
            target_luminance=target_luma,
            is_color_image=is_color_image,
            near_grayscale_neutralize=config.near_grayscale_neutralize,
            max_chroma_boost=config.max_chroma_boost,
        )
        clip_low = float(np.mean(output_linear < 0.0))
        clip_high = float(np.mean(output_linear > 1.0))

    output_linear = np.clip(output_linear, 0.0, 1.0)
    luminance_after = rec709_luminance(output_linear)
    output_srgb = linear_to_srgb(output_linear)

    metadata: dict[str, Any] = {
        "tonal_stage_applied": True,
        "tonal_workflow": workflow,
        "tonal_luminance_mapping": luminance_mapping,
        "tonal_black_white_estimation": "percentile-clipping",
        "tonal_black_white_stats_region": "center-trim-luminance",
        "tonal_black_white_computed_on": "luminance-only",
        "tonal_tone_mapping_applied_to": "luminance-primary",
        "tonal_near_grayscale_neutralized": (not is_color_image) and bool(config.near_grayscale_neutralize),
        "tonal_black_percentile": black_percentile,
        "tonal_white_percentile": white_percentile,
        "tonal_black_point": float(black_point),
        "tonal_white_point": float(white_point),
        "tonal_dynamic_range_p1_p99_before": float(dynamic_range),
        "tonal_strength_base": base_strength,
        "tonal_strength_effective": float(effective_strength),
        "tonal_contrast_strength": contrast_strength,
        "tonal_near_grayscale_strength_target": float(np.clip(config.near_grayscale_strength, 0.0, 1.0)),
        "tonal_near_grayscale_contrast_target": float(np.clip(config.near_grayscale_contrast_strength, 0.0, 1.0)),
        "tonal_near_grayscale_white_percentile_target": float(config.near_grayscale_white_percentile),
        "tonal_near_grayscale_min_effective_strength_target": near_grayscale_min_effective_strength,
        "tonal_near_grayscale_strength_floor_applied": near_grayscale_strength_floor_applied,
        "tonal_white_percentile_global_target": float(config.white_percentile),
        "tonal_guardrail_iterations": int(guardrail_iterations),
        "tonal_guardrail_triggers": guardrail_triggers,
        "tonal_guardrail_hard_limit_applied": hard_guardrail_applied,
        "tonal_guardrail_fallback_used": fallback_used,
        "tonal_guardrail_fallback_mode": fallback_mode,
        "tonal_strength_floor_attempted": float(floor_strength_attempted),
        "tonal_fallback_black_percentile": fallback_black_percentile,
        "tonal_fallback_white_percentile": fallback_white_percentile,
        "tonal_clip_fraction_low_preclip": clip_low,
        "tonal_clip_fraction_high_preclip": clip_high,
        "tonal_clip_fraction_total_preclip": clip_low + clip_high,
        "tonal_luminance_mean_before": float(np.mean(luminance_before)),
        "tonal_luminance_mean_after": float(np.mean(luminance_after)),
        "tonal_luminance_p5_before": float(np.percentile(luminance_before, 5.0)),
        "tonal_luminance_p5_after": float(np.percentile(luminance_after, 5.0)),
        "tonal_luminance_p95_before": float(np.percentile(luminance_before, 95.0)),
        "tonal_luminance_p95_after": float(np.percentile(luminance_after, 95.0)),
    }
    return output_srgb, metadata


def center_trim_region(luminance: np.ndarray, *, trim_fraction: float) -> np.ndarray:
    trim = float(np.clip(trim_fraction, 0.0, 0.2))
    height, width = luminance.shape
    trim_h = int(round(height * trim))
    trim_w = int(round(width * trim))
    if (height - (2 * trim_h)) < 32 or (width - (2 * trim_w)) < 32:
        return luminance
    return luminance[trim_h : height - trim_h, trim_w : width - trim_w]


def robust_black_white_points(
    luminance: np.ndarray,
    *,
    black_percentile: float,
    white_percentile: float,
) -> tuple[float, float]:
    black_p = float(np.clip(black_percentile, 0.0, 15.0))
    white_p = float(np.clip(white_percentile, 85.0, 100.0))
    if white_p <= black_p:
        white_p = black_p + 1.0

    black = float(np.percentile(luminance, black_p))
    white = float(np.percentile(luminance, white_p))

    if (white - black) < 0.05:
        low = float(np.percentile(luminance, max(0.0, black_p * 0.5)))
        high = float(np.percentile(luminance, min(100.0, white_p + ((100.0 - white_p) * 0.5))))
        black = min(black, low)
        white = max(white, high)

    black = float(np.clip(black, 0.0, 0.98))
    white = float(np.clip(white, 0.02, 1.0))
    if white <= black:
        mid = (black + white) * 0.5
        black = max(0.0, mid - 0.02)
        white = min(1.0, mid + 0.02)
    return black, white


def adaptive_strength(*, base_strength: float, dynamic_range: float, dynamic_range_good: float) -> float:
    # Lower strength when image already has healthy dynamic range.
    gap = np.clip((dynamic_range_good - dynamic_range) / max(dynamic_range_good, 1e-6), 0.0, 1.0)
    return float(np.clip(base_strength * (0.3 + (0.7 * gap)), 0.0, 1.0))


def widened_percentiles(*, black_percentile: float, white_percentile: float, expand_amount: float) -> tuple[float, float]:
    expand = float(np.clip(expand_amount, 0.0, 1.0))
    black = max(0.0, black_percentile * (1.0 - (0.5 * expand)))
    white = min(100.0, white_percentile + ((100.0 - white_percentile) * (0.5 * expand)))
    if white <= black:
        white = black + 1.0
    return black, white


def luminance_target(
    *,
    luminance: np.ndarray,
    black_point: float,
    white_point: float,
    strength: float,
    contrast_strength: float,
) -> np.ndarray:
    span = max(white_point - black_point, 1e-6)
    normalized = np.clip((luminance - black_point) / span, 0.0, 1.0)
    curved = normalized + (contrast_strength * (smoothstep(0.0, 1.0, normalized) - normalized))
    return (luminance + (strength * (curved - luminance))).astype(np.float32, copy=False)


def recombine_from_luminance(
    *,
    source_linear: np.ndarray,
    source_luminance: np.ndarray,
    target_luminance: np.ndarray,
    is_color_image: bool,
    near_grayscale_neutralize: bool,
    max_chroma_boost: float,
) -> np.ndarray:
    if not is_color_image and near_grayscale_neutralize:
        return np.repeat(target_luminance[:, :, None], 3, axis=2).astype(np.float32, copy=False)

    ratio = np.divide(
        target_luminance,
        np.maximum(source_luminance, 1e-7),
        out=np.ones_like(source_luminance),
        where=source_luminance > 1e-7,
    )
    corrected = source_linear * ratio[:, :, None]
    if is_color_image:
        corrected = limit_chroma_boost(
            original_linear=source_linear,
            corrected_linear=corrected,
            luminance_reference=target_luminance,
            max_chroma_boost=max_chroma_boost,
        )
    return corrected.astype(np.float32, copy=False)


def limit_chroma_boost(
    *,
    original_linear: np.ndarray,
    corrected_linear: np.ndarray,
    luminance_reference: np.ndarray,
    max_chroma_boost: float,
) -> np.ndarray:
    orig_chroma = np.max(original_linear, axis=2) - np.min(original_linear, axis=2)
    corr_chroma = np.max(corrected_linear, axis=2) - np.min(corrected_linear, axis=2)
    allowed = orig_chroma * (1.0 + max(0.0, float(max_chroma_boost)))
    ratio = np.divide(
        allowed,
        np.maximum(corr_chroma, 1e-7),
        out=np.ones_like(corr_chroma),
        where=corr_chroma > allowed,
    )
    ratio = np.clip(ratio, 0.0, 1.0)
    neutral = luminance_reference[:, :, None]
    return neutral + ((corrected_linear - neutral) * ratio[:, :, None])
