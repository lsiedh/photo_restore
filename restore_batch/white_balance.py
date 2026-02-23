from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .color_math import linear_to_srgb, rec709_luminance, srgb_to_linear
from .models import ImageContext

LUMA_R = 0.2126
LUMA_G = 0.7152
LUMA_B = 0.0722


@dataclass(frozen=True)
class WhiteBalanceConfig:
    method: str = "shades-of-gray"
    strength: float = 0.6
    max_gain: float = 1.25
    shades_of_gray_p: float = 6.0
    gray_edge_sigma: float = 1.0
    min_valid_pixels: int = 1024
    confidence_reduce_threshold: float = 0.45
    confidence_skip_threshold: float = 0.08
    border_band: float = 0.08
    border_texture_max: float = 0.020
    border_neutrality_max: float = 0.045
    border_clip_max: float = 0.004
    border_warm_bias_max: float = 0.035
    fallback_percentile_low: float = 2.0
    fallback_percentile_high: float = 98.0
    clipped_exclusion_threshold: float = 0.995
    clip_fraction_threshold: float = 0.001
    max_guardrail_iterations: int = 10
    max_hue_rotation_deg: float = 18.0
    max_saturation_gain: float = 1.20
    max_luminance_scale: float = 1.20
    chroma_bias_correction: str = "on"
    chroma_bias_radius_spec: str = "16%"
    chroma_bias_cap: float = 0.020
    chroma_bias_strength: float = 0.70
    max_chroma_boost: float = 0.15


class WhiteBalanceStage:
    name = "white-balance"

    def __init__(self, config: WhiteBalanceConfig) -> None:
        self.config = config

    def process(self, context: ImageContext) -> ImageContext:
        if context.image_f32 is None:
            raise RuntimeError("White balance stage expected normalized image data.")

        baseline = baseline_luminance_metadata(context.image_f32)
        mode_meta = normalize_white_balance_mode(self.config.method)

        classification = str(context.metadata.get("image_type", "true-color"))
        if classification != "true-color":
            context.metadata.update(
                {
                    **baseline,
                    "white_balance_method": mode_meta["effective_mode"],
                    "white_balance_cli_mode": mode_meta["cli_mode"],
                    "white_balance_effective_mode": mode_meta["effective_mode"],
                    "white_balance_mode_mapped": mode_meta["mode_mapped"],
                    "white_balance_mode_mapping_reason": mode_meta["mapping_reason"],
                    "white_balance_applied": False,
                    "white_balance_skipped_reason": "non-color-image",
                    "white_balance_reference_method": "none",
                    "white_balance_border_used": False,
                    "white_balance_border_candidate_count": 0,
                    "white_balance_border_region_stats": [],
                    "white_balance_border_reject_reasons": [],
                    "white_balance_illuminant_estimate": [1.0, 1.0, 1.0],
                    "white_balance_estimated_illuminant": [1.0, 1.0, 1.0],
                    "white_balance_channel_scales_raw": [1.0, 1.0, 1.0],
                    "white_balance_channel_scales_applied": [1.0, 1.0, 1.0],
                    "white_balance_global_transform_type": "identity",
                    "white_balance_global_transform_values": {"rgb_scales": [1.0, 1.0, 1.0], "luminance_scalar": 1.0},
                    "white_balance_percentile_exclusion_low": float(self.config.fallback_percentile_low),
                    "white_balance_percentile_exclusion_high": float(self.config.fallback_percentile_high),
                    "white_balance_clipped_exclusion_threshold": float(self.config.clipped_exclusion_threshold),
                    "white_balance_estimator_reliability": {},
                    "white_balance_estimator_weights": {},
                    "white_balance_estimator_illuminants": {},
                    "white_balance_consensus_agreement": None,
                    "white_balance_confidence": 0.0,
                    "white_balance_valid_pixels": 0,
                    "white_balance_valid_fraction": 0.0,
                    "white_balance_sampling_mode": "non-color-skip",
                    "white_balance_sampling_fallback_used": False,
                    "white_balance_strength_requested": float(np.clip(self.config.strength, 0.0, 1.0)),
                    "white_balance_strength_effective": 0.0,
                    "white_balance_strength": 0.0,
                    "white_balance_max_gain": float(max(1.0, self.config.max_gain)),
                    "white_balance_shades_of_gray_p": float(max(1.0, self.config.shades_of_gray_p)),
                    "white_balance_gray_edge_sigma": float(max(0.0, self.config.gray_edge_sigma)),
                    "white_balance_preclip_fraction": 0.0,
                    "white_balance_preclip_fraction_low": 0.0,
                    "white_balance_preclip_fraction_high": 0.0,
                    "white_balance_guardrail_iterations": 0,
                    "white_balance_guardrail_triggers": [],
                    "white_balance_max_hue_rotation_deg": float(self.config.max_hue_rotation_deg),
                    "white_balance_observed_hue_rotation_deg": 0.0,
                    "white_balance_max_saturation_gain": float(self.config.max_saturation_gain),
                    "white_balance_observed_saturation_gain": 1.0,
                    "white_balance_global_luminance_scalar": 1.0,
                    "white_balance_adaptation_space": "none",
                    "white_balance_adaptation_scales_raw": [1.0, 1.0, 1.0],
                    "white_balance_adaptation_scales_applied": [1.0, 1.0, 1.0],
                    "chroma_bias_correction_enabled": bool(str(self.config.chroma_bias_correction).lower() == "on"),
                    "chroma_bias_correction_applied": False,
                    "chroma_bias_correction_skipped_reason": "non-color-image",
                }
            )
            return context

        corrected, metadata = apply_white_balance(context.image_f32, self.config)
        context.image_f32 = corrected
        context.metadata.update(metadata)
        return context


def apply_white_balance(image_srgb: np.ndarray, config: WhiteBalanceConfig) -> tuple[np.ndarray, dict[str, Any]]:
    source = np.clip(image_srgb, 0.0, 1.0).astype(np.float32, copy=False)
    linear = srgb_to_linear(source)
    luminance_before = rec709_luminance(linear)
    mode_meta = normalize_white_balance_mode(config.method)

    strength_requested = float(np.clip(config.strength, 0.0, 1.0))
    max_gain = float(max(1.0, config.max_gain))
    shades_of_gray_p = float(max(1.0, config.shades_of_gray_p))
    gray_edge_sigma = float(max(0.0, config.gray_edge_sigma))
    clip_threshold = float(np.clip(config.clip_fraction_threshold, 0.0, 0.02))

    border = estimate_border_reference(linear=linear, config=config)
    reference_method = "border" if border["used"] else "fallback"

    if border["used"]:
        illuminant = border["illuminant"]
        estimator_illuminants = {"border-reference": [float(x) for x in illuminant]}
        estimator_reliability = {"border-reference": float(border["confidence"])}
        estimator_weights = {"border-reference": 1.0}
        consensus_agreement = 1.0
        confidence = float(border["confidence"])
        valid_pixels = int(border["valid_pixels"])
        valid_fraction = float(border["valid_fraction"])
        sampling_mode = "border-validated"
    else:
        fallback = estimate_illuminant_robust(
            linear=linear,
            method=mode_meta["effective_mode"],
            p=shades_of_gray_p,
            gray_edge_sigma=gray_edge_sigma,
            min_valid_pixels=max(128, int(config.min_valid_pixels)),
            low_percentile=float(np.clip(config.fallback_percentile_low, 0.0, 49.0)),
            high_percentile=float(np.clip(config.fallback_percentile_high, 51.0, 100.0)),
            clipped_exclusion_threshold=float(np.clip(config.clipped_exclusion_threshold, 0.90, 0.9999)),
        )
        illuminant = fallback["illuminant"]
        estimator_illuminants = fallback["estimator_illuminants"]
        estimator_reliability = fallback["estimator_reliability"]
        estimator_weights = fallback["estimator_weights"]
        consensus_agreement = fallback["consensus_agreement"]
        confidence = float(fallback["confidence"])
        valid_pixels = int(fallback["valid_pixels"])
        valid_fraction = float(fallback["valid_fraction"])
        sampling_mode = str(fallback["sampling_mode"])

    illuminant = normalize_illuminant(illuminant)
    illum_mean = float(np.mean(illuminant))
    raw_scales = np.divide(
        illum_mean,
        np.maximum(illuminant, 1e-8),
        out=np.ones(3, dtype=np.float32),
        where=illuminant > 1e-8,
    ).astype(np.float32, copy=False)
    clamped_scales = np.clip(raw_scales, 1.0 / max_gain, max_gain)

    strength_factor = confidence_strength_factor(
        confidence=confidence,
        reduce_threshold=float(np.clip(config.confidence_reduce_threshold, 0.05, 1.0)),
        skip_threshold=float(np.clip(config.confidence_skip_threshold, 0.0, 0.95)),
    )
    strength_effective = float(np.clip(strength_requested * strength_factor, 0.0, 1.0))

    preclip_low = 0.0
    preclip_high = 0.0
    observed_hue_rotation = 0.0
    observed_sat_gain = 1.0
    guardrail_reasons: list[str] = []
    guardrail_iterations = 0
    luma_scalar = 1.0
    applied_scales = np.ones(3, dtype=np.float32)
    corrected_linear = linear.copy()
    max_hue_rotation_deg = float(np.clip(config.max_hue_rotation_deg, 0.0, 90.0))
    max_saturation_gain = float(max(1.0, config.max_saturation_gain))
    max_luma_scale = float(max(1.0, config.max_luminance_scale))

    for _ in range(max(0, int(config.max_guardrail_iterations)) + 1):
        applied_scales = (1.0 + (strength_effective * (clamped_scales - 1.0))).astype(np.float32, copy=False)
        candidate_linear = linear * applied_scales[None, None, :]
        luminance_candidate = rec709_luminance(candidate_linear)
        mean_before = float(np.mean(luminance_before))
        mean_after = float(np.mean(luminance_candidate))
        luma_scalar_raw = mean_before / max(mean_after, 1e-8)
        luma_scalar = float(np.clip(luma_scalar_raw, 1.0 / max_luma_scale, max_luma_scale))
        candidate_linear *= luma_scalar

        preclip_low = float(np.mean(candidate_linear < 0.0))
        preclip_high = float(np.mean(candidate_linear > 1.0))
        observed_hue_rotation, observed_sat_gain = transform_change_proxies(
            original_linear=linear,
            corrected_linear=np.clip(candidate_linear, 0.0, 1.0),
        )

        clip_violation = (preclip_low + preclip_high) > clip_threshold
        hue_violation = observed_hue_rotation > max_hue_rotation_deg
        sat_violation = observed_sat_gain > max_saturation_gain

        if not clip_violation and not hue_violation and not sat_violation:
            corrected_linear = candidate_linear
            break

        if clip_violation and "clip-fraction-limit" not in guardrail_reasons:
            guardrail_reasons.append("clip-fraction-limit")
        if hue_violation and "hue-rotation-limit" not in guardrail_reasons:
            guardrail_reasons.append("hue-rotation-limit")
        if sat_violation and "saturation-gain-limit" not in guardrail_reasons:
            guardrail_reasons.append("saturation-gain-limit")

        strength_effective *= 0.85
        guardrail_iterations += 1
        corrected_linear = candidate_linear

    residual_violation = (
        ((preclip_low + preclip_high) > clip_threshold)
        or (observed_hue_rotation > max_hue_rotation_deg)
        or (observed_sat_gain > max_saturation_gain)
    )
    if residual_violation:
        strength_effective = 0.0
        applied_scales = np.ones(3, dtype=np.float32)
        luma_scalar = 1.0
        corrected_linear = linear.copy()
        preclip_low = 0.0
        preclip_high = 0.0
        observed_hue_rotation = 0.0
        observed_sat_gain = 1.0
        guardrail_reasons.append("wb-disabled-after-guardrails")

    chroma_bias_enabled = str(config.chroma_bias_correction).lower() == "on"
    chroma_bias_meta: dict[str, Any]
    if chroma_bias_enabled and strength_effective > 1e-6:
        corrected_linear, chroma_bias_meta = remove_low_frequency_chroma_bias(
            linear_rgb=corrected_linear,
            radius_spec=str(config.chroma_bias_radius_spec),
            cap=float(np.clip(config.chroma_bias_cap, 0.0, 0.20)),
            strength=float(np.clip(config.chroma_bias_strength, 0.0, 1.0)),
        )
    else:
        chroma_bias_meta = {
            "chroma_bias_correction_enabled": chroma_bias_enabled,
            "chroma_bias_correction_applied": False,
            "chroma_bias_correction_skipped_reason": (
                "disabled"
                if not chroma_bias_enabled
                else "wb-disabled"
            ),
            "chroma_bias_radius_spec": str(config.chroma_bias_radius_spec),
            "chroma_bias_cap": float(np.clip(config.chroma_bias_cap, 0.0, 0.20)),
        }

    corrected_linear = limit_chroma_boost(
        original_linear=linear,
        corrected_linear=corrected_linear,
        max_chroma_boost=float(max(0.0, config.max_chroma_boost)),
    )

    preclip_fraction = float(np.mean((corrected_linear < 0.0) | (corrected_linear > 1.0)))
    corrected_linear = np.clip(corrected_linear, 0.0, 1.0)
    corrected_srgb = linear_to_srgb(corrected_linear)

    luminance_after = rec709_luminance(corrected_linear)
    applied_channel_scales = [float(x) for x in (applied_scales * luma_scalar)]
    global_transform_type = "identity"
    if strength_effective > 1e-6:
        global_transform_type = "diagonal-linear-rgb+global-luminance-scale"

    metadata: dict[str, Any] = {
        "white_balance_method": mode_meta["effective_mode"],
        "white_balance_cli_mode": mode_meta["cli_mode"],
        "white_balance_effective_mode": mode_meta["effective_mode"],
        "white_balance_mode_mapped": mode_meta["mode_mapped"],
        "white_balance_mode_mapping_reason": mode_meta["mapping_reason"],
        "white_balance_applied": bool(strength_effective > 1e-6),
        "white_balance_skipped_reason": None if (strength_effective > 1e-6) else "low-confidence-or-guardrail",
        "white_balance_reference_method": reference_method,
        "white_balance_border_used": bool(border["used"]),
        "white_balance_border_candidate_count": int(border["candidate_count"]),
        "white_balance_border_region_stats": border["region_stats"],
        "white_balance_border_reject_reasons": border["reject_reasons"],
        "white_balance_illuminant_estimate": [float(x) for x in illuminant],
        "white_balance_estimated_illuminant": [float(x) for x in illuminant],
        "white_balance_channel_scales_raw": [float(x) for x in raw_scales],
        "white_balance_channel_scales_applied": applied_channel_scales,
        "white_balance_global_transform_type": global_transform_type,
        "white_balance_global_transform_values": {
            "rgb_scales": applied_channel_scales,
            "luminance_scalar": float(luma_scalar),
            "strength_effective": float(strength_effective),
        },
        "white_balance_percentile_exclusion_low": float(np.clip(config.fallback_percentile_low, 0.0, 49.0)),
        "white_balance_percentile_exclusion_high": float(np.clip(config.fallback_percentile_high, 51.0, 100.0)),
        "white_balance_clipped_exclusion_threshold": float(np.clip(config.clipped_exclusion_threshold, 0.90, 0.9999)),
        "white_balance_estimator_reliability": estimator_reliability,
        "white_balance_estimator_weights": estimator_weights,
        "white_balance_estimator_illuminants": estimator_illuminants,
        "white_balance_consensus_agreement": None if consensus_agreement is None else float(consensus_agreement),
        "white_balance_confidence": float(np.clip(confidence, 0.0, 1.0)),
        "white_balance_valid_pixels": int(valid_pixels),
        "white_balance_valid_fraction": float(valid_fraction),
        "white_balance_sampling_mode": sampling_mode,
        "white_balance_sampling_fallback_used": reference_method != "border",
        "white_balance_strength_requested": strength_requested,
        "white_balance_strength_effective": float(strength_effective),
        "white_balance_strength": float(strength_effective),
        "white_balance_max_gain": max_gain,
        "white_balance_shades_of_gray_p": shades_of_gray_p,
        "white_balance_gray_edge_sigma": gray_edge_sigma,
        "white_balance_preclip_fraction": preclip_fraction,
        "white_balance_preclip_fraction_low": preclip_low,
        "white_balance_preclip_fraction_high": preclip_high,
        "white_balance_guardrail_iterations": int(guardrail_iterations),
        "white_balance_guardrail_triggers": guardrail_reasons,
        "white_balance_max_hue_rotation_deg": max_hue_rotation_deg,
        "white_balance_observed_hue_rotation_deg": float(observed_hue_rotation),
        "white_balance_max_saturation_gain": max_saturation_gain,
        "white_balance_observed_saturation_gain": float(observed_sat_gain),
        "white_balance_global_luminance_scalar": float(luma_scalar),
        "white_balance_luminance_mean_before": float(np.mean(luminance_before)),
        "white_balance_luminance_mean_after": float(np.mean(luminance_after)),
        "white_balance_luminance_p95_before": float(np.percentile(luminance_before, 95.0)),
        "white_balance_luminance_p95_after": float(np.percentile(luminance_after, 95.0)),
        "white_balance_adaptation_space": "global-diagonal-linear-rgb",
        "white_balance_adaptation_scales_raw": [float(x) for x in raw_scales],
        "white_balance_adaptation_scales_applied": applied_channel_scales,
    }
    metadata.update(chroma_bias_meta)
    return corrected_srgb.astype(np.float32, copy=False), metadata


def normalize_white_balance_mode(method: str) -> dict[str, Any]:
    cli_mode = str(method).strip().lower()
    if cli_mode == "none":
        return {
            "cli_mode": cli_mode,
            "effective_mode": "shades-of-gray",
            "mode_mapped": True,
            "mapping_reason": "robust-only-policy-none-mapped-to-shades-of-gray",
        }
    if cli_mode == "gray-world":
        return {
            "cli_mode": cli_mode,
            "effective_mode": "shades-of-gray",
            "mode_mapped": True,
            "mapping_reason": "robust-only-policy-gray-world-mapped-to-shades-of-gray",
        }
    if cli_mode == "consensus":
        return {
            "cli_mode": cli_mode,
            "effective_mode": "consensus-robust",
            "mode_mapped": True,
            "mapping_reason": "robust-only-policy-consensus-mapped-to-robust-ensemble",
        }
    if cli_mode in {"shades-of-gray", "gray-edge", "consensus-robust"}:
        return {
            "cli_mode": cli_mode,
            "effective_mode": cli_mode,
            "mode_mapped": False,
            "mapping_reason": "none",
        }
    return {
        "cli_mode": cli_mode,
        "effective_mode": "shades-of-gray",
        "mode_mapped": True,
        "mapping_reason": "unsupported-mode-mapped-to-shades-of-gray",
    }


def estimate_border_reference(*, linear: np.ndarray, config: WhiteBalanceConfig) -> dict[str, Any]:
    h, w = linear.shape[:2]
    min_dim = max(1, min(h, w))
    band = int(np.clip(round(float(np.clip(config.border_band, 0.01, 0.30)) * min_dim), 4, max(4, min_dim // 2)))
    luminance = rec709_luminance(linear)
    texture = gradient_magnitude(luminance)
    max_rgb = np.max(linear, axis=2)
    min_rgb = np.min(linear, axis=2)
    clip_thr = float(np.clip(config.clipped_exclusion_threshold, 0.90, 0.9999))
    low_thr = float(max(0.0, 1.0 - clip_thr))
    clipped = (max_rgb >= clip_thr) | (min_rgb <= low_thr)

    regions = border_regions(h=h, w=w, band=band)
    region_stats: list[dict[str, Any]] = []
    accepted_means: list[np.ndarray] = []
    accepted_weights: list[float] = []
    reject_reasons: set[str] = set()
    candidate_count = 0

    for region_name, region_mask in regions:
        region_size = int(np.count_nonzero(region_mask))
        if region_size <= 0:
            continue

        region_texture = texture[region_mask]
        region_luma = luminance[region_mask]
        region_rgb = linear[region_mask]
        region_clipped = clipped[region_mask]

        low_texture = region_texture <= float(max(0.0, config.border_texture_max))
        unclipped = ~region_clipped
        candidate_mask = low_texture & unclipped
        candidate_pixels = int(np.count_nonzero(candidate_mask))
        candidate_fraction = float(candidate_pixels / max(region_size, 1))
        min_area = max(64, int(round(0.05 * region_size)))
        if candidate_pixels > 0:
            candidate_count += 1

        reasons: list[str] = []
        if candidate_pixels < min_area:
            reasons.append("insufficient-low-texture-area")

        clip_fraction = float(np.mean(region_clipped))
        if clip_fraction > float(max(0.0, config.border_clip_max)):
            reasons.append("excess-clipping")

        if candidate_pixels > 0:
            candidate_rgb = region_rgb[candidate_mask]
            candidate_luma = region_luma[candidate_mask]
        else:
            candidate_rgb = region_rgb
            candidate_luma = region_luma

        chroma_mag, warm_bias, chroma_var = chroma_stats(candidate_rgb)
        if chroma_mag > float(max(0.0, config.border_neutrality_max)):
            reasons.append("non-neutral-chroma-mean")
        if chroma_var > float(max(0.0, config.border_neutrality_max) ** 2):
            reasons.append("non-neutral-chroma-variance")
        if warm_bias > float(max(0.0, config.border_warm_bias_max)):
            reasons.append("warm-bias")

        luma_p01 = float(np.percentile(candidate_luma, 1.0))
        luma_p99 = float(np.percentile(candidate_luma, 99.0))
        if luma_p99 >= clip_thr or luma_p01 <= low_thr:
            reasons.append("luminance-clipped")

        accepted = len(reasons) == 0
        if accepted:
            accepted_means.append(np.mean(candidate_rgb, axis=0).astype(np.float32, copy=False))
            accepted_weights.append(float(candidate_pixels))
        else:
            for reason in reasons:
                reject_reasons.add(f"{region_name}:{reason}")

        region_stats.append(
            {
                "region": region_name,
                "region_pixels": region_size,
                "candidate_pixels": candidate_pixels,
                "candidate_fraction": candidate_fraction,
                "clip_fraction": clip_fraction,
                "texture_mean": float(np.mean(region_texture)),
                "texture_p95": float(np.percentile(region_texture, 95.0)),
                "chroma_mag": float(chroma_mag),
                "chroma_var": float(chroma_var),
                "warm_bias": float(warm_bias),
                "luminance_p01": luma_p01,
                "luminance_p99": luma_p99,
                "accepted": accepted,
                "reject_reasons": reasons,
            }
        )

    if not accepted_means:
        return {
            "used": False,
            "illuminant": np.ones(3, dtype=np.float32),
            "candidate_count": candidate_count,
            "region_stats": region_stats,
            "reject_reasons": sorted(reject_reasons),
            "confidence": 0.0,
            "valid_pixels": 0,
            "valid_fraction": 0.0,
        }

    weights = np.asarray(accepted_weights, dtype=np.float32)
    means = np.asarray(accepted_means, dtype=np.float32)
    mean_rgb = np.sum(means * weights[:, None], axis=0) / max(float(np.sum(weights)), 1e-8)
    illuminant = normalize_illuminant(mean_rgb)

    accepted_count = int(len(accepted_means))
    valid_pixels = int(np.sum(weights))
    valid_fraction = float(valid_pixels / max(h * w, 1))
    neutrality_mean = float(np.mean([s["chroma_mag"] for s in region_stats if s["accepted"]]))
    clip_mean = float(np.mean([s["clip_fraction"] for s in region_stats if s["accepted"]]))
    confidence = (
        (0.35 * float(np.clip(valid_fraction / 0.06, 0.0, 1.0)))
        + (0.25 * float(np.clip(accepted_count / 3.0, 0.0, 1.0)))
        + (0.25 * float(np.clip(1.0 - (neutrality_mean / max(float(config.border_neutrality_max), 1e-6)), 0.0, 1.0)))
        + (0.15 * float(np.clip(1.0 - (clip_mean / max(float(config.border_clip_max), 1e-6)), 0.0, 1.0)))
    )

    return {
        "used": True,
        "illuminant": illuminant.astype(np.float32, copy=False),
        "candidate_count": candidate_count,
        "region_stats": region_stats,
        "reject_reasons": sorted(reject_reasons),
        "confidence": float(np.clip(confidence, 0.0, 1.0)),
        "valid_pixels": valid_pixels,
        "valid_fraction": valid_fraction,
    }


def estimate_illuminant_robust(
    *,
    linear: np.ndarray,
    method: str,
    p: float,
    gray_edge_sigma: float,
    min_valid_pixels: int,
    low_percentile: float,
    high_percentile: float,
    clipped_exclusion_threshold: float,
) -> dict[str, Any]:
    luminance = rec709_luminance(linear)
    max_rgb = np.max(linear, axis=2)
    min_rgb = np.min(linear, axis=2)
    low_thr = max(0.0, 1.0 - clipped_exclusion_threshold)
    luma_low = float(np.percentile(luminance, low_percentile))
    luma_high = float(np.percentile(luminance, high_percentile))

    mask = (
        (luminance >= luma_low)
        & (luminance <= luma_high)
        & (max_rgb < clipped_exclusion_threshold)
        & (min_rgb > low_thr)
    )
    sampling_mode = "percentile-exclusion"
    valid_pixels = int(np.count_nonzero(mask))
    total_pixels = int(luminance.size)

    if valid_pixels < min_valid_pixels:
        mask = (luminance >= luma_low) & (luminance <= luma_high) & (max_rgb < clipped_exclusion_threshold)
        sampling_mode = "percentile-exclusion-relaxed-lowclip"
        valid_pixels = int(np.count_nonzero(mask))
    if valid_pixels < max(128, min_valid_pixels // 4):
        mask = (luminance >= luma_low) & (luminance <= luma_high)
        sampling_mode = "percentile-exclusion-luma-only"
        valid_pixels = int(np.count_nonzero(mask))
    if valid_pixels < 128:
        mask = np.ones_like(luminance, dtype=bool)
        sampling_mode = "all-pixels-fallback"
        valid_pixels = int(np.count_nonzero(mask))

    valid_fraction = float(valid_pixels / max(total_pixels, 1))
    luma_valid = luminance[mask]
    spread = float(np.percentile(luma_valid, 95.0) - np.percentile(luma_valid, 5.0))
    count_term = float(np.clip(valid_pixels / max(float(min_valid_pixels), 1.0), 0.0, 1.0))
    frac_term = float(np.clip(valid_fraction / 0.20, 0.0, 1.0))
    spread_term = float(np.clip(spread / 0.35, 0.0, 1.0))

    shades_illum = estimate_shades_of_gray_from_mask(linear, valid_mask=mask, p=p)
    gray_illum, gray_reliability = estimate_gray_edge_illuminant(
        linear=linear,
        valid_mask=mask,
        p=p,
        sigma=gray_edge_sigma,
        min_valid_pixels=min_valid_pixels,
    )
    shades_reliability = (0.45 * count_term) + (0.35 * frac_term) + (0.20 * spread_term)
    shades_reliability = float(np.clip(shades_reliability, 0.0, 1.0))
    gray_reliability = float(np.clip(gray_reliability, 0.0, 1.0))

    if method == "gray-edge":
        illuminant = normalize_illuminant(gray_illum)
        estimator_weights = {"gray-edge": 1.0}
        estimator_reliability = {"gray-edge": gray_reliability}
        estimator_illuminants = {"gray-edge": [float(x) for x in gray_illum]}
        confidence = (0.55 * gray_reliability) + (0.30 * count_term) + (0.15 * spread_term)
        consensus_agreement: float | None = 1.0
    elif method == "consensus-robust":
        w_shades = 0.70 * np.power(max(shades_reliability, 1e-6), 1.25)
        w_gray = 0.30 * np.power(max(gray_reliability, 1e-6), 1.25)
        denom = max(w_shades + w_gray, 1e-8)
        consensus = ((w_shades * normalize_illuminant(shades_illum)) + (w_gray * normalize_illuminant(gray_illum))) / denom
        illuminant = normalize_illuminant(consensus.astype(np.float32, copy=False))
        agreement = estimator_agreement([normalize_illuminant(shades_illum), normalize_illuminant(gray_illum)])
        confidence = (0.25 * count_term) + (0.20 * frac_term) + (0.10 * spread_term) + (0.20 * agreement) + (
            0.25 * ((0.70 * shades_reliability) + (0.30 * gray_reliability))
        )
        estimator_weights = {
            "shades-of-gray": float(w_shades / denom),
            "gray-edge": float(w_gray / denom),
        }
        estimator_reliability = {
            "shades-of-gray": shades_reliability,
            "gray-edge": gray_reliability,
        }
        estimator_illuminants = {
            "shades-of-gray": [float(x) for x in shades_illum],
            "gray-edge": [float(x) for x in gray_illum],
        }
        consensus_agreement = float(agreement)
    else:
        illuminant = normalize_illuminant(shades_illum)
        estimator_weights = {"shades-of-gray": 1.0}
        estimator_reliability = {"shades-of-gray": shades_reliability}
        estimator_illuminants = {"shades-of-gray": [float(x) for x in shades_illum]}
        confidence = (0.60 * shades_reliability) + (0.25 * count_term) + (0.15 * spread_term)
        consensus_agreement = 1.0

    if sampling_mode == "all-pixels-fallback":
        confidence *= 0.40
    elif sampling_mode == "percentile-exclusion-luma-only":
        confidence *= 0.72
    elif sampling_mode == "percentile-exclusion-relaxed-lowclip":
        confidence *= 0.86

    return {
        "illuminant": illuminant.astype(np.float32, copy=False),
        "confidence": float(np.clip(confidence, 0.0, 1.0)),
        "valid_pixels": valid_pixels,
        "valid_fraction": valid_fraction,
        "sampling_mode": sampling_mode,
        "estimator_illuminants": estimator_illuminants,
        "estimator_reliability": estimator_reliability,
        "estimator_weights": estimator_weights,
        "consensus_agreement": consensus_agreement,
    }


def remove_low_frequency_chroma_bias(
    *,
    linear_rgb: np.ndarray,
    radius_spec: str,
    cap: float,
    strength: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    luminance = rec709_luminance(linear_rgb)
    height, width = luminance.shape
    radius_px = resolve_radius_px(spec=radius_spec, shape=(height, width), default_px=0.16 * min(height, width))
    if radius_px < 4.0 or cap <= 1e-6 or strength <= 1e-6:
        return linear_rgb.astype(np.float32, copy=False), {
            "chroma_bias_correction_enabled": True,
            "chroma_bias_correction_applied": False,
            "chroma_bias_correction_skipped_reason": "radius-or-cap-too-small",
            "chroma_bias_radius_spec": str(radius_spec),
            "chroma_bias_radius_px_effective": float(radius_px),
            "chroma_bias_cap": float(cap),
            "chroma_bias_strength": float(strength),
        }

    # Opponent chroma coordinates anchored to Rec.709 luminance.
    # u = R - Y, v = B - Y keeps Y unchanged when recombining.
    u = linear_rgb[:, :, 0] - luminance
    v = linear_rgb[:, :, 2] - luminance

    ds = choose_downsample_factor((height, width), max_dim=1024)
    u_small = u[::ds, ::ds]
    v_small = v[::ds, ::ds]
    sigma_small = max(1.0, float(radius_px) / max(3.0 * ds, 1.0))
    bias_u = upsample_nearest(gaussian_blur_2d(u_small, sigma=sigma_small), target_shape=(height, width))
    bias_v = upsample_nearest(gaussian_blur_2d(v_small, sigma=sigma_small), target_shape=(height, width))

    # Conservative correction of low-frequency bias only.
    delta_u = np.clip(strength * bias_u, -cap, cap)
    delta_v = np.clip(strength * bias_v, -cap, cap)
    u_corr = u - delta_u
    v_corr = v - delta_v

    # Prevent large sign flips that can invert hues in aged scans.
    u_flip = (u * u_corr) < 0.0
    v_flip = (v * v_corr) < 0.0
    u_corr = np.where(u_flip, 0.25 * u, u_corr)
    v_corr = np.where(v_flip, 0.25 * v, v_corr)

    # Re-center chroma means near neutral while preserving luminance.
    u_mean = float(np.mean(u_corr))
    v_mean = float(np.mean(v_corr))
    u_corr -= np.clip(u_mean, -0.5 * cap, 0.5 * cap)
    v_corr -= np.clip(v_mean, -0.5 * cap, 0.5 * cap)

    r = luminance + u_corr
    b = luminance + v_corr
    g = (luminance - (LUMA_R * r) - (LUMA_B * b)) / LUMA_G
    corrected = np.stack((r, g, b), axis=2).astype(np.float32, copy=False)

    return corrected, {
        "chroma_bias_correction_enabled": True,
        "chroma_bias_correction_applied": True,
        "chroma_bias_radius_spec": str(radius_spec),
        "chroma_bias_radius_px_effective": float(radius_px),
        "chroma_bias_cap": float(cap),
        "chroma_bias_strength": float(strength),
        "chroma_bias_downsample_factor": int(ds),
        "chroma_bias_delta_u_mean_abs": float(np.mean(np.abs(delta_u))),
        "chroma_bias_delta_v_mean_abs": float(np.mean(np.abs(delta_v))),
        "chroma_bias_u_mean_before": float(np.mean(u)),
        "chroma_bias_v_mean_before": float(np.mean(v)),
        "chroma_bias_u_mean_after": float(np.mean(u_corr)),
        "chroma_bias_v_mean_after": float(np.mean(v_corr)),
    }


def border_regions(*, h: int, w: int, band: int) -> list[tuple[str, np.ndarray]]:
    top = np.zeros((h, w), dtype=bool)
    bottom = np.zeros((h, w), dtype=bool)
    left = np.zeros((h, w), dtype=bool)
    right = np.zeros((h, w), dtype=bool)

    top[:band, :] = True
    bottom[h - band :, :] = True
    inner_top = band
    inner_bottom = max(inner_top, h - band)
    left[inner_top:inner_bottom, :band] = True
    right[inner_top:inner_bottom, w - band :] = True
    return [("top", top), ("bottom", bottom), ("left", left), ("right", right)]


def chroma_stats(rgb: np.ndarray) -> tuple[float, float, float]:
    if rgb.size == 0:
        return 0.0, 0.0, 0.0
    y = (LUMA_R * rgb[:, 0]) + (LUMA_G * rgb[:, 1]) + (LUMA_B * rgb[:, 2])
    u = rgb[:, 0] - y
    v = rgb[:, 2] - y
    chroma_mag = np.sqrt((u * u) + (v * v))
    warm_bias = float(np.mean(rgb[:, 0] - rgb[:, 2]))
    chroma_var = float(np.var(chroma_mag))
    return float(np.mean(chroma_mag)), warm_bias, chroma_var


def estimate_shades_of_gray_from_mask(linear: np.ndarray, *, valid_mask: np.ndarray, p: float) -> np.ndarray:
    pixels = linear[valid_mask]
    if pixels.size == 0:
        pixels = linear.reshape(-1, 3)
    p_clamped = max(1.0, float(p))
    illum = np.power(np.mean(np.power(np.maximum(pixels, 1e-8), p_clamped), axis=0), 1.0 / p_clamped)
    return np.maximum(illum.astype(np.float32, copy=False), 1e-8)


def estimate_gray_edge_illuminant(
    *,
    linear: np.ndarray,
    valid_mask: np.ndarray,
    p: float,
    sigma: float,
    min_valid_pixels: int,
) -> tuple[np.ndarray, float]:
    smooth = np.stack([gaussian_blur_2d(linear[:, :, idx], sigma=sigma) for idx in range(3)], axis=2)
    grad_rgb = gradient_magnitude(smooth)
    grad_luma = gradient_magnitude(rec709_luminance(smooth))

    valid_grad = grad_luma[valid_mask]
    if valid_grad.size == 0:
        return estimate_shades_of_gray_from_mask(linear, valid_mask=valid_mask, p=p), 0.0

    edge_threshold = float(np.percentile(valid_grad, 65.0))
    edge_mask = valid_mask & (grad_luma >= edge_threshold)
    edge_pixels = int(np.count_nonzero(edge_mask))
    if edge_pixels < max(128, min_valid_pixels // 4):
        edge_mask = valid_mask
        edge_pixels = int(np.count_nonzero(edge_mask))

    weights = np.where(edge_mask, grad_luma + 1e-4, 0.0).astype(np.float32, copy=False)
    denom = float(np.sum(weights))
    if denom <= 1e-8:
        return estimate_shades_of_gray_from_mask(linear, valid_mask=valid_mask, p=p), 0.0

    p_clamped = max(1.0, float(p))
    channels: list[float] = []
    for idx in range(3):
        g = np.maximum(grad_rgb[:, :, idx], 1e-8)
        accum = float(np.sum(weights * np.power(g, p_clamped)) / denom)
        channels.append(float(np.power(max(accum, 1e-8), 1.0 / p_clamped)))

    illum = np.maximum(np.array(channels, dtype=np.float32), 1e-8)
    count_term = float(np.clip(edge_pixels / max(float(min_valid_pixels), 1.0), 0.0, 1.0))
    energy_term = float(np.clip(float(np.mean(grad_luma[edge_mask])) / 0.08, 0.0, 1.0))
    reliability = (0.60 * count_term) + (0.40 * energy_term)
    if edge_pixels < max(256, min_valid_pixels // 2):
        reliability *= 0.80
    return illum, float(np.clip(reliability, 0.0, 1.0))


def estimator_agreement(vectors: list[np.ndarray]) -> float:
    if len(vectors) <= 1:
        return 1.0
    distances: list[float] = []
    for idx, va in enumerate(vectors):
        for vb in vectors[idx + 1 :]:
            ratio = np.divide(np.maximum(va, 1e-8), np.maximum(vb, 1e-8))
            distances.append(float(np.mean(np.abs(np.log(np.maximum(ratio, 1e-8))))))
    if not distances:
        return 1.0
    median_distance = float(np.median(np.asarray(distances, dtype=np.float32)))
    return float(np.clip(1.0 / (1.0 + (median_distance / 0.14)), 0.0, 1.0))


def transform_change_proxies(*, original_linear: np.ndarray, corrected_linear: np.ndarray) -> tuple[float, float]:
    lum_orig = rec709_luminance(original_linear)
    lum_corr = rec709_luminance(corrected_linear)

    u_orig = original_linear[:, :, 0] - lum_orig
    v_orig = original_linear[:, :, 2] - lum_orig
    u_corr = corrected_linear[:, :, 0] - lum_corr
    v_corr = corrected_linear[:, :, 2] - lum_corr

    mag_orig = np.sqrt((u_orig * u_orig) + (v_orig * v_orig))
    mag_corr = np.sqrt((u_corr * u_corr) + (v_corr * v_corr))
    valid = (lum_orig > 0.02) & (lum_orig < 0.98) & (mag_orig > 1e-4)
    if int(np.count_nonzero(valid)) < 128:
        return 0.0, 1.0

    ang_orig = np.arctan2(v_orig[valid], u_orig[valid])
    ang_corr = np.arctan2(v_corr[valid], u_corr[valid])
    delta = ang_corr - ang_orig
    delta = np.abs(np.arctan2(np.sin(delta), np.cos(delta)))
    hue_rotation = float(np.degrees(np.percentile(delta, 95.0)))

    sat_ratio = np.divide(
        mag_corr[valid],
        np.maximum(mag_orig[valid], 1e-6),
        out=np.ones_like(mag_corr[valid]),
        where=mag_orig[valid] > 1e-6,
    )
    sat_gain = float(np.percentile(sat_ratio, 95.0))
    return hue_rotation, sat_gain


def normalize_illuminant(illuminant: np.ndarray) -> np.ndarray:
    illum = np.maximum(np.asarray(illuminant, dtype=np.float32).reshape(3), 1e-8)
    return illum / max(float(np.mean(illum)), 1e-8)


def confidence_strength_factor(*, confidence: float, reduce_threshold: float, skip_threshold: float) -> float:
    c = float(np.clip(confidence, 0.0, 1.0))
    skip_t = float(np.clip(skip_threshold, 0.0, 0.95))
    reduce_t = float(max(skip_t + 1e-6, np.clip(reduce_threshold, 0.01, 1.0)))
    if c <= skip_t:
        return 0.0
    if c >= reduce_t:
        return 1.0
    t = (c - skip_t) / (reduce_t - skip_t)
    return float(0.20 + (0.80 * t))


def baseline_luminance_metadata(image_srgb: np.ndarray) -> dict[str, Any]:
    linear = srgb_to_linear(np.clip(image_srgb, 0.0, 1.0).astype(np.float32, copy=False))
    luma = rec709_luminance(linear)
    return {
        "white_balance_luminance_mean_before": float(np.mean(luma)),
        "white_balance_luminance_mean_after": float(np.mean(luma)),
        "white_balance_luminance_p95_before": float(np.percentile(luma, 95.0)),
        "white_balance_luminance_p95_after": float(np.percentile(luma, 95.0)),
        "white_balance_preclip_fraction": 0.0,
        "white_balance_preclip_fraction_low": 0.0,
        "white_balance_preclip_fraction_high": 0.0,
    }


def limit_chroma_boost(*, original_linear: np.ndarray, corrected_linear: np.ndarray, max_chroma_boost: float) -> np.ndarray:
    max_boost = float(max(0.0, max_chroma_boost))
    lum = rec709_luminance(corrected_linear)
    orig_u = original_linear[:, :, 0] - rec709_luminance(original_linear)
    orig_v = original_linear[:, :, 2] - rec709_luminance(original_linear)
    corr_u = corrected_linear[:, :, 0] - lum
    corr_v = corrected_linear[:, :, 2] - lum
    orig_mag = np.sqrt((orig_u * orig_u) + (orig_v * orig_v))
    corr_mag = np.sqrt((corr_u * corr_u) + (corr_v * corr_v))
    allowed = orig_mag * (1.0 + max_boost)
    ratio = np.divide(
        allowed,
        np.maximum(corr_mag, 1e-7),
        out=np.ones_like(corr_mag),
        where=corr_mag > allowed,
    )
    ratio = np.clip(ratio, 0.0, 1.0)
    corr_u *= ratio
    corr_v *= ratio
    r = lum + corr_u
    b = lum + corr_v
    g = (lum - (LUMA_R * r) - (LUMA_B * b)) / LUMA_G
    return np.stack((r, g, b), axis=2).astype(np.float32, copy=False)


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


def choose_downsample_factor(shape: tuple[int, int], max_dim: int) -> int:
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
    return image[y_idx[:, None], x_idx[None, :]].astype(np.float32, copy=False)


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
        for idx, weight in enumerate(kernel):
            out += float(weight) * padded[idx : idx + image.shape[0], :]
        return out
    padded = np.pad(image, ((0, 0), (radius, radius)), mode="reflect")
    out = np.zeros_like(image, dtype=np.float32)
    for idx, weight in enumerate(kernel):
        out += float(weight) * padded[:, idx : idx + image.shape[1]]
    return out


def gradient_magnitude(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        gy, gx = np.gradient(image.astype(np.float32, copy=False))
        return np.sqrt((gx * gx) + (gy * gy)).astype(np.float32, copy=False)
    grads = []
    for idx in range(image.shape[2]):
        gy, gx = np.gradient(image[:, :, idx].astype(np.float32, copy=False))
        grads.append(np.sqrt((gx * gx) + (gy * gy)).astype(np.float32, copy=False))
    return np.stack(grads, axis=2).astype(np.float32, copy=False)
