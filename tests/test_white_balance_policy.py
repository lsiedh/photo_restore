from __future__ import annotations

import numpy as np

from restore_batch.color_math import srgb_to_linear
from restore_batch.white_balance import (
    WhiteBalanceConfig,
    apply_white_balance,
    normalize_white_balance_mode,
)


def _base_color_image(height: int = 160, width: int = 220) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    xn = xx / max(width - 1, 1)
    yn = yy / max(height - 1, 1)
    r = 0.32 + (0.42 * xn) + (0.06 * yn)
    g = 0.28 + (0.35 * xn) + (0.05 * yn)
    b = 0.18 + (0.24 * xn) + (0.03 * yn)
    return np.clip(np.stack((r, g, b), axis=2), 0.0, 1.0).astype(np.float32)


def _with_border(image: np.ndarray, *, band: int, rgb: tuple[float, float, float]) -> np.ndarray:
    out = image.copy()
    out[:band, :, :] = rgb
    out[-band:, :, :] = rgb
    out[band:-band, :band, :] = rgb
    out[band:-band, -band:, :] = rgb
    return out


def test_legacy_mode_mapping_is_deterministic() -> None:
    cases = {
        "none": ("shades-of-gray", True),
        "gray-world": ("shades-of-gray", True),
        "shades-of-gray": ("shades-of-gray", False),
        "consensus": ("consensus-robust", True),
    }
    for cli_mode, (effective, mapped) in cases.items():
        result_a = normalize_white_balance_mode(cli_mode)
        result_b = normalize_white_balance_mode(cli_mode)
        assert result_a == result_b
        assert result_a["effective_mode"] == effective
        assert bool(result_a["mode_mapped"]) is mapped


def test_border_reference_is_used_when_border_is_neutral_and_low_texture() -> None:
    image = _base_color_image()
    image = _with_border(image, band=14, rgb=(0.78, 0.78, 0.78))
    cfg = WhiteBalanceConfig(
        method="shades-of-gray",
        chroma_bias_correction="off",
        border_band=0.08,
        border_texture_max=0.025,
        border_neutrality_max=0.06,
        border_clip_max=0.02,
        border_warm_bias_max=0.05,
    )
    _, meta = apply_white_balance(image, cfg)
    assert meta["white_balance_reference_method"] == "border"
    assert meta["white_balance_border_used"] is True
    assert meta["white_balance_border_candidate_count"] >= 1


def test_border_reference_rejects_warm_or_textured_border_and_falls_back() -> None:
    rng = np.random.default_rng(42)
    image = _base_color_image()
    image = _with_border(image, band=16, rgb=(0.84, 0.64, 0.42))
    # Add deterministic high texture only on border.
    noise = rng.normal(loc=0.0, scale=0.08, size=image.shape).astype(np.float32)
    border_mask = np.zeros(image.shape[:2], dtype=bool)
    border_mask[:16, :] = True
    border_mask[-16:, :] = True
    border_mask[16:-16, :16] = True
    border_mask[16:-16, -16:] = True
    image[border_mask] = np.clip(image[border_mask] + noise[border_mask], 0.0, 1.0)

    cfg = WhiteBalanceConfig(
        method="shades-of-gray",
        chroma_bias_correction="off",
        border_texture_max=0.010,
        border_neutrality_max=0.02,
        border_warm_bias_max=0.015,
    )
    _, meta = apply_white_balance(image, cfg)
    assert meta["white_balance_reference_method"] == "fallback"
    assert meta["white_balance_border_used"] is False
    reasons = " ".join(meta.get("white_balance_border_reject_reasons", []))
    assert ("warm-bias" in reasons) or ("non-neutral" in reasons) or ("insufficient-low-texture-area" in reasons)


def test_fallback_uses_fixed_percentile_exclusion_mask_fields() -> None:
    image = _base_color_image()
    image = _with_border(image, band=14, rgb=(0.88, 0.65, 0.50))
    cfg = WhiteBalanceConfig(
        method="none",  # mapped to shades-of-gray under robust-only policy
        chroma_bias_correction="off",
        border_neutrality_max=0.015,
        border_warm_bias_max=0.010,
        fallback_percentile_low=2.0,
        fallback_percentile_high=98.0,
        clipped_exclusion_threshold=0.995,
    )
    _, meta = apply_white_balance(image, cfg)
    assert meta["white_balance_reference_method"] == "fallback"
    assert meta["white_balance_effective_mode"] == "shades-of-gray"
    assert meta["white_balance_mode_mapped"] is True
    assert meta["white_balance_percentile_exclusion_low"] == 2.0
    assert meta["white_balance_percentile_exclusion_high"] == 98.0
    assert meta["white_balance_clipped_exclusion_threshold"] == 0.995
    assert "percentile-exclusion" in str(meta["white_balance_sampling_mode"])


def test_white_balance_transform_is_spatially_uniform_when_chroma_bias_off() -> None:
    image = _base_color_image(96, 128)
    cfg = WhiteBalanceConfig(
        method="shades-of-gray",
        strength=0.8,
        max_gain=1.12,
        chroma_bias_correction="off",
        skin_saturation_auto="off",
        border_band=0.10,
    )
    corrected, meta = apply_white_balance(image, cfg)
    assert meta["white_balance_applied"] is True

    src_lin = srgb_to_linear(image)
    out_lin = srgb_to_linear(corrected)
    mask = src_lin > 0.05
    ratios = np.divide(out_lin, np.maximum(src_lin, 1e-8), out=np.zeros_like(out_lin), where=mask)
    for ch in range(3):
        channel_ratios = ratios[:, :, ch][mask[:, :, ch]]
        assert float(np.max(channel_ratios) - np.min(channel_ratios)) < 1e-3


def test_white_balance_is_deterministic_for_identical_input_and_args() -> None:
    image = _base_color_image(120, 170)
    cfg = WhiteBalanceConfig(
        method="consensus",
        chroma_bias_correction="on",
        chroma_bias_radius_spec="14%",
    )
    corrected_a, meta_a = apply_white_balance(image, cfg)
    corrected_b, meta_b = apply_white_balance(image, cfg)
    assert np.array_equal(corrected_a, corrected_b)
    assert meta_a["white_balance_global_transform_values"] == meta_b["white_balance_global_transform_values"]
    assert meta_a["white_balance_border_used"] == meta_b["white_balance_border_used"]
    assert meta_a["white_balance_mode_mapping_reason"] == meta_b["white_balance_mode_mapping_reason"]


def test_aggressive_cast_mode_increases_effective_strength_or_gain() -> None:
    image = _base_color_image(120, 170)
    conservative_cfg = WhiteBalanceConfig(
        method="shades-of-gray",
        strength=0.55,
        cast_removal_mode="conservative",
        chroma_bias_correction="off",
    )
    aggressive_cfg = WhiteBalanceConfig(
        method="shades-of-gray",
        strength=0.55,
        cast_removal_mode="aggressive",
        chroma_bias_correction="off",
    )
    _, meta_cons = apply_white_balance(image, conservative_cfg)
    _, meta_aggr = apply_white_balance(image, aggressive_cfg)
    assert meta_cons["white_balance_cast_removal_mode"] == "conservative"
    assert meta_aggr["white_balance_cast_removal_mode"] == "aggressive"
    assert meta_aggr["white_balance_strength_requested"] >= meta_cons["white_balance_strength_requested"]
    assert meta_aggr["white_balance_max_gain"] >= meta_cons["white_balance_max_gain"]


def test_skin_saturation_auto_adjust_is_deterministic() -> None:
    image = _base_color_image(128, 192)
    # Inject a deterministic skin-like patch so the mask has enough pixels.
    image[36:92, 54:142, :] = np.array([0.72, 0.56, 0.46], dtype=np.float32)
    cfg = WhiteBalanceConfig(
        method="shades-of-gray",
        cast_removal_mode="aggressive",
        skin_saturation_auto="on",
        chroma_bias_correction="off",
    )
    out_a, meta_a = apply_white_balance(image, cfg)
    out_b, meta_b = apply_white_balance(image, cfg)
    assert np.array_equal(out_a, out_b)
    assert meta_a["white_balance_skin_saturation_auto_enabled"] is True
    assert meta_a["white_balance_skin_mask_pixels"] == meta_b["white_balance_skin_mask_pixels"]
    assert meta_a["white_balance_skin_saturation_adjust_factor"] == meta_b["white_balance_skin_saturation_adjust_factor"]
