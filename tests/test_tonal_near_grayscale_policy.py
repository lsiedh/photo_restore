from __future__ import annotations

import numpy as np

from restore_batch.tonal import TonalNormalizationConfig, apply_global_tonal_normalization


def test_near_grayscale_uses_stronger_floor_and_tighter_white_percentile() -> None:
    gradient = np.linspace(0.0, 1.0, num=256, dtype=np.float32)
    luma = np.tile(gradient[None, :], (192, 1))
    image = np.repeat(luma[:, :, None], 3, axis=2)

    _, meta = apply_global_tonal_normalization(
        image_srgb=image,
        config=TonalNormalizationConfig(),
        is_color_image=False,
    )
    assert meta["tonal_white_percentile"] == 99.2
    assert meta["tonal_contrast_strength"] == 0.30
    assert meta["tonal_strength_effective"] >= 0.50
    assert meta["tonal_near_grayscale_strength_floor_applied"] is True


def test_color_workflow_retains_global_white_percentile_defaults() -> None:
    yy, xx = np.mgrid[0:128, 0:192].astype(np.float32)
    xn = xx / max(191.0, 1.0)
    yn = yy / max(127.0, 1.0)
    image = np.clip(
        np.stack(
            (
                0.25 + (0.60 * xn),
                0.20 + (0.45 * xn) + (0.05 * yn),
                0.15 + (0.35 * xn) + (0.08 * yn),
            ),
            axis=2,
        ),
        0.0,
        1.0,
    ).astype(np.float32)

    cfg = TonalNormalizationConfig()
    _, meta = apply_global_tonal_normalization(
        image_srgb=image,
        config=cfg,
        is_color_image=True,
    )
    assert meta["tonal_white_percentile"] == cfg.white_percentile
    assert meta["tonal_contrast_strength"] == cfg.contrast_strength
    assert meta["tonal_near_grayscale_strength_floor_applied"] is False
