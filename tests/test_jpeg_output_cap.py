from __future__ import annotations

import numpy as np
import pytest

from restore_batch.jpeg_out import JpegExportConfig, JpegExportError, encode_jpeg_to_target


def _gradient_image(height: int = 320, width: int = 480) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    xn = xx / max(width - 1, 1)
    yn = yy / max(height - 1, 1)
    r = 0.20 + (0.70 * xn)
    g = 0.15 + (0.55 * yn)
    b = 0.10 + (0.45 * ((xn + yn) * 0.5))
    return np.clip(np.stack((r, g, b), axis=2), 0.0, 1.0).astype(np.float32)


def test_jpeg_export_meets_size_cap_without_resize() -> None:
    image = _gradient_image()
    jpeg_bytes, metadata = encode_jpeg_to_target(image_srgb_f32=image, config=JpegExportConfig(max_mb=3.0))
    assert len(jpeg_bytes) <= int(3.0 * 1024 * 1024)
    assert metadata["jpg_target_met"] is True
    assert metadata["jpg_downscale_iterations"] == 0
    assert metadata["output_format"] == "JPEG"
    assert metadata["export_intermediate_format"] == "TIFF"
    assert metadata["export_intermediate_tiff_in_memory"] is True


def test_jpeg_export_downscales_when_needed() -> None:
    rng = np.random.default_rng(7)
    image = rng.random((1600, 2200, 3), dtype=np.float32)
    jpeg_bytes, metadata = encode_jpeg_to_target(
        image_srgb_f32=image,
        config=JpegExportConfig(max_mb=0.20, quality_max=92, quality_min=80, quality_step=4, downscale_step=0.85, min_side=128),
    )
    assert len(jpeg_bytes) <= int(0.20 * 1024 * 1024)
    assert metadata["jpg_target_met"] is True
    assert metadata["jpg_downscale_iterations"] >= 1
    assert metadata["jpg_export_dims"][0] < 2200
    assert metadata["jpg_export_dims"][1] < 1600


def test_jpeg_export_is_deterministic_for_same_input_and_args() -> None:
    image = _gradient_image(512, 768)
    cfg = JpegExportConfig(max_mb=1.0, quality_max=92, quality_min=62, quality_step=4, downscale_step=0.90, min_side=128)
    bytes_a, meta_a = encode_jpeg_to_target(image_srgb_f32=image, config=cfg)
    bytes_b, meta_b = encode_jpeg_to_target(image_srgb_f32=image, config=cfg)
    assert bytes_a == bytes_b
    assert meta_a["jpg_quality_final"] == meta_b["jpg_quality_final"]
    assert meta_a["jpg_export_dims"] == meta_b["jpg_export_dims"]
    assert meta_a["jpg_downscale_iterations"] == meta_b["jpg_downscale_iterations"]


def test_jpeg_export_failure_includes_reason_when_cap_is_unreachable() -> None:
    rng = np.random.default_rng(123)
    image = rng.random((800, 1200, 3), dtype=np.float32)
    with pytest.raises(JpegExportError) as exc_info:
        encode_jpeg_to_target(
            image_srgb_f32=image,
            config=JpegExportConfig(max_mb=0.01, quality_max=92, quality_min=90, quality_step=1, downscale_step=0.90, min_side=780),
        )
    assert exc_info.value.metadata["export_failure_reason"] == "jpg-cap-unmet-at-min-side"
    assert exc_info.value.metadata["jpg_target_met"] is False
