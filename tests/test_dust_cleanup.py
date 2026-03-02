from __future__ import annotations

import numpy as np

from restore_batch.dust import DustCleanupConfig, DustCleanupStage
from restore_batch.models import ImageContext


def test_dust_cleanup_removes_impulse_specks_and_preserves_neutrality(tmp_path) -> None:
    image = np.full((72, 72, 3), 0.5, dtype=np.float32)
    image[18, 20, :] = 0.0
    image[40, 42, :] = 1.0
    image[55, 12, :] = 0.0

    stage = DustCleanupStage(
        DustCleanupConfig(
            mode="on",
            mad_multiplier=2.5,
            min_contrast=0.010,
            min_component_px=1,
            max_component_px=16,
            max_mask_fraction=0.20,
        )
    )
    context = ImageContext(
        input_path=tmp_path / "in.jpg",
        output_path=tmp_path / "out.jpg",
        image_f32=image.copy(),
        metadata={"image_type": "near-grayscale"},
    )

    out = stage.process(context)
    assert out.image_f32 is not None
    assert out.metadata["dust_clean_applied"] is True
    assert int(out.metadata["dust_clean_mask_pixels"]) >= 3
    # Near-grayscale workflow should remain neutral.
    assert np.max(np.abs(out.image_f32[:, :, 0] - out.image_f32[:, :, 1])) < 1e-6
    assert np.max(np.abs(out.image_f32[:, :, 1] - out.image_f32[:, :, 2])) < 1e-6
    # Specks should move toward neighborhood luminance.
    assert float(out.image_f32[18, 20, 0]) > 0.10
    assert float(out.image_f32[40, 42, 0]) < 0.90


def test_dust_cleanup_mode_off_is_noop(tmp_path) -> None:
    image = np.full((32, 32, 3), 0.4, dtype=np.float32)
    stage = DustCleanupStage(DustCleanupConfig(mode="off"))
    context = ImageContext(
        input_path=tmp_path / "in.jpg",
        output_path=tmp_path / "out.jpg",
        image_f32=image.copy(),
        metadata={"image_type": "true-color"},
    )
    out = stage.process(context)
    assert out.image_f32 is not None
    assert np.allclose(out.image_f32, image)
    assert out.metadata["dust_clean_applied"] is False
    assert out.metadata["dust_clean_skipped_reason"] == "dust-clean-off"


def test_dust_cleanup_guardrail_skips_when_mask_risk_high(tmp_path) -> None:
    yy, xx = np.mgrid[0:64, 0:64]
    image = np.stack(
        (
            ((xx % 2) * 1.0),
            ((yy % 2) * 1.0),
            (((xx + yy) % 2) * 1.0),
        ),
        axis=-1,
    ).astype(np.float32)

    stage = DustCleanupStage(
        DustCleanupConfig(
            mode="on",
            mad_multiplier=1.5,
            min_contrast=0.001,
            min_component_px=1,
            max_component_px=2000,
            max_mask_fraction=0.0001,
        )
    )
    context = ImageContext(
        input_path=tmp_path / "in.jpg",
        output_path=tmp_path / "out.jpg",
        image_f32=image.copy(),
        metadata={"image_type": "true-color"},
    )
    out = stage.process(context)
    assert out.metadata["dust_clean_applied"] is False
    assert str(out.metadata["dust_clean_skipped_reason"]) in {
        "candidate-mask-too-large-risk",
        "mask-too-large-risk",
    }
