from __future__ import annotations

import numpy as np

from restore_batch.redeye import correct_redeye_patch, red_eye_mask


def test_red_eye_mask_uses_configurable_red_excess_threshold() -> None:
    patch = np.zeros((5, 5, 3), dtype=np.float32)
    patch[:, :, 0] = 0.30
    patch[:, :, 1] = 0.24
    patch[:, :, 2] = 0.22

    loose = red_eye_mask(patch_rgb=patch, red_ratio=1.1, min_red=0.20, min_red_excess=0.05)
    strict = red_eye_mask(patch_rgb=patch, red_ratio=1.1, min_red=0.20, min_red_excess=0.08)
    assert int(np.count_nonzero(loose)) > int(np.count_nonzero(strict))


def test_correct_redeye_patch_neutralizes_green_cast_risk() -> None:
    patch = np.zeros((8, 8, 3), dtype=np.float32)
    patch[:, :, 0] = 0.62
    patch[:, :, 1] = 0.23
    patch[:, :, 2] = 0.12
    alpha = np.ones((8, 8), dtype=np.float32)

    corrected = correct_redeye_patch(patch_rgb=patch, alpha=alpha, darken_factor=0.55)

    # Red should come down from strong red-eye values.
    assert float(np.mean(corrected[:, :, 0])) < float(np.mean(patch[:, :, 0]))
    # G/B should be pulled toward each other to avoid green/cyan residual tint.
    assert float(np.mean(np.abs(corrected[:, :, 1] - corrected[:, :, 2]))) < float(
        np.mean(np.abs(patch[:, :, 1] - patch[:, :, 2]))
    )
    # Prevent strong green-over-red shift in corrected regions.
    assert float(np.mean(corrected[:, :, 1] - corrected[:, :, 0])) <= 0.03
