from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from restore_batch.classification import ClassificationConfig, ImageTypeClassificationStage
from restore_batch.models import ImageContext


def _load_testpic(name: str) -> np.ndarray:
    root = Path(__file__).resolve().parents[1]
    path = root / "TestPics" / name
    arr = np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 255.0
    return arr.astype(np.float32, copy=False)


def test_sepia_like_image_is_classified_near_grayscale_with_gate_enabled() -> None:
    image = _load_testpic("Image_20260218_0011.jpg")
    stage = ImageTypeClassificationStage(ClassificationConfig())
    ctx = ImageContext(input_path=Path("in.jpg"), output_path=Path("out.png"), image_f32=image)
    stage.process(ctx)

    assert ctx.metadata["image_type"] == "near-grayscale"
    assert ctx.metadata["classification_sepia_gate_enabled"] is True
    assert ctx.metadata["classification_method"] == "chromaticity-variance+sepia-monochrome-gate"
    assert ctx.metadata["classification_sepia_like"] is True


def test_same_image_reverts_to_true_color_when_sepia_gate_disabled() -> None:
    image = _load_testpic("Image_20260218_0011.jpg")
    stage = ImageTypeClassificationStage(ClassificationConfig(sepia_gate_enabled=False))
    ctx = ImageContext(input_path=Path("in.jpg"), output_path=Path("out.png"), image_f32=image)
    stage.process(ctx)

    assert ctx.metadata["classification_sepia_gate_enabled"] is False
    assert ctx.metadata["classification_sepia_like"] is False
    assert ctx.metadata["image_type"] == "true-color"
