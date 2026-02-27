from __future__ import annotations

import base64
import io
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from restore_batch.models import ImageContext
from restore_batch.openai_repair import (
    DEFAULT_OPENAI_REPAIR_PROMPT_BW,
    DEFAULT_OPENAI_REPAIR_PROMPT_COLOR,
    OpenAIRepairConfig,
    OpenAIRepairStage,
)


def _image_gradient(height: int = 64, width: int = 64) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    xn = xx / max(width - 1, 1)
    yn = yy / max(height - 1, 1)
    r = 0.22 + (0.55 * xn)
    g = 0.20 + (0.42 * xn) + (0.06 * yn)
    b = 0.18 + (0.30 * xn) + (0.08 * yn)
    return np.clip(np.stack((r, g, b), axis=2), 0.0, 1.0).astype(np.float32)


def _solid_image(value: float = 0.5, height: int = 64, width: int = 64) -> np.ndarray:
    return np.full((height, width, 3), fill_value=value, dtype=np.float32)


def _ctx(*, image: np.ndarray, image_type: str = "true-color") -> ImageContext:
    return ImageContext(
        input_path=Path("/tmp/in.jpg"),
        output_path=Path("/tmp/out.jpg"),
        image_f32=image.copy(),
        metadata={"image_type": image_type},
    )


def _b64_png(rgb_f32: np.ndarray) -> str:
    u8 = np.clip(np.round(rgb_f32 * 255.0), 0.0, 255.0).astype(np.uint8)
    with io.BytesIO() as handle:
        Image.fromarray(u8, mode="RGB").save(handle, format="PNG")
        return base64.b64encode(handle.getvalue()).decode("ascii")


def _install_fake_openai(monkeypatch: pytest.MonkeyPatch, edit_callable) -> None:
    fake_module = ModuleType("openai")

    class FakeOpenAI:
        def __init__(self, api_key: str) -> None:
            self.api_key = api_key
            self.images = SimpleNamespace(edit=edit_callable)

    setattr(fake_module, "OpenAI", FakeOpenAI)
    monkeypatch.setitem(sys.modules, "openai", fake_module)


def test_prompt_selection_uses_color_default_for_true_color(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    stage = OpenAIRepairStage(OpenAIRepairConfig(mode="on", failure_mode="fail-open"))
    context = _ctx(image=_image_gradient(), image_type="true-color")
    stage.process(context)

    assert context.metadata["openai_repair_prompt_mode"] == "color-default"
    assert context.metadata["openai_repair_prompt"] == DEFAULT_OPENAI_REPAIR_PROMPT_COLOR


def test_prompt_selection_uses_bw_default_for_near_grayscale(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    stage = OpenAIRepairStage(OpenAIRepairConfig(mode="on", failure_mode="fail-open"))
    context = _ctx(image=_solid_image(), image_type="near-grayscale")
    stage.process(context)

    assert context.metadata["openai_repair_prompt_mode"] == "bw-default"
    assert context.metadata["openai_repair_prompt"] == DEFAULT_OPENAI_REPAIR_PROMPT_BW


def test_prompt_selection_uses_override_when_provided(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    override = "custom repair prompt"
    stage = OpenAIRepairStage(
        OpenAIRepairConfig(
            mode="on",
            failure_mode="fail-open",
            prompt_override=override,
        )
    )
    context = _ctx(image=_image_gradient(), image_type="true-color")
    stage.process(context)

    assert context.metadata["openai_repair_prompt_mode"] == "override"
    assert context.metadata["openai_repair_prompt"] == override


def test_mode_off_keeps_image_and_sets_disabled_metadata() -> None:
    image = _image_gradient()
    context = _ctx(image=image)
    stage = OpenAIRepairStage(OpenAIRepairConfig(mode="off"))
    out = stage.process(context)

    assert np.array_equal(out.image_f32, image)
    assert out.metadata["openai_repair_enabled"] is False
    assert out.metadata["openai_repair_skipped_reason"] == "openai-repair-off"


def test_missing_api_key_fail_open_skips_without_mutation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    image = _image_gradient()
    context = _ctx(image=image)
    stage = OpenAIRepairStage(OpenAIRepairConfig(mode="on", failure_mode="fail-open"))
    out = stage.process(context)

    assert np.array_equal(out.image_f32, image)
    assert out.metadata["openai_repair_applied"] is False
    assert out.metadata["openai_repair_skipped_reason"] == "api-key-missing"


def test_missing_api_key_fail_closed_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    stage = OpenAIRepairStage(OpenAIRepairConfig(mode="on", failure_mode="fail-closed"))
    context = _ctx(image=_image_gradient())
    with pytest.raises(RuntimeError):
        stage.process(context)
    assert context.metadata["openai_repair_skipped_reason"] == "api-key-missing"
    assert context.metadata["openai_repair_applied"] is False


def test_auto_mask_empty_candidate_skips(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    calls: list[dict[str, object]] = []

    def _edit(**kwargs):
        calls.append(kwargs)
        return {"data": [{"b64_json": _b64_png(_solid_image(0.5, 64, 64))}]}

    _install_fake_openai(monkeypatch, _edit)
    stage = OpenAIRepairStage(
        OpenAIRepairConfig(
            mode="on",
            mask_mode="auto",
            failure_mode="fail-open",
        )
    )
    context = _ctx(image=_solid_image(0.5, 64, 64))
    out = stage.process(context)

    assert out.metadata["openai_repair_applied"] is False
    assert out.metadata["openai_repair_skipped_reason"] == "no-damage-candidates"
    assert len(calls) == 0


def test_auto_mask_empty_candidate_fail_closed_still_skips(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    calls: list[dict[str, object]] = []

    def _edit(**kwargs):
        calls.append(kwargs)
        return {"data": [{"b64_json": _b64_png(_solid_image(0.5, 64, 64))}]}

    _install_fake_openai(monkeypatch, _edit)
    stage = OpenAIRepairStage(
        OpenAIRepairConfig(
            mode="on",
            mask_mode="auto",
            failure_mode="fail-closed",
        )
    )
    context = _ctx(image=_solid_image(0.5, 64, 64))
    out = stage.process(context)

    assert out.metadata["openai_repair_applied"] is False
    assert out.metadata["openai_repair_skipped_reason"] == "no-damage-candidates"
    assert len(calls) == 0


def test_auto_mask_overcoverage_skips(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    calls: list[dict[str, object]] = []

    def _edit(**kwargs):
        calls.append(kwargs)
        return {"data": [{"b64_json": _b64_png(_solid_image(0.6, 64, 64))}]}

    _install_fake_openai(monkeypatch, _edit)
    noisy = _image_gradient(64, 64)
    noisy[8:56, 8:56, 0] = 0.95
    noisy[8:56, 8:56, 1] = 0.10
    noisy[8:56, 8:56, 2] = 0.10
    stage = OpenAIRepairStage(
        OpenAIRepairConfig(
            mode="on",
            mask_mode="auto",
            failure_mode="fail-open",
            min_mask_px=1,
            min_mask_fraction=0.0,
            max_mask_fraction=0.0001,
        )
    )
    context = _ctx(image=noisy)
    out = stage.process(context)

    assert out.metadata["openai_repair_applied"] is False
    assert out.metadata["openai_repair_skipped_reason"] == "mask-too-large-risk"
    assert len(calls) == 0


def test_mocked_successful_edit_applies_result(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    calls: list[dict[str, object]] = []
    repaired = np.clip(_image_gradient(64, 64) * 0.6, 0.0, 1.0).astype(np.float32)

    def _edit(**kwargs):
        calls.append(kwargs)
        return {"data": [{"b64_json": _b64_png(repaired)}]}

    _install_fake_openai(monkeypatch, _edit)
    stage = OpenAIRepairStage(
        OpenAIRepairConfig(
            mode="on",
            mask_mode="none",
            failure_mode="fail-open",
        )
    )
    context = _ctx(image=_image_gradient(64, 64))
    out = stage.process(context)

    assert out.metadata["openai_repair_applied"] is True
    assert out.metadata["openai_repair_request_attempts"] == 1
    assert len(calls) == 1
    assert not np.array_equal(out.image_f32, _image_gradient(64, 64))


def test_input_fidelity_retry_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    calls: list[dict[str, object]] = []
    repaired = np.clip(_image_gradient(64, 64) * 0.9, 0.0, 1.0).astype(np.float32)

    def _edit(**kwargs):
        calls.append(kwargs)
        if "input_fidelity" in kwargs:
            raise RuntimeError("input_fidelity unsupported")
        return {"data": [{"b64_json": _b64_png(repaired)}]}

    _install_fake_openai(monkeypatch, _edit)
    stage = OpenAIRepairStage(
        OpenAIRepairConfig(
            mode="on",
            mask_mode="none",
            failure_mode="fail-open",
            input_fidelity="high",
        )
    )
    context = _ctx(image=_image_gradient(64, 64))
    out = stage.process(context)

    assert out.metadata["openai_repair_applied"] is True
    assert out.metadata["openai_repair_retry_without_input_fidelity"] is True
    assert out.metadata["openai_repair_request_attempts"] == 2
    assert len(calls) == 2
    assert "input_fidelity" in calls[0]
    assert "input_fidelity" not in calls[1]


def test_output_size_normalization_to_input_dims(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    resized_response = np.clip(_image_gradient(32, 32), 0.0, 1.0).astype(np.float32)

    def _edit(**kwargs):
        _ = kwargs
        return {"data": [{"b64_json": _b64_png(resized_response)}]}

    _install_fake_openai(monkeypatch, _edit)
    stage = OpenAIRepairStage(
        OpenAIRepairConfig(
            mode="on",
            mask_mode="none",
            failure_mode="fail-open",
        )
    )
    image = _image_gradient(20, 12)
    context = _ctx(image=image)
    out = stage.process(context)

    assert out.metadata["openai_repair_applied"] is True
    assert out.metadata["openai_repair_output_resized_to_input"] is True
    assert out.image_f32 is not None
    assert out.image_f32.shape[:2] == image.shape[:2]


def test_structure_guard_skips_large_geometry_change(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    calls: list[dict[str, object]] = []
    rng = np.random.default_rng(0)
    distorted = rng.random((64, 64, 3), dtype=np.float32)

    def _edit(**kwargs):
        calls.append(kwargs)
        return {"data": [{"b64_json": _b64_png(distorted)}]}

    _install_fake_openai(monkeypatch, _edit)
    stage = OpenAIRepairStage(
        OpenAIRepairConfig(
            mode="on",
            mask_mode="none",
            failure_mode="fail-closed",
        )
    )
    context = _ctx(image=_image_gradient(64, 64))
    out = stage.process(context)

    assert len(calls) == 1
    assert out.metadata["openai_repair_applied"] is False
    assert out.metadata["openai_repair_skipped_reason"] == "structure-change-risk"
    assert out.metadata["openai_repair_structure_guard_triggered"] is True


def test_masked_merge_preserves_unmasked_regions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    source = _solid_image(0.50, 64, 64)
    repaired = source.copy()
    repaired[:, :, :] = 0.10

    calls: list[dict[str, object]] = []

    def _edit(**kwargs):
        calls.append(kwargs)
        return {"data": [{"b64_json": _b64_png(repaired)}]}

    _install_fake_openai(monkeypatch, _edit)
    stage = OpenAIRepairStage(
        OpenAIRepairConfig(
            mode="on",
            mask_mode="auto",
            failure_mode="fail-open",
            min_mask_px=1,
            min_mask_fraction=0.0,
            max_mask_fraction=1.0,
        )
    )

    # Force deterministic small central mask.
    import restore_batch.openai_repair as repair_mod

    def _fake_mask(**kwargs):  # noqa: ANN003
        _ = kwargs
        mask = np.zeros((64, 64), dtype=bool)
        mask[24:40, 24:40] = True
        return mask, {"openai_repair_mask_generator": "test"}

    monkeypatch.setattr(repair_mod, "generate_auto_repair_mask", _fake_mask)

    context = _ctx(image=source)
    out = stage.process(context)
    assert out.metadata["openai_repair_applied"] is True
    assert out.metadata["openai_repair_merge_strategy"] == "masked-feather-blend"

    out_img = out.image_f32
    assert out_img is not None
    # Far-away corner should remain source-like due masked merge.
    assert float(np.max(np.abs(out_img[:8, :8, :] - source[:8, :8, :]))) < 1e-4
    # Center mask region should change.
    assert float(np.mean(out_img[28:36, 28:36, :])) < 0.45
