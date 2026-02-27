from __future__ import annotations

import base64
import io
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .color_math import rec709_luminance, srgb_to_linear
from .models import ImageContext

DEFAULT_OPENAI_REPAIR_PROMPT_COLOR = (
    "Please remove the stains and damage to the attached picture. "
    "Also fix any color cast, should it be calculated to exist. "
    "Also remove any visible photo paper texture. "
    "Leave everything else the same, especially the faces."
)
DEFAULT_OPENAI_REPAIR_PROMPT_BW = (
    "Please remove the stains and damage to the attached picture. "
    "Make sure the picture has a full tonal range. "
    "Also remove any visible photo paper texture. "
    "Leave everything else the same, especially the faces."
)
OPENAI_IMAGE_SIZE_CHOICES: tuple[tuple[int, int], ...] = (
    (1024, 1024),
    (1536, 1024),
    (1024, 1536),
)


@dataclass(frozen=True)
class CanvasMapping:
    source_w: int
    source_h: int
    canvas_w: int
    canvas_h: int
    placed_x: int
    placed_y: int
    placed_w: int
    placed_h: int


@dataclass(frozen=True)
class OpenAIRepairConfig:
    mode: str = "off"
    mask_mode: str = "none"
    aggressiveness: str = "conservative"
    model: str = "gpt-image-1"
    quality: str = "medium"
    input_fidelity: str = "high"
    size: str = "auto"
    prompt_color: str = DEFAULT_OPENAI_REPAIR_PROMPT_COLOR
    prompt_bw: str = DEFAULT_OPENAI_REPAIR_PROMPT_BW
    prompt_override: str | None = None
    failure_mode: str = "fail-closed"
    min_mask_px: int = 96
    min_mask_fraction: float = 0.0005
    max_mask_fraction: float = 0.20
    structure_guard_enabled: bool = True
    structure_guard_min_grad_corr: float = 0.40
    structure_guard_min_edge_iou: float = 0.18
    save_mask_preview: bool = False
    mask_preview_subdir: str = "_openai_repair_masks"


class OpenAIRepairStage:
    name = "openai-repair"

    def __init__(self, config: OpenAIRepairConfig) -> None:
        self.config = config

    def process(self, context: ImageContext) -> ImageContext:
        if context.image_f32 is None:
            raise RuntimeError("OpenAI repair stage expected image data.")

        if self.config.mode == "off":
            context.metadata.update(openai_repair_disabled_metadata(config=self.config))
            return context

        source = np.clip(context.image_f32, 0.0, 1.0).astype(np.float32, copy=False)
        image_type = str(context.metadata.get("image_type", "true-color"))
        prompt_mode, prompt = choose_repair_prompt(config=self.config, image_type=image_type)

        metadata: dict[str, Any] = {
            "openai_repair_requested": True,
            "openai_repair_enabled": True,
            "openai_repair_applied": False,
            "openai_repair_failure_mode": self.config.failure_mode,
            "openai_repair_model": self.config.model,
            "openai_repair_quality": self.config.quality,
            "openai_repair_input_fidelity": self.config.input_fidelity,
            "openai_repair_size": self.config.size,
            "openai_repair_mask_mode": self.config.mask_mode,
            "openai_repair_aggressiveness": self.config.aggressiveness,
            "openai_repair_face_preserve_applied": False,
            "openai_repair_face_preserve_detector_available": False,
            "openai_repair_face_preserve_detector_warning": None,
            "openai_repair_face_preserve_face_count": 0,
            "openai_repair_prompt_mode": prompt_mode,
            "openai_repair_prompt": prompt,
            "openai_repair_input_size": [int(source.shape[1]), int(source.shape[0])],
            "openai_repair_output_size": [int(source.shape[1]), int(source.shape[0])],
            "openai_repair_output_raw_size": None,
            "openai_repair_output_resized_to_input": False,
            "openai_repair_resize_strategy": "none",
            "openai_repair_merge_strategy": "none",
            "openai_repair_merge_alpha_mean": 0.0,
            "openai_repair_merge_alpha_max": 0.0,
            "openai_repair_api_size_request": self.config.size,
            "openai_repair_api_canvas_size": None,
            "openai_repair_api_canvas_scale": 1.0,
            "openai_repair_api_canvas_offset_xy": [0, 0],
            "openai_repair_api_canvas_inner_size": [int(source.shape[1]), int(source.shape[0])],
            "openai_repair_mask_pixels": 0,
            "openai_repair_mask_fraction": 0.0,
            "openai_repair_mask_preview_path": None,
            "openai_repair_min_mask_px": int(max(0, self.config.min_mask_px)),
            "openai_repair_min_mask_fraction": float(max(0.0, self.config.min_mask_fraction)),
            "openai_repair_max_mask_fraction": float(np.clip(self.config.max_mask_fraction, 0.0, 1.0)),
            "openai_repair_structure_guard_enabled": bool(self.config.structure_guard_enabled),
            "openai_repair_structure_guard_triggered": False,
            "openai_repair_structure_grad_corr": None,
            "openai_repair_structure_edge_iou": None,
            "openai_repair_structure_min_grad_corr": float(self.config.structure_guard_min_grad_corr),
            "openai_repair_structure_min_edge_iou": float(self.config.structure_guard_min_edge_iou),
            "openai_repair_retry_without_input_fidelity": False,
            "openai_repair_request_attempts": 0,
            "openai_repair_latency_ms": None,
        }

        api_key = str(os.getenv("OPENAI_API_KEY", "")).strip()
        metadata["openai_repair_api_key_present"] = bool(api_key)
        if not api_key:
            return self._skip_or_raise(
                context=context,
                metadata=metadata,
                reason="api-key-missing",
                error="OPENAI_API_KEY is not set.",
            )

        try:
            from openai import OpenAI  # type: ignore
        except Exception as exc:  # noqa: BLE001 - optional dependency.
            return self._skip_or_raise(
                context=context,
                metadata=metadata,
                reason="openai-package-missing",
                error=f"OpenAI package import failed: {exc}",
            )

        if self.config.mask_mode == "auto":
            repair_mask, mask_meta = generate_auto_repair_mask(source=source, aggressiveness=self.config.aggressiveness)
            metadata.update(mask_meta)
            mask_pixels = int(np.count_nonzero(repair_mask))
            mask_fraction = float(mask_pixels / max(repair_mask.size, 1))
            metadata["openai_repair_mask_pixels"] = mask_pixels
            metadata["openai_repair_mask_fraction"] = mask_fraction

            min_mask_px = int(max(0, self.config.min_mask_px))
            min_mask_fraction = float(max(0.0, self.config.min_mask_fraction))
            max_mask_fraction = float(np.clip(self.config.max_mask_fraction, 0.0, 1.0))

            if mask_pixels < min_mask_px or mask_fraction < min_mask_fraction:
                return self._skip_or_raise(
                    context=context,
                    metadata=metadata,
                    reason="no-damage-candidates",
                    error=None,
                    raise_on_fail_closed=False,
                )
            if mask_fraction > max_mask_fraction:
                return self._skip_or_raise(
                    context=context,
                    metadata=metadata,
                    reason="mask-too-large-risk",
                    error=None,
                    raise_on_fail_closed=False,
                )

            if self.config.save_mask_preview:
                preview_path = save_mask_preview(
                    mask=repair_mask,
                    output_path=context.output_path,
                    subdir=self.config.mask_preview_subdir,
                )
                metadata["openai_repair_mask_preview_path"] = str(preview_path)
        else:
            repair_mask = None

        api_size_request, (canvas_w, canvas_h) = resolve_api_size_request(
            size_spec=self.config.size,
            source_w=int(source.shape[1]),
            source_h=int(source.shape[0]),
        )
        source_canvas, mask_canvas, mapping = prepare_canvas_inputs(
            source=source,
            repair_mask=repair_mask,
            canvas_w=canvas_w,
            canvas_h=canvas_h,
        )
        metadata["openai_repair_api_size_request"] = api_size_request
        metadata["openai_repair_api_canvas_size"] = [int(canvas_w), int(canvas_h)]
        metadata["openai_repair_api_canvas_scale"] = float(mapping.placed_w / max(mapping.source_w, 1))
        metadata["openai_repair_api_canvas_offset_xy"] = [int(mapping.placed_x), int(mapping.placed_y)]
        metadata["openai_repair_api_canvas_inner_size"] = [int(mapping.placed_w), int(mapping.placed_h)]

        client = OpenAI(api_key=api_key)
        source_png = rgb_f32_to_png_bytes(source_canvas)
        mask_png = mask_to_openai_png_bytes(mask_canvas) if mask_canvas is not None else None

        started = time.perf_counter()
        try:
            response, request_meta = invoke_openai_edit_with_retry(
                client=client,
                source_png=source_png,
                mask_png=mask_png,
                model=self.config.model,
                prompt=prompt,
                quality=self.config.quality,
                size=api_size_request,
                input_fidelity=self.config.input_fidelity,
            )
            metadata.update(request_meta)
            repaired_raw = decode_openai_edit_output(response)
        except Exception as exc:  # noqa: BLE001 - fail-open/closed controlled by config.
            metadata["openai_repair_latency_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
            return self._skip_or_raise(
                context=context,
                metadata=metadata,
                reason="api-error",
                error=str(exc),
            )

        metadata["openai_repair_latency_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
        repaired_raw = np.clip(repaired_raw, 0.0, 1.0).astype(np.float32, copy=False)
        metadata["openai_repair_output_raw_size"] = [int(repaired_raw.shape[1]), int(repaired_raw.shape[0])]

        repaired = recover_from_api_canvas(
            repaired_raw=repaired_raw,
            mapping=mapping,
        )
        repaired = np.clip(repaired, 0.0, 1.0).astype(np.float32, copy=False)

        resized_any = (
            (int(repaired_raw.shape[1]) != int(mapping.source_w))
            or (int(repaired_raw.shape[0]) != int(mapping.source_h))
            or (int(repaired_raw.shape[1]) != int(mapping.canvas_w))
            or (int(repaired_raw.shape[0]) != int(mapping.canvas_h))
            or (int(mapping.source_w) != int(mapping.placed_w))
            or (int(mapping.source_h) != int(mapping.placed_h))
        )
        metadata["openai_repair_output_resized_to_input"] = bool(resized_any)
        metadata["openai_repair_resize_strategy"] = "letterbox-fit-crop-resize"

        if repair_mask is not None:
            alpha = build_mask_alpha(repair_mask=repair_mask, source_shape=source.shape)
            repaired = blend_repair_with_source(source=source, repaired=repaired, alpha=alpha)
            metadata["openai_repair_merge_strategy"] = "masked-feather-blend"
            metadata["openai_repair_merge_alpha_mean"] = float(np.mean(alpha))
            metadata["openai_repair_merge_alpha_max"] = float(np.max(alpha))
        else:
            metadata["openai_repair_merge_strategy"] = "full-frame-replace"
            metadata["openai_repair_merge_alpha_mean"] = 1.0
            metadata["openai_repair_merge_alpha_max"] = 1.0

        repaired, face_meta = apply_face_preservation(source=source, repaired=repaired)
        metadata.update(face_meta)

        guard_active = bool(self.config.structure_guard_enabled and (repair_mask is None))
        if guard_active:
            structure_meta = evaluate_structure_guard(
                source=source,
                candidate=repaired,
                min_grad_corr=float(self.config.structure_guard_min_grad_corr),
                min_edge_iou=float(self.config.structure_guard_min_edge_iou),
            )
            metadata.update(structure_meta)
            if bool(structure_meta["openai_repair_structure_guard_triggered"]):
                return self._skip_or_raise(
                    context=context,
                    metadata=metadata,
                    reason="structure-change-risk",
                    error=None,
                    raise_on_fail_closed=False,
                )
        else:
            metadata["openai_repair_structure_guard_enabled"] = bool(guard_active)
            metadata["openai_repair_structure_guard_triggered"] = False
            metadata["openai_repair_structure_grad_corr"] = None
            metadata["openai_repair_structure_edge_iou"] = None

        metadata["openai_repair_applied"] = True
        metadata["openai_repair_output_size"] = [int(repaired.shape[1]), int(repaired.shape[0])]
        metadata["openai_repair_skipped_reason"] = None
        context.image_f32 = repaired
        context.metadata.update(metadata)
        return context

    def _skip_or_raise(
        self,
        *,
        context: ImageContext,
        metadata: dict[str, Any],
        reason: str,
        error: str | None,
        raise_on_fail_closed: bool = True,
    ) -> ImageContext:
        metadata["openai_repair_applied"] = False
        metadata["openai_repair_skipped_reason"] = reason
        if error is not None:
            metadata["openai_repair_error"] = error
        context.metadata.update(metadata)
        if (not raise_on_fail_closed) or self.config.failure_mode == "fail-open":
            return context
        raise RuntimeError(f"OpenAI repair failed ({reason}): {error or 'no additional error detail'}")


def choose_repair_prompt(*, config: OpenAIRepairConfig, image_type: str) -> tuple[str, str]:
    override = (config.prompt_override or "").strip()
    if override:
        return "override", override
    if image_type == "true-color":
        return "color-default", config.prompt_color
    return "bw-default", config.prompt_bw


def openai_repair_disabled_metadata(*, config: OpenAIRepairConfig) -> dict[str, Any]:
    return {
        "openai_repair_requested": False,
        "openai_repair_enabled": False,
        "openai_repair_applied": False,
        "openai_repair_failure_mode": config.failure_mode,
        "openai_repair_model": config.model,
        "openai_repair_quality": config.quality,
        "openai_repair_input_fidelity": config.input_fidelity,
        "openai_repair_size": config.size,
        "openai_repair_mask_mode": config.mask_mode,
        "openai_repair_aggressiveness": config.aggressiveness,
        "openai_repair_face_preserve_applied": False,
        "openai_repair_face_preserve_detector_available": False,
        "openai_repair_face_preserve_detector_warning": None,
        "openai_repair_face_preserve_face_count": 0,
        "openai_repair_prompt_mode": "disabled",
        "openai_repair_prompt": None,
        "openai_repair_mask_pixels": 0,
        "openai_repair_mask_fraction": 0.0,
        "openai_repair_mask_preview_path": None,
        "openai_repair_min_mask_px": int(max(0, config.min_mask_px)),
        "openai_repair_min_mask_fraction": float(max(0.0, config.min_mask_fraction)),
        "openai_repair_max_mask_fraction": float(np.clip(config.max_mask_fraction, 0.0, 1.0)),
        "openai_repair_structure_guard_enabled": bool(config.structure_guard_enabled),
        "openai_repair_structure_guard_triggered": False,
        "openai_repair_structure_grad_corr": None,
        "openai_repair_structure_edge_iou": None,
        "openai_repair_structure_min_grad_corr": float(config.structure_guard_min_grad_corr),
        "openai_repair_structure_min_edge_iou": float(config.structure_guard_min_edge_iou),
        "openai_repair_retry_without_input_fidelity": False,
        "openai_repair_request_attempts": 0,
        "openai_repair_skipped_reason": "openai-repair-off",
    }


def invoke_openai_edit_with_retry(
    *,
    client: Any,
    source_png: bytes,
    mask_png: bytes | None,
    model: str,
    prompt: str,
    quality: str,
    size: str,
    input_fidelity: str,
) -> tuple[Any, dict[str, Any]]:
    attempts = 0
    retry_without_fidelity = False

    try:
        attempts += 1
        response = call_openai_edit(
            client=client,
            source_png=source_png,
            mask_png=mask_png,
            model=model,
            prompt=prompt,
            quality=quality,
            size=size,
            input_fidelity=input_fidelity,
            include_input_fidelity=True,
        )
    except Exception as exc:  # noqa: BLE001
        if should_retry_without_input_fidelity(exc):
            retry_without_fidelity = True
            attempts += 1
            response = call_openai_edit(
                client=client,
                source_png=source_png,
                mask_png=mask_png,
                model=model,
                prompt=prompt,
                quality=quality,
                size=size,
                input_fidelity=input_fidelity,
                include_input_fidelity=False,
            )
        else:
            raise

    return response, {
        "openai_repair_request_attempts": int(attempts),
        "openai_repair_retry_without_input_fidelity": bool(retry_without_fidelity),
    }


def call_openai_edit(
    *,
    client: Any,
    source_png: bytes,
    mask_png: bytes | None,
    model: str,
    prompt: str,
    quality: str,
    size: str,
    input_fidelity: str,
    include_input_fidelity: bool,
) -> Any:
    kwargs: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "image": named_bytes_io(source_png, "input.png"),
        "quality": quality,
        "size": size,
        "output_format": "png",
        "n": 1,
    }
    if mask_png is not None:
        kwargs["mask"] = named_bytes_io(mask_png, "mask.png")
    if include_input_fidelity:
        kwargs["input_fidelity"] = input_fidelity
    return client.images.edit(**kwargs)


def should_retry_without_input_fidelity(exc: Exception) -> bool:
    text = str(exc).lower()
    return ("input_fidelity" in text) and (
        ("unsupported" in text)
        or ("not support" in text)
        or ("not allowed" in text)
        or ("invalid" in text)
        or ("unknown" in text)
    )


def decode_openai_edit_output(response: Any) -> np.ndarray:
    data = response.get("data") if isinstance(response, dict) else getattr(response, "data", None)
    if not data:
        raise RuntimeError("OpenAI image edit returned no data entries.")
    first = data[0]
    b64_payload = first.get("b64_json") if isinstance(first, dict) else getattr(first, "b64_json", None)
    if not b64_payload:
        raise RuntimeError("OpenAI image edit response missing b64_json payload.")
    raw = base64.b64decode(b64_payload)
    with Image.open(io.BytesIO(raw)) as image:
        rgb = image.convert("RGB")
        arr = np.asarray(rgb).astype(np.float32) / 255.0
    return arr.astype(np.float32, copy=False)


def named_bytes_io(data: bytes, name: str) -> io.BytesIO:
    buf = io.BytesIO(data)
    buf.name = name  # type: ignore[attr-defined]
    return buf


def rgb_f32_to_png_bytes(image_rgb_f32: np.ndarray) -> bytes:
    u8 = np.clip(np.round(image_rgb_f32 * 255.0), 0.0, 255.0).astype(np.uint8)
    with io.BytesIO() as handle:
        Image.fromarray(u8, mode="RGB").save(handle, format="PNG")
        return handle.getvalue()


def mask_to_openai_png_bytes(mask: np.ndarray) -> bytes:
    h, w = mask.shape
    rgba = np.full((h, w, 4), fill_value=255, dtype=np.uint8)
    rgba[:, :, 3] = np.where(mask, 0, 255).astype(np.uint8)
    with io.BytesIO() as handle:
        Image.fromarray(rgba, mode="RGBA").save(handle, format="PNG")
        return handle.getvalue()


def resize_rgb_f32(image_rgb_f32: np.ndarray, *, target_w: int, target_h: int) -> np.ndarray:
    u8 = np.clip(np.round(image_rgb_f32 * 255.0), 0.0, 255.0).astype(np.uint8)
    resized = Image.fromarray(u8, mode="RGB").resize((target_w, target_h), resample=Image.Resampling.BICUBIC)
    return (np.asarray(resized).astype(np.float32) / 255.0).astype(np.float32, copy=False)


def save_mask_preview(*, mask: np.ndarray, output_path: Path, subdir: str) -> Path:
    preview_dir = output_path.parent / subdir
    preview_dir.mkdir(parents=True, exist_ok=True)
    preview_path = preview_dir / f"{output_path.stem}.png"
    mask_u8 = np.where(mask, 255, 0).astype(np.uint8)
    Image.fromarray(mask_u8, mode="L").save(preview_path, format="PNG")
    return preview_path


def resolve_api_size_request(*, size_spec: str, source_w: int, source_h: int) -> tuple[str, tuple[int, int]]:
    size_spec = str(size_spec).strip().lower()
    if size_spec != "auto":
        w, h = parse_size_spec(size_spec)
        return size_spec, (w, h)

    src_aspect = float(source_w / max(source_h, 1))
    best_w, best_h = OPENAI_IMAGE_SIZE_CHOICES[0]
    best_score = float("inf")
    for cand_w, cand_h in OPENAI_IMAGE_SIZE_CHOICES:
        cand_aspect = float(cand_w / max(cand_h, 1))
        # Aspect closeness in log-space avoids directional bias.
        score = abs(float(np.log(max(src_aspect, 1e-8))) - float(np.log(max(cand_aspect, 1e-8))))
        if score < best_score:
            best_score = score
            best_w, best_h = cand_w, cand_h
    return f"{best_w}x{best_h}", (best_w, best_h)


def parse_size_spec(size_spec: str) -> tuple[int, int]:
    parts = size_spec.split("x")
    if len(parts) != 2:
        raise ValueError(f"Invalid size spec: {size_spec}")
    w = int(parts[0])
    h = int(parts[1])
    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid non-positive size spec: {size_spec}")
    return w, h


def prepare_canvas_inputs(
    *,
    source: np.ndarray,
    repair_mask: np.ndarray | None,
    canvas_w: int,
    canvas_h: int,
) -> tuple[np.ndarray, np.ndarray | None, CanvasMapping]:
    source_h, source_w = source.shape[:2]
    scale = min(float(canvas_w / max(source_w, 1)), float(canvas_h / max(source_h, 1)))
    placed_w = max(1, int(round(source_w * scale)))
    placed_h = max(1, int(round(source_h * scale)))
    placed_x = max(0, int((canvas_w - placed_w) // 2))
    placed_y = max(0, int((canvas_h - placed_h) // 2))

    fill_rgb = estimate_canvas_fill(source)
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
    canvas[:, :, 0] = float(fill_rgb[0])
    canvas[:, :, 1] = float(fill_rgb[1])
    canvas[:, :, 2] = float(fill_rgb[2])

    source_resized = resize_rgb_f32(source, target_w=placed_w, target_h=placed_h)
    canvas[placed_y : placed_y + placed_h, placed_x : placed_x + placed_w, :] = source_resized

    mask_canvas: np.ndarray | None = None
    if repair_mask is not None:
        mask_canvas = np.zeros((canvas_h, canvas_w), dtype=bool)
        mask_resized = resize_mask_bool(repair_mask, target_w=placed_w, target_h=placed_h)
        mask_canvas[placed_y : placed_y + placed_h, placed_x : placed_x + placed_w] = mask_resized

    mapping = CanvasMapping(
        source_w=int(source_w),
        source_h=int(source_h),
        canvas_w=int(canvas_w),
        canvas_h=int(canvas_h),
        placed_x=int(placed_x),
        placed_y=int(placed_y),
        placed_w=int(placed_w),
        placed_h=int(placed_h),
    )
    return canvas, mask_canvas, mapping


def estimate_canvas_fill(source: np.ndarray) -> np.ndarray:
    h, w = source.shape[:2]
    if h < 2 or w < 2:
        return np.mean(source, axis=(0, 1)).astype(np.float32, copy=False)

    border_rows = [source[0, :, :], source[-1, :, :]]
    border_cols = [source[:, 0, :], source[:, -1, :]]
    border = np.concatenate(border_rows + border_cols, axis=0).astype(np.float32, copy=False)
    return np.median(border, axis=0).astype(np.float32, copy=False)


def resize_mask_bool(mask: np.ndarray, *, target_w: int, target_h: int) -> np.ndarray:
    mask_u8 = np.where(mask, 255, 0).astype(np.uint8)
    pil = Image.fromarray(mask_u8, mode="L")
    resized = pil.resize((target_w, target_h), resample=Image.Resampling.NEAREST)
    return (np.asarray(resized) >= 128).astype(bool, copy=False)


def recover_from_api_canvas(*, repaired_raw: np.ndarray, mapping: CanvasMapping) -> np.ndarray:
    repaired = np.clip(repaired_raw, 0.0, 1.0).astype(np.float32, copy=False)
    if repaired.shape[1] != mapping.canvas_w or repaired.shape[0] != mapping.canvas_h:
        repaired = resize_rgb_f32(repaired, target_w=mapping.canvas_w, target_h=mapping.canvas_h)

    x0 = int(np.clip(mapping.placed_x, 0, max(mapping.canvas_w - 1, 0)))
    y0 = int(np.clip(mapping.placed_y, 0, max(mapping.canvas_h - 1, 0)))
    x1 = int(np.clip(mapping.placed_x + mapping.placed_w, x0 + 1, mapping.canvas_w))
    y1 = int(np.clip(mapping.placed_y + mapping.placed_h, y0 + 1, mapping.canvas_h))

    cropped = repaired[y0:y1, x0:x1, :]
    if cropped.size == 0:
        cropped = repaired

    if cropped.shape[1] != mapping.source_w or cropped.shape[0] != mapping.source_h:
        cropped = resize_rgb_f32(cropped, target_w=mapping.source_w, target_h=mapping.source_h)
    return np.clip(cropped, 0.0, 1.0).astype(np.float32, copy=False)


def build_mask_alpha(*, repair_mask: np.ndarray, source_shape: tuple[int, int, int]) -> np.ndarray:
    h, w = source_shape[:2]
    mask = repair_mask
    if mask.shape != (h, w):
        mask = resize_mask_bool(mask, target_w=w, target_h=h)

    alpha = mask.astype(np.float32, copy=False)
    min_side = max(1, min(h, w))
    sigma = float(np.clip(min_side * 0.0035, 0.8, 4.0))
    alpha = gaussian_blur(alpha, sigma=sigma)
    if float(np.max(alpha)) > 1e-8:
        alpha = alpha / float(np.max(alpha))
    return np.clip(alpha, 0.0, 1.0).astype(np.float32, copy=False)


def blend_repair_with_source(*, source: np.ndarray, repaired: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    a = np.clip(alpha, 0.0, 1.0).astype(np.float32, copy=False)
    out = (source * (1.0 - a[:, :, None])) + (repaired * a[:, :, None])
    return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)


def apply_face_preservation(*, source: np.ndarray, repaired: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    boxes, detector_meta = detect_face_boxes_opencv(source=source)
    metadata: dict[str, Any] = {
        "openai_repair_face_preserve_detector_available": bool(detector_meta["available"]),
        "openai_repair_face_preserve_detector_warning": detector_meta["warning"],
        "openai_repair_face_preserve_face_count": int(len(boxes)),
        "openai_repair_face_preserve_applied": False,
    }
    if len(boxes) == 0:
        return repaired, metadata

    alpha = build_face_preserve_alpha(height=source.shape[0], width=source.shape[1], boxes=boxes)
    out = blend_repair_with_source(source=repaired, repaired=source, alpha=alpha)
    metadata["openai_repair_face_preserve_applied"] = True
    return out, metadata


def detect_face_boxes_opencv(*, source: np.ndarray) -> tuple[list[tuple[int, int, int, int]], dict[str, Any]]:
    try:
        import cv2  # type: ignore
    except Exception as exc:  # noqa: BLE001
        return [], {"available": False, "warning": f"opencv-unavailable: {exc}"}

    candidates = [
        Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml",
        Path(cv2.data.haarcascades) / "haarcascade_frontalface_alt2.xml",
    ]
    cascade_path = next((p for p in candidates if p.exists()), None)
    if cascade_path is None:
        return [], {"available": False, "warning": "opencv-haarcascade-not-found"}

    cascade = cv2.CascadeClassifier(str(cascade_path))
    if cascade.empty():
        return [], {"available": False, "warning": "opencv-haarcascade-load-failed"}

    gray = np.clip(np.mean(source, axis=2) * 255.0, 0.0, 255.0).astype(np.uint8)
    min_side = max(24, int(round(min(gray.shape[0], gray.shape[1]) * 0.04)))
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.10,
        minNeighbors=4,
        minSize=(min_side, min_side),
    )
    if faces is None or len(faces) == 0:
        return [], {"available": True, "warning": None}
    boxes = [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in np.asarray(faces, dtype=np.int32)]
    return boxes, {"available": True, "warning": None}


def build_face_preserve_alpha(*, height: int, width: int, boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    alpha = np.zeros((height, width), dtype=np.float32)
    for x, y, w, h in boxes:
        ex = int(round(w * 0.20))
        ey = int(round(h * 0.22))
        x0 = max(0, x - ex)
        y0 = max(0, y - ey)
        x1 = min(width, x + w + ex)
        y1 = min(height, y + h + ey)
        if x1 <= x0 or y1 <= y0:
            continue
        alpha[y0:y1, x0:x1] = 1.0

    sigma = max(1.0, float(min(height, width)) * 0.006)
    alpha = gaussian_blur(alpha, sigma=sigma)
    max_val = float(np.max(alpha))
    if max_val > 1e-8:
        alpha /= max_val
    return np.clip(alpha, 0.0, 1.0).astype(np.float32, copy=False)


def evaluate_structure_guard(
    *,
    source: np.ndarray,
    candidate: np.ndarray,
    min_grad_corr: float,
    min_edge_iou: float,
) -> dict[str, Any]:
    src_small = downsample_rgb_max_side(source, max_side=768)
    cand_small = downsample_rgb_max_side(candidate, max_side=768)

    src_y = rec709_luminance(srgb_to_linear(np.clip(src_small, 0.0, 1.0).astype(np.float32, copy=False)))
    cand_y = rec709_luminance(srgb_to_linear(np.clip(cand_small, 0.0, 1.0).astype(np.float32, copy=False)))

    src_grad = gradient_magnitude(src_y)
    cand_grad = gradient_magnitude(cand_y)
    src_scale = max(float(np.percentile(src_grad, 95.0)), 1e-8)
    cand_scale = max(float(np.percentile(cand_grad, 95.0)), 1e-8)
    src_norm = (src_grad / src_scale).astype(np.float32, copy=False)
    cand_norm = (cand_grad / cand_scale).astype(np.float32, copy=False)

    grad_corr = float(pearson_corrcoef(src_norm, cand_norm))
    edge_thr_src = max(float(np.percentile(src_norm, 90.0)), 0.05)
    edge_thr_cand = max(float(np.percentile(cand_norm, 90.0)), 0.05)
    src_edges = src_norm >= edge_thr_src
    cand_edges = cand_norm >= edge_thr_cand
    inter = int(np.count_nonzero(src_edges & cand_edges))
    union = int(np.count_nonzero(src_edges | cand_edges))
    edge_iou = 1.0 if union == 0 else float(inter / union)

    trigger = bool((grad_corr < float(min_grad_corr)) and (edge_iou < float(min_edge_iou)))
    return {
        "openai_repair_structure_guard_enabled": True,
        "openai_repair_structure_guard_triggered": bool(trigger),
        "openai_repair_structure_grad_corr": float(grad_corr),
        "openai_repair_structure_edge_iou": float(edge_iou),
        "openai_repair_structure_min_grad_corr": float(min_grad_corr),
        "openai_repair_structure_min_edge_iou": float(min_edge_iou),
    }


def downsample_rgb_max_side(image_rgb_f32: np.ndarray, *, max_side: int) -> np.ndarray:
    h, w = image_rgb_f32.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return image_rgb_f32.astype(np.float32, copy=False)
    scale = float(max_side / max(longest, 1))
    target_w = max(1, int(round(w * scale)))
    target_h = max(1, int(round(h * scale)))
    return resize_rgb_f32(image_rgb_f32, target_w=target_w, target_h=target_h)


def pearson_corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    af = a.reshape(-1).astype(np.float64, copy=False)
    bf = b.reshape(-1).astype(np.float64, copy=False)
    if af.size == 0 or bf.size == 0:
        return 1.0
    am = float(np.mean(af))
    bm = float(np.mean(bf))
    ax = af - am
    bx = bf - bm
    denom = float(np.sqrt(np.sum(ax * ax) * np.sum(bx * bx)))
    if denom <= 1e-12:
        return 1.0
    corr = float(np.sum(ax * bx) / denom)
    return float(np.clip(corr, -1.0, 1.0))


def generate_auto_repair_mask(*, source: np.ndarray, aggressiveness: str) -> tuple[np.ndarray, dict[str, Any]]:
    profile = AUTO_MASK_PROFILES.get(aggressiveness, AUTO_MASK_PROFILES["balanced"])
    linear = srgb_to_linear(np.clip(source, 0.0, 1.0).astype(np.float32, copy=False))
    y = rec709_luminance(linear)

    lowfreq = np.abs(y - gaussian_blur(y, sigma=float(profile["low_sigma"])))
    highfreq = np.abs(y - gaussian_blur(y, sigma=float(profile["detail_sigma"])))

    rg = linear[:, :, 0] - linear[:, :, 1]
    bg = linear[:, :, 2] - linear[:, :, 1]
    rg_low = gaussian_blur(rg, sigma=float(profile["low_sigma"]))
    bg_low = gaussian_blur(bg, sigma=float(profile["low_sigma"]))
    chroma = np.sqrt(np.square(rg - rg_low) + np.square(bg - bg_low)).astype(np.float32, copy=False)

    low_thr = max(float(np.percentile(lowfreq, float(profile["low_q"]))), float(profile["low_abs_min"]))
    high_thr = max(float(np.percentile(highfreq, float(profile["high_q"]))), float(profile["high_abs_min"]))
    chroma_thr = max(float(np.percentile(chroma, float(profile["chroma_q"]))), float(profile["chroma_abs_min"]))

    grad = gradient_magnitude(y)
    grad_thr = max(float(np.percentile(grad, float(profile["edge_q"]))), float(profile["edge_abs_min"]))
    edge_map = smoothstep(grad_thr * 0.70, grad_thr * 1.35, grad)

    candidate = (
        (lowfreq >= low_thr)
        | (chroma >= chroma_thr)
        | ((highfreq >= high_thr) & ((lowfreq >= (0.65 * low_thr)) | (chroma >= (0.55 * chroma_thr))))
    )
    candidate &= edge_map <= float(profile["edge_keep_max"])

    cleaned = gaussian_blur(candidate.astype(np.float32), sigma=float(profile["clean_sigma"]))
    cleaned_mask = cleaned >= 0.50
    expanded = gaussian_blur(cleaned_mask.astype(np.float32), sigma=float(profile["expand_sigma"]))
    final_mask = expanded >= float(profile["expand_threshold"])
    final_mask = (gaussian_blur(final_mask.astype(np.float32), sigma=0.7) >= 0.42).astype(bool, copy=False)

    mask_pixels = int(np.count_nonzero(final_mask))
    mask_fraction = float(mask_pixels / max(final_mask.size, 1))

    meta = {
        "openai_repair_mask_generator": "auto-residual",
        "openai_repair_mask_low_threshold": float(low_thr),
        "openai_repair_mask_high_threshold": float(high_thr),
        "openai_repair_mask_chroma_threshold": float(chroma_thr),
        "openai_repair_mask_edge_threshold": float(grad_thr),
        "openai_repair_mask_low_sigma": float(profile["low_sigma"]),
        "openai_repair_mask_detail_sigma": float(profile["detail_sigma"]),
        "openai_repair_mask_clean_sigma": float(profile["clean_sigma"]),
        "openai_repair_mask_expand_sigma": float(profile["expand_sigma"]),
        "openai_repair_mask_expand_threshold": float(profile["expand_threshold"]),
        "openai_repair_mask_pixels": int(mask_pixels),
        "openai_repair_mask_fraction": float(mask_fraction),
    }
    return final_mask, meta


AUTO_MASK_PROFILES: dict[str, dict[str, float]] = {
    "conservative": {
        "low_sigma": 22.0,
        "detail_sigma": 2.2,
        "low_q": 99.6,
        "high_q": 99.7,
        "chroma_q": 99.7,
        "low_abs_min": 0.020,
        "high_abs_min": 0.018,
        "chroma_abs_min": 0.018,
        "edge_q": 90.0,
        "edge_abs_min": 0.014,
        "edge_keep_max": 0.30,
        "clean_sigma": 1.0,
        "expand_sigma": 0.8,
        "expand_threshold": 0.32,
    },
    "balanced": {
        "low_sigma": 16.0,
        "detail_sigma": 1.7,
        "low_q": 99.2,
        "high_q": 99.3,
        "chroma_q": 99.3,
        "low_abs_min": 0.014,
        "high_abs_min": 0.012,
        "chroma_abs_min": 0.012,
        "edge_q": 92.0,
        "edge_abs_min": 0.012,
        "edge_keep_max": 0.42,
        "clean_sigma": 1.1,
        "expand_sigma": 1.0,
        "expand_threshold": 0.26,
    },
    "aggressive": {
        "low_sigma": 10.0,
        "detail_sigma": 1.3,
        "low_q": 98.8,
        "high_q": 98.9,
        "chroma_q": 98.9,
        "low_abs_min": 0.010,
        "high_abs_min": 0.009,
        "chroma_abs_min": 0.009,
        "edge_q": 94.0,
        "edge_abs_min": 0.010,
        "edge_keep_max": 0.56,
        "clean_sigma": 1.2,
        "expand_sigma": 1.3,
        "expand_threshold": 0.20,
    },
}


def gradient_magnitude(y: np.ndarray) -> np.ndarray:
    gx = np.zeros_like(y, dtype=np.float32)
    gy = np.zeros_like(y, dtype=np.float32)
    gx[:, 1:-1] = 0.5 * (y[:, 2:] - y[:, :-2])
    gy[1:-1, :] = 0.5 * (y[2:, :] - y[:-2, :])
    return np.sqrt((gx * gx) + (gy * gy)).astype(np.float32, copy=False)


def smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    if edge1 <= edge0:
        return np.zeros_like(x, dtype=np.float32)
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return (t * t * (3.0 - (2.0 * t))).astype(np.float32, copy=False)


def gaussian_blur(image: np.ndarray, *, sigma: float) -> np.ndarray:
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
        for i, weight in enumerate(kernel):
            out += float(weight) * padded[i : i + image.shape[0], :]
        return out

    padded = np.pad(image, ((0, 0), (radius, radius)), mode="reflect")
    out = np.zeros_like(image, dtype=np.float32)
    for i, weight in enumerate(kernel):
        out += float(weight) * padded[:, i : i + image.shape[1]]
    return out
