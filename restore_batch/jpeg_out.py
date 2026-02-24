from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .tiff16 import encode_rgb_u16_tiff_bytes


@dataclass(frozen=True)
class JpegExportConfig:
    max_mb: float = 3.0
    quality_max: int = 92
    quality_min: int = 62
    quality_step: int = 4
    downscale_step: float = 0.90
    min_side: int = 320


class JpegExportError(RuntimeError):
    def __init__(self, message: str, *, metadata: dict[str, Any]) -> None:
        super().__init__(message)
        self.metadata = metadata


def export_jpg_with_cap(*, image_srgb_f32: np.ndarray, output_path: Path, config: JpegExportConfig) -> dict[str, Any]:
    jpeg_bytes, metadata = encode_jpeg_to_target(image_srgb_f32=image_srgb_f32, config=config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        handle.write(jpeg_bytes)
    return metadata


def encode_jpeg_to_target(*, image_srgb_f32: np.ndarray, config: JpegExportConfig) -> tuple[bytes, dict[str, Any]]:
    if image_srgb_f32.ndim != 3 or image_srgb_f32.shape[2] != 3:
        raise ValueError("Expected float image with shape (height, width, 3).")

    safe = normalize_export_config(config)
    target_max_bytes = int(round(safe.max_mb * 1024.0 * 1024.0))

    image_u16, conversion_meta = float_to_u16_with_clip_metadata(image_srgb_f32)
    tiff_bytes = encode_rgb_u16_tiff_bytes(image_u16)
    base_rgb = pil_rgb_from_tiff_bytes(tiff_bytes)
    original_width, original_height = base_rgb.size

    current = base_rgb
    downscale_iterations = 0
    quality_candidates = quality_schedule(safe.quality_max, safe.quality_min, safe.quality_step)

    while True:
        for quality in quality_candidates:
            jpeg_bytes = encode_jpeg_bytes(current, quality=quality)
            size_bytes = len(jpeg_bytes)
            if size_bytes <= target_max_bytes:
                width, height = current.size
                metadata: dict[str, Any] = {}
                metadata.update(conversion_meta)
                metadata.update(
                    {
                        "output_bit_depth": 8,
                        "output_channels": 3,
                        "output_color_space": "sRGB",
                        "output_format": "JPEG",
                        "output_file_size_bytes": int(size_bytes),
                        "output_file_size_mb": float(size_bytes / (1024.0 * 1024.0)),
                        "jpg_target_max_bytes": int(target_max_bytes),
                        "jpg_target_met": True,
                        "jpg_quality_final": int(quality),
                        "jpg_subsampling": "4:2:0",
                        "jpg_optimize": True,
                        "jpg_progressive": False,
                        "jpg_downscale_iterations": int(downscale_iterations),
                        "jpg_scale_factor_total": float(width / max(original_width, 1)),
                        "jpg_export_dims": [int(width), int(height)],
                        "export_intermediate_format": "TIFF",
                        "export_intermediate_tiff_in_memory": True,
                    }
                )
                return jpeg_bytes, metadata

        current_width, current_height = current.size
        next_width = max(1, int(np.floor(current_width * safe.downscale_step)))
        next_height = max(1, int(np.floor(current_height * safe.downscale_step)))
        if next_width >= current_width and current_width > 1:
            next_width = current_width - 1
        if next_height >= current_height and current_height > 1:
            next_height = current_height - 1

        if min(next_width, next_height) < safe.min_side:
            failure_meta: dict[str, Any] = {}
            failure_meta.update(conversion_meta)
            failure_meta.update(
                {
                    "output_bit_depth": 8,
                    "output_channels": 3,
                    "output_color_space": "sRGB",
                    "output_format": "JPEG",
                    "jpg_target_max_bytes": int(target_max_bytes),
                    "jpg_target_met": False,
                    "jpg_subsampling": "4:2:0",
                    "jpg_optimize": True,
                    "jpg_progressive": False,
                    "jpg_downscale_iterations": int(downscale_iterations),
                    "jpg_scale_factor_total": float(current_width / max(original_width, 1)),
                    "jpg_export_dims": [int(current_width), int(current_height)],
                    "export_intermediate_format": "TIFF",
                    "export_intermediate_tiff_in_memory": True,
                    "export_failure_reason": "jpg-cap-unmet-at-min-side",
                }
            )
            raise JpegExportError("jpg-cap-unmet-at-min-side", metadata=failure_meta)

        current = current.resize((next_width, next_height), resample=Image.Resampling.BICUBIC)
        downscale_iterations += 1


def normalize_export_config(config: JpegExportConfig) -> JpegExportConfig:
    quality_max = int(np.clip(config.quality_max, 1, 100))
    quality_min = int(np.clip(config.quality_min, 1, quality_max))
    quality_step = int(max(1, config.quality_step))
    downscale_step = float(np.clip(config.downscale_step, 0.50, 0.99))
    min_side = int(max(8, config.min_side))
    max_mb = float(max(0.01, config.max_mb))
    return JpegExportConfig(
        max_mb=max_mb,
        quality_max=quality_max,
        quality_min=quality_min,
        quality_step=quality_step,
        downscale_step=downscale_step,
        min_side=min_side,
    )


def quality_schedule(quality_max: int, quality_min: int, quality_step: int) -> list[int]:
    quality_step = max(1, int(quality_step))
    quality_max = int(np.clip(quality_max, 1, 100))
    quality_min = int(np.clip(quality_min, 1, quality_max))
    values = list(range(quality_max, quality_min - 1, -quality_step))
    if not values:
        return [quality_min]
    if values[-1] != quality_min:
        values.append(quality_min)
    return values


def float_to_u16_with_clip_metadata(image_srgb_f32: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    clipped_low = int(np.count_nonzero(image_srgb_f32 < 0.0))
    clipped_high = int(np.count_nonzero(image_srgb_f32 > 1.0))
    clipped_total = clipped_low + clipped_high
    clipped = np.clip(image_srgb_f32, 0.0, 1.0)
    image_u16 = np.rint(clipped * 65535.0).astype(np.uint16)
    metadata = {
        "conversion_clipped_values": int(clipped_total),
        "conversion_clipped_low": int(clipped_low),
        "conversion_clipped_high": int(clipped_high),
    }
    return image_u16, metadata


def pil_rgb_from_tiff_bytes(tiff_bytes: bytes) -> Image.Image:
    with Image.open(BytesIO(tiff_bytes)) as handle:
        return handle.convert("RGB").copy()


def encode_jpeg_bytes(image_rgb: Image.Image, *, quality: int) -> bytes:
    output = BytesIO()
    image_rgb.save(
        output,
        format="JPEG",
        quality=int(np.clip(quality, 1, 100)),
        optimize=True,
        progressive=False,
        subsampling=2,
    )
    return output.getvalue()
