from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import ExifTags, Image, ImageCms, ImageOps

from .models import ImageContext

ORIENTATION_TAG = ExifTags.Base.Orientation


@dataclass(frozen=True)
class NormalizationConfig:
    alpha_background_rgb: tuple[int, int, int] = (255, 255, 255)
    strict_icc: bool = False


class NormalizationStage:
    name = "normalize-input"

    def __init__(self, config: NormalizationConfig) -> None:
        self.config = config
        self._srgb_profile = ImageCms.createProfile("sRGB")

    def process(self, context: ImageContext) -> ImageContext:
        with Image.open(context.input_path) as source:
            source.load()
            source_array = np.asarray(source)
            metadata: dict[str, Any] = {
                "source_format": source.format,
                "source_mode": source.mode,
                "source_size": [source.width, source.height],
                "source_bit_depth": infer_bit_depth(source.mode),
                "source_has_icc_profile": bool(source.info.get("icc_profile")),
                "source_has_alpha": has_alpha(source),
                "source_numpy_dtype": str(source_array.dtype),
            }

            oriented, orientation_before, orientation_applied = apply_exif_orientation(source)
            metadata["exif_orientation_before"] = orientation_before
            metadata["exif_orientation_applied"] = orientation_applied

            image_f32, color_meta = normalize_to_working_float(
                oriented,
                srgb_profile=self._srgb_profile,
                alpha_background_rgb=self.config.alpha_background_rgb,
                strict_icc=self.config.strict_icc,
            )
            metadata.update(color_meta)
            metadata["strict_icc_mode"] = bool(self.config.strict_icc)

            # Stage 1 output is gamma-encoded sRGB float32 in [0, 1].
            # All restoration stages that compute statistics explicitly convert this
            # working buffer to linear-light RGB first, then convert back as needed.
            metadata["working_color_space"] = "sRGB"
            metadata["working_transfer_function"] = "sRGB-IEC61966-2-1"
            metadata["working_statistical_domain"] = "linear-sRGB"
            metadata["working_linearization_required"] = True
            metadata["working_linearization_boundary_in"] = "stage-input"
            metadata["working_linearization_boundary_out"] = "stage-output"
            metadata["working_dtype"] = "float32"
            metadata["working_range"] = [0.0, 1.0]
            metadata["normalized_size"] = [int(image_f32.shape[1]), int(image_f32.shape[0])]

            context.image_f32 = image_f32
            context.metadata.update(metadata)
            return context


def infer_bit_depth(mode: str) -> int:
    if mode in {"1", "L", "P", "RGB", "RGBA", "CMYK", "YCbCr", "LAB", "HSV", "LA", "PA"}:
        return 8
    if mode in {"I;16", "I;16L", "I;16B", "I;16N"}:
        return 16
    if mode in {"I", "F"}:
        return 32
    return 8


def has_alpha(image: Image.Image) -> bool:
    bands = image.getbands()
    return ("A" in bands) or ("transparency" in image.info)


def apply_exif_orientation(image: Image.Image) -> tuple[Image.Image, int | None, bool]:
    exif = image.getexif()
    orientation_before = exif.get(ORIENTATION_TAG)
    oriented = ImageOps.exif_transpose(image)
    orientation_applied = orientation_before not in (None, 1)
    return oriented, orientation_before, orientation_applied


def convert_to_srgb(
    image: Image.Image,
    *,
    srgb_profile: ImageCms.ImageCmsProfile,
    alpha_background_rgb: tuple[int, int, int],
    strict_icc: bool,
) -> tuple[Image.Image, dict[str, Any]]:
    icc_profile = image.info.get("icc_profile")
    alpha_present = has_alpha(image)
    output_mode = "RGBA" if alpha_present else "RGB"

    converted, icc_meta = convert_profile_to_srgb(
        image=image,
        output_mode=output_mode,
        icc_profile=icc_profile,
        srgb_profile=srgb_profile,
        strict_icc=strict_icc,
    )

    metadata: dict[str, Any] = {
        "color_space_target": "sRGB",
        **icc_meta,
        "alpha_removed": False,
    }

    if alpha_present:
        rgba = converted if converted.mode == "RGBA" else converted.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, alpha_background_rgb + (255,))
        converted = Image.alpha_composite(bg, rgba).convert("RGB")
        metadata["alpha_removed"] = True
        metadata["alpha_background_rgb"] = list(alpha_background_rgb)

    if converted.mode != "RGB":
        converted = converted.convert("RGB")

    return converted, metadata


def normalize_to_working_float(
    image: Image.Image,
    *,
    srgb_profile: ImageCms.ImageCmsProfile,
    alpha_background_rgb: tuple[int, int, int],
    strict_icc: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    icc_profile = image.info.get("icc_profile")
    alpha_present = has_alpha(image)
    native = np.asarray(image)

    if should_preserve_high_depth_grayscale(native=native, has_alpha=alpha_present, has_icc_profile=bool(icc_profile)):
        gray_f32 = native.astype(np.float32) / 65535.0
        working = np.stack((gray_f32, gray_f32, gray_f32), axis=-1)
        metadata: dict[str, Any] = {
            "color_space_target": "sRGB",
            "source_color_profile": None,
            "color_transform": "grayscale-u16-replicate-to-rgb",
            "icc_transform_applied": False,
            "alpha_removed": False,
            "high_bit_depth_preserved": True,
        }
        return np.clip(working, 0.0, 1.0), metadata

    converted_rgb, metadata = convert_to_srgb(
        image,
        srgb_profile=srgb_profile,
        alpha_background_rgb=alpha_background_rgb,
        strict_icc=strict_icc,
    )
    metadata["high_bit_depth_preserved"] = False
    return pil_to_working_float(converted_rgb), metadata


def convert_profile_to_srgb(
    *,
    image: Image.Image,
    output_mode: str,
    icc_profile: bytes | None,
    srgb_profile: ImageCms.ImageCmsProfile,
    strict_icc: bool,
) -> tuple[Image.Image, dict[str, Any]]:
    metadata: dict[str, Any] = {
        "source_color_profile": None,
        "color_transform": "assumed-srgb",
        "icc_transform_applied": False,
        "icc_fallback_used": False,
        "icc_fallback_assumed_srgb": False,
    }

    if not icc_profile:
        return image.convert(output_mode), metadata

    try:
        src_profile = ImageCms.ImageCmsProfile(io.BytesIO(icc_profile))
        profile_name = ImageCms.getProfileName(src_profile).strip().replace("\x00", "")
        converted = ImageCms.profileToProfile(
            image,
            src_profile,
            srgb_profile,
            outputMode=output_mode,
        )
        metadata.update(
            {
                "source_color_profile": profile_name or "embedded-profile",
                "color_transform": "icc-to-srgb",
                "icc_transform_applied": True,
            }
        )
        return converted, metadata
    except Exception as exc:  # noqa: BLE001 - keep processing on ICC failures.
        if strict_icc:
            raise RuntimeError(f"Strict ICC mode: failed to convert embedded profile ({exc})") from exc
        metadata.update(
            {
                "source_color_profile": "embedded-profile-unreadable",
                "color_transform": "icc-fallback-convert",
                "icc_transform_warning": str(exc),
                "icc_transform_applied": False,
                "icc_fallback_used": True,
                "icc_fallback_assumed_srgb": True,
            }
        )
        return image.convert(output_mode), metadata


def should_preserve_high_depth_grayscale(*, native: np.ndarray, has_alpha: bool, has_icc_profile: bool) -> bool:
    if has_alpha or has_icc_profile:
        return False
    if native.ndim != 2:
        return False
    if native.dtype != np.uint16:
        return False
    return True


def pil_to_working_float(image: Image.Image) -> np.ndarray:
    array = np.asarray(image)

    if array.ndim == 2:
        array = np.stack((array, array, array), axis=-1)
    if array.ndim != 3:
        raise ValueError(f"Unexpected image shape after normalization: {array.shape}")
    if array.shape[2] > 3:
        array = array[:, :, :3]

    if np.issubdtype(array.dtype, np.floating):
        array_f32 = array.astype(np.float32, copy=False)
        if float(np.nanmax(array_f32)) > 1.0:
            array_f32 /= 255.0
        return np.clip(array_f32, 0.0, 1.0)

    if array.dtype == np.uint16:
        return array.astype(np.float32) / 65535.0

    return array.astype(np.float32) / 255.0
