from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .models import ImageContext


@dataclass(frozen=True)
class DustCleanupConfig:
    mode: str = "on"
    response_sigma: float = 1.15
    response_sigma_wide: float = 2.20
    mad_multiplier: float = 5.00
    min_contrast: float = 0.020
    texture_percentile: float = 62.0
    min_component_px: int = 3
    max_component_px: int = 220
    max_component_aspect: float = 3.50
    min_component_fill: float = 0.08
    max_mask_fraction: float = 0.015
    inpaint_radius: float = 2.20
    save_mask_preview: bool = False
    mask_preview_subdir: str = "_dust_masks"


class DustCleanupStage:
    name = "dust-cleanup"

    def __init__(self, config: DustCleanupConfig) -> None:
        self.config = config
        self._cv2: Any | None = None
        self._warning: str | None = None
        try:
            import cv2  # type: ignore

            self._cv2 = cv2
        except Exception as exc:  # noqa: BLE001 - optional dependency.
            self._warning = f"opencv-unavailable: {exc}"

    def process(self, context: ImageContext) -> ImageContext:
        if context.image_f32 is None:
            raise RuntimeError("Dust cleanup stage expected image data.")

        if self.config.mode == "off":
            context.metadata.update(dust_disabled_metadata(config=self.config, warning=self._warning))
            return context

        image_type = str(context.metadata.get("image_type", "true-color"))
        is_color = image_type == "true-color"
        output, metadata = apply_dust_cleanup(
            image_srgb=context.image_f32,
            config=self.config,
            is_color_image=is_color,
            output_path=context.output_path,
            cv2_module=self._cv2,
            warning=self._warning,
        )
        context.image_f32 = output
        context.metadata.update(metadata)
        return context


def dust_disabled_metadata(*, config: DustCleanupConfig, warning: str | None) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "dust_clean_requested": False,
        "dust_clean_enabled": False,
        "dust_clean_applied": False,
        "dust_clean_method": "global-hat-mad-inpaint",
        "dust_clean_backend": "disabled",
        "dust_clean_skipped_reason": "dust-clean-off",
        "dust_clean_mad_multiplier": float(max(0.0, config.mad_multiplier)),
        "dust_clean_min_contrast": float(max(0.0, config.min_contrast)),
        "dust_clean_texture_percentile": float(np.clip(config.texture_percentile, 1.0, 99.0)),
        "dust_clean_inpaint_radius": float(max(0.5, config.inpaint_radius)),
        "dust_clean_candidate_pixels": 0,
        "dust_clean_candidate_fraction": 0.0,
        "dust_clean_mask_pixels": 0,
        "dust_clean_mask_fraction": 0.0,
        "dust_clean_components_total": 0,
        "dust_clean_components_kept": 0,
        "dust_clean_component_reject_counts": {},
        "dust_clean_sharpness_proxy_before": 0.0,
        "dust_clean_sharpness_proxy_after": 0.0,
        "dust_clean_luma_delta_mean_abs": 0.0,
        "dust_clean_mask_preview_path": None,
    }
    if warning is not None:
        metadata["dust_clean_warning"] = warning
    return metadata


def apply_dust_cleanup(
    *,
    image_srgb: np.ndarray,
    config: DustCleanupConfig,
    is_color_image: bool,
    output_path: Path,
    cv2_module: Any | None,
    warning: str | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    source = np.clip(image_srgb, 0.0, 1.0).astype(np.float32, copy=False)
    y, cb, cr = rgb_to_ycbcr(source)
    sharp_before = variance_of_laplacian(y)

    response_sigma = float(max(0.45, config.response_sigma))
    response_sigma_wide = float(max(response_sigma, config.response_sigma_wide))
    mad_multiplier = float(max(0.0, config.mad_multiplier))
    min_contrast = float(max(0.0, config.min_contrast))
    texture_percentile = float(np.clip(config.texture_percentile, 1.0, 99.0))
    max_mask_fraction = float(np.clip(config.max_mask_fraction, 1e-6, 1.0))

    grad = gradient_magnitude(y)
    texture_threshold = float(np.percentile(grad, texture_percentile))
    smooth_mask = grad <= texture_threshold
    if int(np.count_nonzero(smooth_mask)) < 1024:
        smooth_mask = np.ones_like(smooth_mask, dtype=bool)

    blur_base = gaussian_blur(y, sigma=response_sigma)
    blur_wide = gaussian_blur(y, sigma=response_sigma_wide)

    dark_response = np.maximum(0.0, blur_base - y)
    bright_response = np.maximum(0.0, y - blur_base)
    if response_sigma_wide > response_sigma:
        dark_response = np.maximum(dark_response, 0.85 * np.maximum(0.0, blur_wide - y))
        bright_response = np.maximum(bright_response, 0.85 * np.maximum(0.0, y - blur_wide))

    dark_threshold, dark_median, dark_sigma = robust_threshold(
        dark_response[smooth_mask],
        mad_multiplier=mad_multiplier,
        min_threshold=min_contrast,
    )
    bright_threshold, bright_median, bright_sigma = robust_threshold(
        bright_response[smooth_mask],
        mad_multiplier=mad_multiplier,
        min_threshold=min_contrast,
    )

    candidate = smooth_mask & ((dark_response >= dark_threshold) | (bright_response >= bright_threshold))
    candidate_pixels = int(np.count_nonzero(candidate))
    candidate_fraction = float(candidate_pixels / max(candidate.size, 1))

    metadata: dict[str, Any] = {
        "dust_clean_requested": True,
        "dust_clean_enabled": True,
        "dust_clean_applied": False,
        "dust_clean_method": "global-hat-mad-inpaint",
        "dust_clean_backend": "opencv-telea" if cv2_module is not None else "gaussian-fallback",
        "dust_clean_mad_multiplier": mad_multiplier,
        "dust_clean_min_contrast": min_contrast,
        "dust_clean_texture_percentile": texture_percentile,
        "dust_clean_texture_threshold": texture_threshold,
        "dust_clean_response_sigma": response_sigma,
        "dust_clean_response_sigma_wide": response_sigma_wide,
        "dust_clean_dark_threshold": dark_threshold,
        "dust_clean_bright_threshold": bright_threshold,
        "dust_clean_dark_median": dark_median,
        "dust_clean_bright_median": bright_median,
        "dust_clean_dark_sigma": dark_sigma,
        "dust_clean_bright_sigma": bright_sigma,
        "dust_clean_candidate_pixels": candidate_pixels,
        "dust_clean_candidate_fraction": candidate_fraction,
        "dust_clean_inpaint_radius": float(max(0.5, config.inpaint_radius)),
        "dust_clean_max_mask_fraction": max_mask_fraction,
        "dust_clean_mask_preview_path": None,
    }
    if warning is not None:
        metadata["dust_clean_warning"] = warning

    if candidate_pixels == 0:
        metadata.update(
            {
                "dust_clean_skipped_reason": "no-candidates",
                "dust_clean_mask_pixels": 0,
                "dust_clean_mask_fraction": 0.0,
                "dust_clean_components_total": 0,
                "dust_clean_components_kept": 0,
                "dust_clean_component_reject_counts": {},
                "dust_clean_sharpness_proxy_before": float(sharp_before),
                "dust_clean_sharpness_proxy_after": float(sharp_before),
                "dust_clean_luma_delta_mean_abs": 0.0,
            }
        )
        return source, metadata
    if candidate_fraction > max_mask_fraction:
        metadata.update(
            {
                "dust_clean_skipped_reason": "candidate-mask-too-large-risk",
                "dust_clean_mask_pixels": int(candidate_pixels),
                "dust_clean_mask_fraction": float(candidate_fraction),
                "dust_clean_components_total": 0,
                "dust_clean_components_kept": 0,
                "dust_clean_component_reject_counts": {"mask_too_large": 1},
                "dust_clean_sharpness_proxy_before": float(sharp_before),
                "dust_clean_sharpness_proxy_after": float(sharp_before),
                "dust_clean_luma_delta_mean_abs": 0.0,
            }
        )
        return source, metadata

    dust_mask, component_stats = filter_small_components(
        candidate,
        min_area=max(1, int(config.min_component_px)),
        max_area=max(int(config.min_component_px), int(config.max_component_px)),
        max_aspect=float(max(1.0, config.max_component_aspect)),
        min_fill=float(np.clip(config.min_component_fill, 0.0, 1.0)),
        cv2_module=cv2_module,
    )
    mask_pixels = int(np.count_nonzero(dust_mask))
    mask_fraction = float(mask_pixels / max(dust_mask.size, 1))
    metadata.update(
        {
            "dust_clean_mask_pixels": mask_pixels,
            "dust_clean_mask_fraction": mask_fraction,
            "dust_clean_components_total": int(component_stats["components_total"]),
            "dust_clean_components_kept": int(component_stats["components_kept"]),
            "dust_clean_component_reject_counts": component_stats["reject_counts"],
        }
    )

    if mask_pixels == 0:
        metadata["dust_clean_skipped_reason"] = "no-components-kept"
        metadata["dust_clean_sharpness_proxy_before"] = float(sharp_before)
        metadata["dust_clean_sharpness_proxy_after"] = float(sharp_before)
        metadata["dust_clean_luma_delta_mean_abs"] = 0.0
        return source, metadata
    if mask_fraction > max_mask_fraction:
        metadata["dust_clean_skipped_reason"] = "mask-too-large-risk"
        metadata["dust_clean_sharpness_proxy_before"] = float(sharp_before)
        metadata["dust_clean_sharpness_proxy_after"] = float(sharp_before)
        metadata["dust_clean_luma_delta_mean_abs"] = 0.0
        return source, metadata

    if config.save_mask_preview:
        preview_path = write_mask_preview(
            mask=dust_mask,
            output_path=output_path,
            subdir=config.mask_preview_subdir,
        )
        metadata["dust_clean_mask_preview_path"] = str(preview_path)

    inpaint_radius = float(max(0.5, config.inpaint_radius))
    if cv2_module is not None:
        y_clean = inpaint_luma_cv2(y=y, mask=dust_mask, radius=inpaint_radius, cv2_module=cv2_module)
    else:
        y_clean = inpaint_luma_fallback(y=y, mask=dust_mask, radius=inpaint_radius)

    if is_color_image:
        output = ycbcr_to_rgb(y_clean, cb, cr)
    else:
        output = np.repeat(y_clean[:, :, None], 3, axis=2)
    output = np.clip(output, 0.0, 1.0).astype(np.float32, copy=False)

    sharp_after = variance_of_laplacian(y_clean)
    luma_delta = float(np.mean(np.abs(y_clean - y)))
    metadata.update(
        {
            "dust_clean_applied": True,
            "dust_clean_skipped_reason": None,
            "dust_clean_sharpness_proxy_before": float(sharp_before),
            "dust_clean_sharpness_proxy_after": float(sharp_after),
            "dust_clean_luma_delta_mean_abs": luma_delta,
        }
    )
    return output, metadata


def robust_threshold(values: np.ndarray, *, mad_multiplier: float, min_threshold: float) -> tuple[float, float, float]:
    if values.size == 0:
        return float(min_threshold), 0.0, 0.0
    vals = values.astype(np.float32, copy=False).reshape(-1)
    median = float(np.median(vals))
    mad = float(np.median(np.abs(vals - median)))
    sigma = float(1.4826 * mad)
    threshold = float(max(min_threshold, median + (mad_multiplier * sigma)))
    return threshold, median, sigma


def filter_small_components(
    mask: np.ndarray,
    *,
    min_area: int,
    max_area: int,
    max_aspect: float,
    min_fill: float,
    cv2_module: Any | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    reject_counts: dict[str, int] = {
        "too_small": 0,
        "too_large": 0,
        "too_elongated": 0,
        "too_sparse": 0,
    }

    if cv2_module is not None:
        mask8 = (mask.astype(np.uint8, copy=False) * 255).astype(np.uint8, copy=False)
        labels_count, labels, stats, _ = cv2_module.connectedComponentsWithStats(mask8, connectivity=8)
        keep = np.zeros_like(mask, dtype=bool)
        components_total = int(max(0, labels_count - 1))
        components_kept = 0
        for label in range(1, labels_count):
            x = int(stats[label, cv2_module.CC_STAT_LEFT])
            y = int(stats[label, cv2_module.CC_STAT_TOP])
            w = int(stats[label, cv2_module.CC_STAT_WIDTH])
            h = int(stats[label, cv2_module.CC_STAT_HEIGHT])
            area = int(stats[label, cv2_module.CC_STAT_AREA])

            reason = component_reject_reason(
                area=area,
                width=w,
                height=h,
                min_area=min_area,
                max_area=max_area,
                max_aspect=max_aspect,
                min_fill=min_fill,
            )
            if reason is not None:
                reject_counts[reason] = int(reject_counts.get(reason, 0) + 1)
                continue
            keep[labels == label] = True
            components_kept += 1

        return keep, {
            "components_total": components_total,
            "components_kept": components_kept,
            "reject_counts": reject_counts,
        }

    return filter_small_components_fallback(
        mask=mask,
        min_area=min_area,
        max_area=max_area,
        max_aspect=max_aspect,
        min_fill=min_fill,
        reject_counts=reject_counts,
    )


def filter_small_components_fallback(
    *,
    mask: np.ndarray,
    min_area: int,
    max_area: int,
    max_aspect: float,
    min_fill: float,
    reject_counts: dict[str, int],
) -> tuple[np.ndarray, dict[str, Any]]:
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=bool)
    keep = np.zeros((h, w), dtype=bool)
    components_total = 0
    components_kept = 0

    ys, xs = np.nonzero(mask)
    for idx in range(len(ys)):
        sy = int(ys[idx])
        sx = int(xs[idx])
        if visited[sy, sx]:
            continue
        components_total += 1
        stack = [(sy, sx)]
        coords: list[tuple[int, int]] = []
        min_y = sy
        max_y = sy
        min_x = sx
        max_x = sx

        while stack:
            cy, cx = stack.pop()
            if cy < 0 or cy >= h or cx < 0 or cx >= w:
                continue
            if visited[cy, cx] or not bool(mask[cy, cx]):
                continue
            visited[cy, cx] = True
            coords.append((cy, cx))
            min_y = min(min_y, cy)
            max_y = max(max_y, cy)
            min_x = min(min_x, cx)
            max_x = max(max_x, cx)

            stack.append((cy - 1, cx - 1))
            stack.append((cy - 1, cx))
            stack.append((cy - 1, cx + 1))
            stack.append((cy, cx - 1))
            stack.append((cy, cx + 1))
            stack.append((cy + 1, cx - 1))
            stack.append((cy + 1, cx))
            stack.append((cy + 1, cx + 1))

        area = len(coords)
        width = max_x - min_x + 1
        height = max_y - min_y + 1
        reason = component_reject_reason(
            area=area,
            width=width,
            height=height,
            min_area=min_area,
            max_area=max_area,
            max_aspect=max_aspect,
            min_fill=min_fill,
        )
        if reason is not None:
            reject_counts[reason] = int(reject_counts.get(reason, 0) + 1)
            continue

        for cy, cx in coords:
            keep[cy, cx] = True
        components_kept += 1

    return keep, {
        "components_total": int(components_total),
        "components_kept": int(components_kept),
        "reject_counts": reject_counts,
    }


def component_reject_reason(
    *,
    area: int,
    width: int,
    height: int,
    min_area: int,
    max_area: int,
    max_aspect: float,
    min_fill: float,
) -> str | None:
    if area < min_area:
        return "too_small"
    if area > max_area:
        return "too_large"
    aspect = float(max(width, height) / max(1, min(width, height)))
    if aspect > max_aspect:
        return "too_elongated"
    fill = float(area / max(1, width * height))
    if fill < min_fill:
        return "too_sparse"
    return None


def write_mask_preview(*, mask: np.ndarray, output_path: Path, subdir: str) -> Path:
    preview_dir = output_path.parent / subdir
    preview_dir.mkdir(parents=True, exist_ok=True)
    preview_path = preview_dir / f"{output_path.stem}_dustmask.png"
    mask_u8 = np.clip(mask.astype(np.float32, copy=False) * 255.0, 0.0, 255.0).astype(np.uint8)
    Image.fromarray(mask_u8, mode="L").save(preview_path)
    return preview_path


def inpaint_luma_cv2(*, y: np.ndarray, mask: np.ndarray, radius: float, cv2_module: Any) -> np.ndarray:
    y_u8 = np.clip((y * 255.0) + 0.5, 0.0, 255.0).astype(np.uint8)
    mask_u8 = (mask.astype(np.uint8, copy=False) * 255).astype(np.uint8, copy=False)
    y_inpaint_u8 = cv2_module.inpaint(y_u8, mask_u8, float(radius), cv2_module.INPAINT_TELEA)
    return np.clip(y_inpaint_u8.astype(np.float32) / 255.0, 0.0, 1.0)


def inpaint_luma_fallback(*, y: np.ndarray, mask: np.ndarray, radius: float) -> np.ndarray:
    blur = gaussian_blur(y, sigma=max(0.8, radius * 0.85))
    alpha = gaussian_blur(mask.astype(np.float32, copy=False), sigma=0.85)
    max_alpha = float(np.max(alpha))
    if max_alpha > 1e-8:
        alpha /= max_alpha
    return np.clip(y + (np.clip(alpha, 0.0, 1.0) * (blur - y)), 0.0, 1.0)


def variance_of_laplacian(y: np.ndarray) -> float:
    p = np.pad(y, ((1, 1), (1, 1)), mode="reflect")
    lap = (
        -4.0 * p[1:-1, 1:-1]
        + p[:-2, 1:-1]
        + p[2:, 1:-1]
        + p[1:-1, :-2]
        + p[1:-1, 2:]
    )
    return float(np.var(lap))


def gradient_magnitude(y: np.ndarray) -> np.ndarray:
    gx = np.zeros_like(y, dtype=np.float32)
    gy = np.zeros_like(y, dtype=np.float32)
    gx[:, 1:-1] = 0.5 * (y[:, 2:] - y[:, :-2])
    gy[1:-1, :] = 0.5 * (y[2:, :] - y[:-2, :])
    return np.sqrt((gx * gx) + (gy * gy)).astype(np.float32, copy=False)


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
        for i, w in enumerate(kernel):
            out += float(w) * padded[i : i + image.shape[0], :]
        return out

    padded = np.pad(image, ((0, 0), (radius, radius)), mode="reflect")
    out = np.zeros_like(image, dtype=np.float32)
    for i, w in enumerate(kernel):
        out += float(w) * padded[:, i : i + image.shape[1]]
    return out


def rgb_to_ycbcr(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]
    y = (0.2990 * r) + (0.5870 * g) + (0.1140 * b)
    cb = ((b - y) * 0.5640) + 0.5
    cr = ((r - y) * 0.7130) + 0.5
    return y.astype(np.float32, copy=False), cb.astype(np.float32, copy=False), cr.astype(np.float32, copy=False)


def ycbcr_to_rgb(y: np.ndarray, cb: np.ndarray, cr: np.ndarray) -> np.ndarray:
    r = y + (1.4030 * (cr - 0.5))
    g = y - (0.3440 * (cb - 0.5)) - (0.7140 * (cr - 0.5))
    b = y + (1.7730 * (cb - 0.5))
    return np.stack((r, g, b), axis=-1).astype(np.float32, copy=False)
