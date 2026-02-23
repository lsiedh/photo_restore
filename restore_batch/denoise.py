from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .models import ImageContext


@dataclass(frozen=True)
class DenoiseConfig:
    method: str = "edge-aware-luma"
    strength: float = 0.22
    chroma_strength: float = 0.08
    auto_strength: bool = False
    noise_low: float = 0.0015
    noise_high: float = 0.0100
    edge_protect_percentile: float = 70.0
    skin_protection: float = 0.45
    min_sharpness_ratio: float = 0.72
    max_sharpness_guard_iters: int = 3


class DenoiseStage:
    name = "denoise"

    def __init__(self, config: DenoiseConfig) -> None:
        self.config = config

    def process(self, context: ImageContext) -> ImageContext:
        if context.image_f32 is None:
            raise RuntimeError("Denoise stage expected image data.")

        image_type = str(context.metadata.get("image_type", "true-color"))
        is_color = image_type == "true-color"
        if self.config.method == "none":
            metadata = denoise_baseline_metadata(context.image_f32, method="none", auto_strength=self.config.auto_strength)
            metadata["denoise_applied"] = False
            metadata["denoise_skipped_reason"] = "method-none"
            context.metadata.update(metadata)
            return context

        output, metadata = apply_edge_aware_luma_denoise(
            image_srgb=context.image_f32,
            is_color_image=is_color,
            config=self.config,
        )
        context.image_f32 = output
        context.metadata.update(metadata)
        return context


def apply_edge_aware_luma_denoise(
    *,
    image_srgb: np.ndarray,
    is_color_image: bool,
    config: DenoiseConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    source = np.clip(image_srgb, 0.0, 1.0).astype(np.float32, copy=False)
    y, cb, cr = rgb_to_ycbcr(source)

    noise_before = noise_proxy(y)
    sharp_before = variance_of_laplacian(y)

    base_strength = float(np.clip(config.strength, 0.0, 1.0))
    chosen_strength = choose_strength(
        base_strength=base_strength,
        noise_value=noise_before,
        auto_strength=config.auto_strength,
        noise_low=float(max(1e-8, config.noise_low)),
        noise_high=float(max(config.noise_low + 1e-8, config.noise_high)),
    )
    chosen_chroma_strength = float(np.clip(config.chroma_strength, 0.0, 1.0)) * (0.4 + (0.6 * chosen_strength))

    denoised_y, denoised_cb, denoised_cr, guard_iters = denoise_with_sharpness_guard(
        y=y,
        cb=cb,
        cr=cr,
        is_color_image=is_color_image,
        strength=chosen_strength,
        chroma_strength=chosen_chroma_strength,
        edge_protect_percentile=config.edge_protect_percentile,
        skin_protection=config.skin_protection,
        min_sharpness_ratio=config.min_sharpness_ratio,
        max_iters=config.max_sharpness_guard_iters,
        sharpness_before=sharp_before,
    )

    if is_color_image:
        # Keep global chroma mean stable to avoid introducing cast.
        denoised_cb += float(np.mean(cb) - np.mean(denoised_cb))
        denoised_cr += float(np.mean(cr) - np.mean(denoised_cr))
        out = ycbcr_to_rgb(denoised_y, denoised_cb, denoised_cr)
    else:
        # Preserve neutrality for near-grayscale workflow.
        out = np.repeat(denoised_y[:, :, None], 3, axis=2)

    out = np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)
    y_after, _, _ = rgb_to_ycbcr(out)
    noise_after = noise_proxy(y_after)
    sharp_after = variance_of_laplacian(y_after)

    metadata: dict[str, Any] = {
        "denoise_applied": True,
        "denoise_method": config.method,
        "denoise_workflow": "color" if is_color_image else "near-grayscale",
        "denoise_strength_base": base_strength,
        "denoise_strength_chosen": float(chosen_strength),
        "denoise_chroma_strength_chosen": float(chosen_chroma_strength if is_color_image else 0.0),
        "denoise_auto_strength_enabled": bool(config.auto_strength),
        "denoise_auto_strength_adjusted": bool(config.auto_strength and abs(chosen_strength - base_strength) > 1e-9),
        "denoise_noise_proxy_before": float(noise_before),
        "denoise_noise_proxy_after": float(noise_after),
        "denoise_sharpness_proxy_before": float(sharp_before),
        "denoise_sharpness_proxy_after": float(sharp_after),
        "denoise_sharpness_guard_iterations": int(guard_iters),
    }
    return out, metadata


def denoise_baseline_metadata(image_srgb: np.ndarray, *, method: str, auto_strength: bool) -> dict[str, Any]:
    y, _, _ = rgb_to_ycbcr(np.clip(image_srgb, 0.0, 1.0).astype(np.float32, copy=False))
    noise = noise_proxy(y)
    sharp = variance_of_laplacian(y)
    return {
        "denoise_method": method,
        "denoise_auto_strength_enabled": bool(auto_strength),
        "denoise_auto_strength_adjusted": False,
        "denoise_strength_base": 0.0,
        "denoise_strength_chosen": 0.0,
        "denoise_chroma_strength_chosen": 0.0,
        "denoise_noise_proxy_before": float(noise),
        "denoise_noise_proxy_after": float(noise),
        "denoise_sharpness_proxy_before": float(sharp),
        "denoise_sharpness_proxy_after": float(sharp),
        "denoise_sharpness_guard_iterations": 0,
    }


def choose_strength(
    *,
    base_strength: float,
    noise_value: float,
    auto_strength: bool,
    noise_low: float,
    noise_high: float,
) -> float:
    if not auto_strength:
        return base_strength
    t = float(np.clip((noise_value - noise_low) / max(noise_high - noise_low, 1e-8), 0.0, 1.0))
    # Reduce strength strongly for already clean images.
    return float(np.clip(base_strength * (0.20 + (0.80 * t)), 0.0, 1.0))


def denoise_with_sharpness_guard(
    *,
    y: np.ndarray,
    cb: np.ndarray,
    cr: np.ndarray,
    is_color_image: bool,
    strength: float,
    chroma_strength: float,
    edge_protect_percentile: float,
    skin_protection: float,
    min_sharpness_ratio: float,
    max_iters: int,
    sharpness_before: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    current_strength = float(np.clip(strength, 0.0, 1.0))
    current_chroma_strength = float(np.clip(chroma_strength, 0.0, 1.0))
    guard_iters = 0
    best_y = y
    best_cb = cb
    best_cr = cr

    while True:
        y_d, cb_d, cr_d = denoise_once(
            y=y,
            cb=cb,
            cr=cr,
            is_color_image=is_color_image,
            strength=current_strength,
            chroma_strength=current_chroma_strength,
            edge_protect_percentile=edge_protect_percentile,
            skin_protection=skin_protection,
        )
        sharp_after = variance_of_laplacian(y_d)
        best_y, best_cb, best_cr = y_d, cb_d, cr_d
        ratio = sharp_after / max(sharpness_before, 1e-8)
        if ratio >= float(np.clip(min_sharpness_ratio, 0.0, 1.0)):
            break
        if guard_iters >= max(0, int(max_iters)):
            break
        current_strength *= 0.68
        current_chroma_strength *= 0.68
        guard_iters += 1

    return best_y, best_cb, best_cr, guard_iters


def denoise_once(
    *,
    y: np.ndarray,
    cb: np.ndarray,
    cr: np.ndarray,
    is_color_image: bool,
    strength: float,
    chroma_strength: float,
    edge_protect_percentile: float,
    skin_protection: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_blur = gaussian_blur(y, sigma=(0.55 + (1.75 * strength)))
    grad = gradient_magnitude(y)
    edge_threshold = float(np.percentile(grad, np.clip(edge_protect_percentile, 50.0, 95.0)))
    edge_alpha = smoothstep(edge_threshold * 0.50, edge_threshold * 1.50, grad)
    local_alpha = np.clip(strength * (1.0 - edge_alpha), 0.0, 1.0)

    if is_color_image:
        skin_mask = estimate_skin_mask_from_ycbcr(y, cb, cr).astype(np.float32)
        local_alpha *= (1.0 - (np.clip(skin_protection, 0.0, 1.0) * skin_mask))

    y_d = y + (local_alpha * (y_blur - y))

    if not is_color_image:
        return np.clip(y_d, 0.0, 1.0), cb, cr

    cb_blur = gaussian_blur(cb, sigma=(0.45 + (1.20 * chroma_strength)))
    cr_blur = gaussian_blur(cr, sigma=(0.45 + (1.20 * chroma_strength)))
    chroma_alpha = np.clip(chroma_strength * (1.0 - (0.85 * edge_alpha)), 0.0, 1.0)
    cb_d = cb + (chroma_alpha * (cb_blur - cb))
    cr_d = cr + (chroma_alpha * (cr_blur - cr))

    return np.clip(y_d, 0.0, 1.0), np.clip(cb_d, 0.0, 1.0), np.clip(cr_d, 0.0, 1.0)


def noise_proxy(y: np.ndarray) -> float:
    smooth = gaussian_blur(y, sigma=1.10)
    high = y - smooth
    grad = gradient_magnitude(y)
    threshold = float(np.percentile(grad, 35.0))
    mask = grad <= threshold
    if int(np.count_nonzero(mask)) < 256:
        mask = np.ones_like(mask, dtype=bool)
    vals = high[mask]
    median = float(np.median(vals))
    mad = float(np.median(np.abs(vals - median)))
    return float(1.4826 * mad)


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


def smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    if edge1 <= edge0:
        return np.zeros_like(x, dtype=np.float32)
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return (t * t * (3.0 - (2.0 * t))).astype(np.float32, copy=False)


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


def estimate_skin_mask_from_ycbcr(y: np.ndarray, cb: np.ndarray, cr: np.ndarray) -> np.ndarray:
    return (
        (y > 0.12)
        & (cb >= 0.26)
        & (cb <= 0.44)
        & (cr >= 0.52)
        & (cr <= 0.68)
    )
