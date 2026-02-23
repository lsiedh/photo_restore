from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .models import ImageContext

LOGGER = logging.getLogger("restore_batch")


@dataclass(frozen=True)
class SharpenConfig:
    method: str = "edge-aware-unsharp"
    amount: float = 0.30
    radius: float = 1.20
    threshold: float = 0.02
    edge_percentile_low: float = 55.0
    edge_percentile_high: float = 95.0
    edge_feather_sigma: float = 1.0
    strong_edge_percentile: float = 99.0
    strong_edge_suppression: float = 0.55
    shadow_protection: float = 0.35
    highlight_protection: float = 0.45
    shadow_luma_end: float = 0.08
    highlight_luma_start: float = 0.92
    skin_protection: float = 0.30
    detail_clip_base: float = 0.05
    max_clip_fraction: float = 0.001
    max_guardrail_iterations: int = 3
    face_sharpen_enabled: bool = False
    face_boost: float = 1.15
    face_min_size_ratio: float = 0.06
    face_edge_priority: float = 0.65


class SharpenStage:
    name = "sharpen"

    def __init__(self, config: SharpenConfig) -> None:
        self.config = config
        self._face_detector = OptionalFaceDetector(enabled=config.face_sharpen_enabled)
        if config.face_sharpen_enabled and self._face_detector.warning is not None:
            LOGGER.warning("Face sharpening requested but detector unavailable: %s", self._face_detector.warning)

    def process(self, context: ImageContext) -> ImageContext:
        if context.image_f32 is None:
            raise RuntimeError("Sharpen stage expected image data.")

        image_type = str(context.metadata.get("image_type", "true-color"))
        is_color = image_type == "true-color"

        if self.config.method == "none":
            metadata = sharpen_baseline_metadata(
                context.image_f32,
                method="none",
                is_color_image=is_color,
                face_enabled=self.config.face_sharpen_enabled,
                face_warning=self._face_detector.warning,
                face_detector=self._face_detector.detector_name,
            )
            metadata["sharpen_applied"] = False
            metadata["sharpen_skipped_reason"] = "method-none"
            context.metadata.update(metadata)
            return context

        output, metadata = apply_edge_aware_sharpen(
            image_srgb=context.image_f32,
            is_color_image=is_color,
            config=self.config,
            face_detector=self._face_detector,
        )
        context.image_f32 = output
        context.metadata.update(metadata)
        return context


@dataclass
class OptionalFaceDetector:
    enabled: bool
    warning: str | None = None
    detector_name: str = "disabled"
    _cv2: Any | None = None
    _cascade: Any | None = None

    def __post_init__(self) -> None:
        if not self.enabled:
            return

        try:
            import cv2  # type: ignore
        except Exception as exc:  # noqa: BLE001 - optional dependency.
            self.warning = f"opencv-unavailable: {exc}"
            self.detector_name = "opencv-haar"
            return

        candidates = [
            Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml",
            Path(cv2.data.haarcascades) / "haarcascade_frontalface_alt2.xml",
        ]
        cascade_path = next((path for path in candidates if path.exists()), None)
        if cascade_path is None:
            self.warning = "opencv-haarcascade-not-found"
            self.detector_name = "opencv-haar"
            return

        classifier = cv2.CascadeClassifier(str(cascade_path))
        if classifier.empty():
            self.warning = f"opencv-haarcascade-load-failed: {cascade_path}"
            self.detector_name = "opencv-haar"
            return

        self._cv2 = cv2
        self._cascade = classifier
        self.detector_name = "opencv-haar"

    def detect(self, y: np.ndarray, *, min_size_ratio: float) -> np.ndarray:
        if not self.enabled or self._cv2 is None or self._cascade is None:
            return np.zeros((0, 4), dtype=np.int32)

        h, w = y.shape
        min_side = max(16, int(round(min(h, w) * float(np.clip(min_size_ratio, 0.01, 0.25)))))
        gray8 = np.clip(y * 255.0, 0.0, 255.0).astype(np.uint8)
        faces = self._cascade.detectMultiScale(
            gray8,
            scaleFactor=1.10,
            minNeighbors=5,
            minSize=(min_side, min_side),
        )
        if faces is None or len(faces) == 0:
            return np.zeros((0, 4), dtype=np.int32)
        return np.asarray(faces, dtype=np.int32)


def apply_edge_aware_sharpen(
    *,
    image_srgb: np.ndarray,
    is_color_image: bool,
    config: SharpenConfig,
    face_detector: OptionalFaceDetector,
) -> tuple[np.ndarray, dict[str, Any]]:
    source = np.clip(image_srgb, 0.0, 1.0).astype(np.float32, copy=False)
    y, cb, cr = rgb_to_ycbcr(source)

    sharp_before = variance_of_laplacian(y)
    amount = float(np.clip(config.amount, 0.0, 2.0))
    radius = float(np.clip(config.radius, 0.3, 4.0))
    threshold = float(np.clip(config.threshold, 0.0, 0.25))

    blur = gaussian_blur(y, sigma=radius)
    detail = y - blur
    abs_detail = np.abs(detail)
    detail_soft = np.sign(detail) * np.maximum(abs_detail - threshold, 0.0)

    grad = gradient_magnitude(y)
    edge_low = float(np.percentile(grad, np.clip(config.edge_percentile_low, 30.0, 90.0)))
    edge_high = float(np.percentile(grad, np.clip(config.edge_percentile_high, 75.0, 99.9)))
    if edge_high <= edge_low:
        edge_high = edge_low + 1e-6

    edge_mask = smoothstep(edge_low, edge_high, grad)
    edge_mask = gaussian_blur(edge_mask, sigma=max(0.0, float(config.edge_feather_sigma)))
    edge_mask = np.clip(edge_mask, 0.0, 1.0)

    strong_edge_value = float(np.percentile(grad, np.clip(config.strong_edge_percentile, 90.0, 99.99)))
    strong_edge_mask = smoothstep(strong_edge_value * 0.85, strong_edge_value * 1.15, grad)
    halo_guard = 1.0 - (np.clip(config.strong_edge_suppression, 0.0, 1.0) * strong_edge_mask)
    halo_guard = np.clip(halo_guard, 0.15, 1.0)

    shadow_mask = 1.0 - smoothstep(0.0, float(np.clip(config.shadow_luma_end, 0.01, 0.35)), y)
    highlight_mask = smoothstep(float(np.clip(config.highlight_luma_start, 0.65, 0.99)), 1.0, y)
    tone_guard = 1.0 - (
        np.clip(config.shadow_protection, 0.0, 1.0) * shadow_mask
        + np.clip(config.highlight_protection, 0.0, 1.0) * highlight_mask
    )
    tone_guard = np.clip(tone_guard, 0.15, 1.0)

    skin_mask = np.zeros_like(y, dtype=np.float32)
    if is_color_image:
        skin_mask = estimate_skin_mask_from_ycbcr(y, cb, cr).astype(np.float32)
    skin_guard = 1.0 - (np.clip(config.skin_protection, 0.0, 1.0) * skin_mask)

    face_mask = np.zeros_like(y, dtype=np.float32)
    faces_detected = 0
    face_warning = face_detector.warning
    if config.face_sharpen_enabled:
        faces = face_detector.detect(y, min_size_ratio=config.face_min_size_ratio)
        faces_detected = int(faces.shape[0])
        if faces_detected > 0:
            face_mask = build_soft_face_mask(y.shape, faces)

    edge_priority = smoothstep(edge_low, edge_high, grad)
    face_edge_priority = float(np.clip(config.face_edge_priority, 0.0, 1.0))
    face_emphasis = face_mask * ((1.0 - face_edge_priority) + (face_edge_priority * edge_priority))
    face_boost = float(max(1.0, config.face_boost))
    face_boost_map = 1.0 + ((face_boost - 1.0) * face_emphasis)

    detail_clip_base = float(max(0.0, config.detail_clip_base))
    local_contrast = gaussian_blur(abs_detail, sigma=max(0.5, radius * 0.75))
    detail_cap = detail_clip_base + threshold + (1.40 * local_contrast)
    detail_clamped = np.clip(detail_soft, -detail_cap, detail_cap)

    local_weight = edge_mask * halo_guard * tone_guard * skin_guard
    delta = amount * local_weight * face_boost_map * detail_clamped
    delta = np.clip(delta, -0.18, 0.18)

    clip_threshold = float(max(0.0, config.max_clip_fraction))
    guard_iters = 0
    blend = 1.0
    clip_fraction = 0.0
    y_sharp = y

    while True:
        y_candidate = y + (delta * blend)
        clip_fraction = float(np.mean((y_candidate < 0.0) | (y_candidate > 1.0)))
        y_sharp = y_candidate
        if clip_fraction <= clip_threshold:
            break
        if guard_iters >= max(0, int(config.max_guardrail_iterations)):
            break
        blend *= 0.75
        guard_iters += 1

    y_sharp = np.clip(y_sharp, 0.0, 1.0)

    if is_color_image:
        output = ycbcr_to_rgb(y_sharp, cb, cr)
    else:
        output = np.repeat(y_sharp[:, :, None], 3, axis=2)

    output = np.clip(output, 0.0, 1.0).astype(np.float32, copy=False)
    y_after, _, _ = rgb_to_ycbcr(output)
    sharp_after = variance_of_laplacian(y_after)

    metadata: dict[str, Any] = {
        "sharpen_applied": True,
        "sharpen_method": config.method,
        "sharpen_workflow": "color" if is_color_image else "near-grayscale",
        "sharpen_amount": amount,
        "sharpen_radius": radius,
        "sharpen_threshold": threshold,
        "sharpen_edge_strength_mean": float(np.mean(grad)),
        "sharpen_edge_mask_mean": float(np.mean(edge_mask)),
        "sharpen_edge_mask_p95": float(np.percentile(edge_mask, 95.0)),
        "sharpen_halo_guard_mean": float(np.mean(halo_guard)),
        "sharpen_clip_fraction_preclip": clip_fraction,
        "sharpen_clip_fraction_threshold": clip_threshold,
        "sharpen_guardrail_iterations": int(guard_iters),
        "sharpen_face_enabled": bool(config.face_sharpen_enabled),
        "sharpen_face_detector": face_detector.detector_name,
        "sharpen_faces_detected": faces_detected,
        "sharpen_face_boost_factor": face_boost if config.face_sharpen_enabled else 1.0,
        "sharpen_face_mask_mean": float(np.mean(face_mask)) if config.face_sharpen_enabled else 0.0,
        "sharpen_sharpness_proxy_before": float(sharp_before),
        "sharpen_sharpness_proxy_after": float(sharp_after),
    }
    if face_warning is not None:
        metadata["sharpen_face_warning"] = face_warning
    return output, metadata


def sharpen_baseline_metadata(
    image_srgb: np.ndarray,
    *,
    method: str,
    is_color_image: bool,
    face_enabled: bool,
    face_warning: str | None,
    face_detector: str,
) -> dict[str, Any]:
    y, _, _ = rgb_to_ycbcr(np.clip(image_srgb, 0.0, 1.0).astype(np.float32, copy=False))
    sharp = variance_of_laplacian(y)
    metadata: dict[str, Any] = {
        "sharpen_method": method,
        "sharpen_workflow": "color" if is_color_image else "near-grayscale",
        "sharpen_amount": 0.0,
        "sharpen_radius": 0.0,
        "sharpen_threshold": 0.0,
        "sharpen_edge_strength_mean": 0.0,
        "sharpen_edge_mask_mean": 0.0,
        "sharpen_edge_mask_p95": 0.0,
        "sharpen_halo_guard_mean": 1.0,
        "sharpen_clip_fraction_preclip": 0.0,
        "sharpen_clip_fraction_threshold": 0.0,
        "sharpen_guardrail_iterations": 0,
        "sharpen_face_enabled": bool(face_enabled),
        "sharpen_face_detector": face_detector,
        "sharpen_faces_detected": 0,
        "sharpen_face_boost_factor": 1.0,
        "sharpen_face_mask_mean": 0.0,
        "sharpen_sharpness_proxy_before": float(sharp),
        "sharpen_sharpness_proxy_after": float(sharp),
    }
    if face_warning is not None:
        metadata["sharpen_face_warning"] = face_warning
    return metadata


def build_soft_face_mask(shape: tuple[int, int], faces: np.ndarray) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=np.float32)
    sizes: list[float] = []

    for x, y, fw, fh in faces:
        expand_x = int(round(0.18 * fw))
        expand_y = int(round(0.22 * fh))
        x0 = max(0, int(x - expand_x))
        y0 = max(0, int(y - expand_y))
        x1 = min(w, int(x + fw + expand_x))
        y1 = min(h, int(y + fh + expand_y))
        if x1 <= x0 or y1 <= y0:
            continue
        mask[y0:y1, x0:x1] = np.maximum(mask[y0:y1, x0:x1], 1.0)
        sizes.append(float(min(fw, fh)))

    if not sizes:
        return mask

    sigma = max(2.0, float(np.median(sizes)) * 0.12)
    mask = gaussian_blur(mask, sigma=sigma)
    max_val = float(np.max(mask))
    if max_val > 1e-6:
        mask /= max_val
    return np.clip(mask, 0.0, 1.0)


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
