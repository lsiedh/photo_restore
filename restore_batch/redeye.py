from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .models import ImageContext

LOGGER = logging.getLogger("restore_batch")


@dataclass(frozen=True)
class RedEyeConfig:
    mode: str = "off"
    strength: float = 0.70
    red_ratio: float = 1.65
    min_red: float = 0.20
    min_red_excess: float = 0.12
    min_eye_px: int = 12
    min_mask_px: int = 8
    max_mask_fraction: float = 0.12
    feather_sigma: float = 1.20
    darken_factor: float = 0.55


@dataclass(frozen=True)
class FaceBox:
    x: int
    y: int
    w: int
    h: int
    confidence: float = 1.0


@dataclass(frozen=True)
class EyeBox:
    x: int
    y: int
    w: int
    h: int
    face_index: int


class OpenCvRedEyeDetector:
    name = "opencv-haar-face-eye"

    def __init__(self) -> None:
        self._cv2: Any | None = None
        self._face_cascade: Any | None = None
        self._eye_cascade: Any | None = None
        self._warning: str | None = None
        self._version: str | None = None

        try:
            import cv2  # type: ignore
        except Exception as exc:  # noqa: BLE001 - optional dependency.
            self._warning = f"opencv-unavailable: {exc}"
            return

        face_candidates = [
            Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml",
            Path(cv2.data.haarcascades) / "haarcascade_frontalface_alt2.xml",
        ]
        eye_candidates = [
            Path(cv2.data.haarcascades) / "haarcascade_eye_tree_eyeglasses.xml",
            Path(cv2.data.haarcascades) / "haarcascade_eye.xml",
        ]
        face_path = next((p for p in face_candidates if p.exists()), None)
        eye_path = next((p for p in eye_candidates if p.exists()), None)
        if face_path is None or eye_path is None:
            self._warning = "opencv-haarcascade-not-found"
            return

        face_cascade = cv2.CascadeClassifier(str(face_path))
        eye_cascade = cv2.CascadeClassifier(str(eye_path))
        if face_cascade.empty() or eye_cascade.empty():
            self._warning = "opencv-haarcascade-load-failed"
            return

        self._cv2 = cv2
        self._face_cascade = face_cascade
        self._eye_cascade = eye_cascade
        self._version = str(getattr(cv2, "__version__", "unknown"))

    @property
    def warning(self) -> str | None:
        return self._warning

    @property
    def version(self) -> str | None:
        return self._version

    def is_available(self) -> tuple[bool, str | None]:
        ok = self._cv2 is not None and self._face_cascade is not None and self._eye_cascade is not None
        return ok, self._warning

    def detect(self, image_rgb_f32: np.ndarray, *, min_eye_px: int) -> tuple[list[FaceBox], list[EyeBox]]:
        ok, warning = self.is_available()
        if not ok:
            raise RuntimeError(warning or "detector-unavailable")

        gray = rgb_to_gray_u8(np.clip(image_rgb_f32, 0.0, 1.0).astype(np.float32, copy=False))
        h, w = gray.shape
        min_face_side = max(40, int(round(min(h, w) * 0.08)))

        faces_raw = self._face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.08,
            minNeighbors=5,
            minSize=(min_face_side, min_face_side),
        )
        if faces_raw is None or len(faces_raw) == 0:
            return [], []

        faces: list[FaceBox] = []
        eyes: list[EyeBox] = []
        for face_index, (fx, fy, fw, fh) in enumerate(np.asarray(faces_raw, dtype=np.int32)):
            faces.append(FaceBox(int(fx), int(fy), int(fw), int(fh), 1.0))

            eye_search_h = max(int(round(fh * 0.60)), min_eye_px * 2)
            y1 = min(h, int(fy + eye_search_h))
            face_roi = gray[int(fy) : y1, int(fx) : int(fx + fw)]
            if face_roi.size == 0:
                continue

            min_eye_side = max(int(min_eye_px), int(round(min(fw, fh) * 0.10)))
            max_eye_side = max(min_eye_side + 1, int(round(min(fw, fh) * 0.55)))
            eyes_raw = self._eye_cascade.detectMultiScale(
                face_roi,
                scaleFactor=1.05,
                minNeighbors=4,
                minSize=(min_eye_side, min_eye_side),
                maxSize=(max_eye_side, max_eye_side),
            )
            if eyes_raw is None or len(eyes_raw) == 0:
                continue

            # Keep the strongest eye candidates by area per face to avoid overprocessing.
            eye_boxes = sorted(np.asarray(eyes_raw, dtype=np.int32).tolist(), key=lambda b: b[2] * b[3], reverse=True)[:4]
            for ex, ey, ew, eh in eye_boxes:
                gx = int(fx + ex)
                gy = int(fy + ey)
                eyes.append(EyeBox(gx, gy, int(ew), int(eh), face_index))

        return faces, dedupe_eyes(eyes, iou_threshold=0.35)


class RedEyeStage:
    name = "redeye-reduction"

    def __init__(self, config: RedEyeConfig) -> None:
        self.config = config
        self.detector = OpenCvRedEyeDetector()
        ok, warning = self.detector.is_available()
        if not ok and warning is not None:
            LOGGER.warning("Red-eye reduction detector unavailable: %s", warning)

    def process(self, context: ImageContext) -> ImageContext:
        if context.image_f32 is None:
            raise RuntimeError("Red-eye stage expected image data.")

        image_type = str(context.metadata.get("image_type", "true-color"))
        is_color_image = image_type == "true-color"
        if self.config.mode == "off":
            context.metadata.update(redeye_disabled_metadata(config=self.config, detector=self.detector))
            return context
        if not is_color_image:
            context.metadata.update(redeye_skipped_metadata(config=self.config, detector=self.detector, reason="non-color-image"))
            return context

        output, metadata = apply_redeye_reduction(
            image_srgb=context.image_f32,
            config=self.config,
            detector=self.detector,
        )
        context.image_f32 = output
        context.metadata.update(metadata)
        return context


def apply_redeye_reduction(
    *,
    image_srgb: np.ndarray,
    config: RedEyeConfig,
    detector: OpenCvRedEyeDetector,
) -> tuple[np.ndarray, dict[str, Any]]:
    source = np.clip(image_srgb, 0.0, 1.0).astype(np.float32, copy=False)
    output = source.copy()

    det_ok, det_warning = detector.is_available()
    metadata: dict[str, Any] = {
        "redeye_requested": True,
        "redeye_enabled": True,
        "redeye_applied": False,
        "redeye_method": "haar-eye-red-dominance",
        "redeye_detector": detector.name,
        "redeye_detector_version": detector.version,
        "redeye_detector_available": bool(det_ok),
        "redeye_detector_warning": det_warning,
        "redeye_strength": float(np.clip(config.strength, 0.0, 1.0)),
        "redeye_red_ratio": float(max(1.0, config.red_ratio)),
        "redeye_min_red": float(np.clip(config.min_red, 0.0, 1.0)),
        "redeye_min_red_excess": float(np.clip(config.min_red_excess, 0.0, 1.0)),
        "redeye_min_eye_px": int(max(1, config.min_eye_px)),
        "redeye_min_mask_px": int(max(1, config.min_mask_px)),
        "redeye_max_mask_fraction": float(np.clip(config.max_mask_fraction, 0.01, 1.0)),
        "redeye_feather_sigma": float(max(0.0, config.feather_sigma)),
        "redeye_darken_factor": float(np.clip(config.darken_factor, 0.05, 1.0)),
        "redeye_faces_detected": 0,
        "redeye_eyes_detected": 0,
        "redeye_eyes_processed": 0,
        "redeye_eyes_skipped": 0,
        "redeye_pixels_corrected": 0,
        "redeye_detected_faces": [],
        "redeye_processed_eyes": [],
        "redeye_skipped_eyes": [],
    }
    if not det_ok:
        metadata["redeye_skipped_reason"] = "detector-unavailable"
        return source, metadata

    try:
        faces, eyes = detector.detect(source, min_eye_px=max(1, int(config.min_eye_px)))
    except Exception as exc:  # noqa: BLE001 - fail gracefully.
        metadata["redeye_detector_warning"] = str(exc)
        metadata["redeye_skipped_reason"] = "detector-error"
        return source, metadata

    metadata["redeye_faces_detected"] = int(len(faces))
    metadata["redeye_eyes_detected"] = int(len(eyes))
    metadata["redeye_detected_faces"] = [face_to_meta(face) for face in faces]
    if len(faces) == 0:
        metadata["redeye_skipped_reason"] = "no-faces-detected"
        return source, metadata
    if len(eyes) == 0:
        metadata["redeye_skipped_reason"] = "no-eyes-detected"
        return source, metadata

    processed_eyes: list[dict[str, Any]] = []
    skipped_eyes: list[dict[str, Any]] = []
    total_pixels_corrected = 0
    safe_strength = float(np.clip(config.strength, 0.0, 1.0))
    safe_red_ratio = float(max(1.0, config.red_ratio))
    safe_min_red = float(np.clip(config.min_red, 0.0, 1.0))
    safe_min_red_excess = float(np.clip(config.min_red_excess, 0.0, 1.0))
    safe_min_mask_px = int(max(1, config.min_mask_px))
    safe_max_mask_fraction = float(np.clip(config.max_mask_fraction, 0.01, 1.0))
    safe_feather_sigma = float(max(0.0, config.feather_sigma))
    safe_darken = float(np.clip(config.darken_factor, 0.05, 1.0))

    for eye_index, eye in enumerate(eyes):
        x0 = max(0, int(eye.x))
        y0 = max(0, int(eye.y))
        x1 = min(output.shape[1], int(eye.x + eye.w))
        y1 = min(output.shape[0], int(eye.y + eye.h))
        if x1 <= x0 or y1 <= y0:
            skipped_eyes.append({"index": eye_index, "reason": "invalid-eye-bbox", **eye_to_meta(eye)})
            continue

        patch = output[y0:y1, x0:x1, :]
        if patch.shape[0] < int(config.min_eye_px) or patch.shape[1] < int(config.min_eye_px):
            skipped_eyes.append({"index": eye_index, "reason": "eye-too-small", **eye_to_meta(eye)})
            continue

        raw_mask = red_eye_mask(
            patch_rgb=patch,
            red_ratio=safe_red_ratio,
            min_red=safe_min_red,
            min_red_excess=safe_min_red_excess,
        )
        raw_pixels = int(np.count_nonzero(raw_mask))
        raw_fraction = float(raw_pixels / max(raw_mask.size, 1))
        if raw_pixels < safe_min_mask_px:
            skipped_eyes.append(
                {
                    "index": eye_index,
                    "reason": "low-red-pixel-count",
                    "mask_pixels": raw_pixels,
                    "mask_fraction": raw_fraction,
                    **eye_to_meta(eye),
                }
            )
            continue
        if raw_fraction > safe_max_mask_fraction:
            skipped_eyes.append(
                {
                    "index": eye_index,
                    "reason": "high-mask-fraction-risk",
                    "mask_pixels": raw_pixels,
                    "mask_fraction": raw_fraction,
                    **eye_to_meta(eye),
                }
            )
            continue

        mask = raw_mask.astype(np.float32, copy=False)
        if safe_feather_sigma > 1e-6:
            mask = gaussian_blur(mask, sigma=safe_feather_sigma)
            max_val = float(np.max(mask))
            if max_val > 1e-8:
                mask /= max_val
        alpha = np.clip(mask * safe_strength, 0.0, 1.0)
        if float(np.max(alpha)) < 1e-4:
            skipped_eyes.append({"index": eye_index, "reason": "low-alpha", **eye_to_meta(eye)})
            continue

        corrected_patch = correct_redeye_patch(patch, alpha=alpha, darken_factor=safe_darken)
        output[y0:y1, x0:x1, :] = corrected_patch

        corrected_pixels = int(np.count_nonzero(alpha > 0.05))
        total_pixels_corrected += corrected_pixels
        processed_eyes.append(
            {
                "index": eye_index,
                "mask_pixels": raw_pixels,
                "mask_fraction": raw_fraction,
                "alpha_max": float(np.max(alpha)),
                "corrected_pixels": corrected_pixels,
                **eye_to_meta(eye),
            }
        )

    output = np.clip(output, 0.0, 1.0).astype(np.float32, copy=False)
    metadata.update(
        {
            "redeye_applied": len(processed_eyes) > 0,
            "redeye_eyes_processed": int(len(processed_eyes)),
            "redeye_eyes_skipped": int(len(skipped_eyes)),
            "redeye_pixels_corrected": int(total_pixels_corrected),
            "redeye_processed_eyes": processed_eyes,
            "redeye_skipped_eyes": skipped_eyes,
        }
    )
    if len(processed_eyes) == 0:
        metadata["redeye_skipped_reason"] = "all-eyes-skipped"
    return output, metadata


def redeye_disabled_metadata(*, config: RedEyeConfig, detector: OpenCvRedEyeDetector) -> dict[str, Any]:
    det_ok, det_warning = detector.is_available()
    return {
        "redeye_requested": False,
        "redeye_enabled": False,
        "redeye_applied": False,
        "redeye_method": "haar-eye-red-dominance",
        "redeye_detector": detector.name,
        "redeye_detector_version": detector.version,
        "redeye_detector_available": bool(det_ok),
        "redeye_detector_warning": det_warning,
        "redeye_strength": float(np.clip(config.strength, 0.0, 1.0)),
        "redeye_red_ratio": float(max(1.0, config.red_ratio)),
        "redeye_min_red": float(np.clip(config.min_red, 0.0, 1.0)),
        "redeye_min_red_excess": float(np.clip(config.min_red_excess, 0.0, 1.0)),
        "redeye_min_eye_px": int(max(1, config.min_eye_px)),
        "redeye_min_mask_px": int(max(1, config.min_mask_px)),
        "redeye_max_mask_fraction": float(np.clip(config.max_mask_fraction, 0.01, 1.0)),
        "redeye_feather_sigma": float(max(0.0, config.feather_sigma)),
        "redeye_darken_factor": float(np.clip(config.darken_factor, 0.05, 1.0)),
        "redeye_faces_detected": 0,
        "redeye_eyes_detected": 0,
        "redeye_eyes_processed": 0,
        "redeye_eyes_skipped": 0,
        "redeye_pixels_corrected": 0,
        "redeye_detected_faces": [],
        "redeye_processed_eyes": [],
        "redeye_skipped_eyes": [],
        "redeye_skipped_reason": "redeye-off",
    }


def redeye_skipped_metadata(*, config: RedEyeConfig, detector: OpenCvRedEyeDetector, reason: str) -> dict[str, Any]:
    data = redeye_disabled_metadata(config=config, detector=detector)
    data["redeye_requested"] = True
    data["redeye_enabled"] = True
    data["redeye_skipped_reason"] = reason
    return data


def red_eye_mask(*, patch_rgb: np.ndarray, red_ratio: float, min_red: float, min_red_excess: float) -> np.ndarray:
    r = patch_rgb[:, :, 0]
    g = patch_rgb[:, :, 1]
    b = patch_rgb[:, :, 2]
    luma = (0.2126 * r) + (0.7152 * g) + (0.0722 * b)

    red_dominance = (r >= min_red) & (r > (g * red_ratio)) & (r > (b * red_ratio))
    not_skin_like = ((g + b) <= 1.2) & (luma <= 0.90) & (luma >= 0.02)
    chroma_margin = (r - np.maximum(g, b)) >= min_red_excess
    center_prior = eye_center_prior(patch_rgb.shape[0], patch_rgb.shape[1])
    return red_dominance & not_skin_like & chroma_margin & center_prior


def correct_redeye_patch(patch_rgb: np.ndarray, *, alpha: np.ndarray, darken_factor: float) -> np.ndarray:
    r = patch_rgb[:, :, 0]
    g = patch_rgb[:, :, 1]
    b = patch_rgb[:, :, 2]
    neutral = 0.5 * (g + b)

    # Keep red reduction conservative and avoid pushing corrected pupils toward green/cyan.
    target_r = np.minimum(neutral * darken_factor, np.maximum(g, b) * 1.02)
    target_r = np.maximum(target_r, neutral * 0.68)
    target_r = np.clip(target_r, 0.0, 1.0)
    target_gb = np.clip((0.70 * neutral) + (0.30 * target_r), 0.0, 1.0)
    target_gb = np.minimum(target_gb, target_r + 0.03)
    target_g = target_gb
    target_b = target_gb

    r_new = r + (alpha * (target_r - r))
    g_new = g + (alpha * (target_g - g))
    b_new = b + (alpha * (target_b - b))
    return np.clip(np.stack((r_new, g_new, b_new), axis=2), 0.0, 1.0).astype(np.float32, copy=False)


def dedupe_eyes(eyes: list[EyeBox], *, iou_threshold: float) -> list[EyeBox]:
    if not eyes:
        return []
    sorted_eyes = sorted(eyes, key=lambda e: (e.w * e.h), reverse=True)
    kept: list[EyeBox] = []
    for eye in sorted_eyes:
        if all(iou_eye(eye, k) < iou_threshold for k in kept):
            kept.append(eye)
    return kept


def iou_eye(a: EyeBox, b: EyeBox) -> float:
    ax0, ay0, ax1, ay1 = a.x, a.y, a.x + a.w, a.y + a.h
    bx0, by0, bx1, by1 = b.x, b.y, b.x + b.w, b.y + b.h
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    iw = max(0, ix1 - ix0)
    ih = max(0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = (a.w * a.h) + (b.w * b.h) - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def face_to_meta(face: FaceBox) -> dict[str, Any]:
    return {"bbox": [int(face.x), int(face.y), int(face.w), int(face.h)], "confidence": float(face.confidence)}


def eye_to_meta(eye: EyeBox) -> dict[str, Any]:
    return {"bbox": [int(eye.x), int(eye.y), int(eye.w), int(eye.h)], "face_index": int(eye.face_index)}


def rgb_to_gray_u8(rgb_f32: np.ndarray) -> np.ndarray:
    y = (0.2990 * rgb_f32[:, :, 0]) + (0.5870 * rgb_f32[:, :, 1]) + (0.1140 * rgb_f32[:, :, 2])
    return np.clip(y * 255.0, 0.0, 255.0).astype(np.uint8)


def eye_center_prior(height: int, width: int) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    y = (yy + 0.5 - (height * 0.5)) / max(height * 0.5, 1.0)
    x = (xx + 0.5 - (width * 0.5)) / max(width * 0.5, 1.0)
    # Keep a soft ellipse around the eye center to reduce false positives.
    return ((x * x) / 1.15 + (y * y) / 1.00) <= 1.0


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
