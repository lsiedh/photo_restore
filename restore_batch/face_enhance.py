from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from PIL import Image

from .models import ImageContext

LOGGER = logging.getLogger("restore_batch")
GFPGAN_DEFAULT_MODEL_URL = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
GFPGAN_MODEL_FILENAMES = ("GFPGANv1.4.pth", "GFPGANv1.3.pth")


@dataclass(frozen=True)
class FaceEnhancementConfig:
    mode: str = "on"
    backend: str = "gfpgan"
    strength: float = 0.35
    codeformer_fidelity: float = 0.70
    blend: float = 0.60
    feather: float = 0.15
    crop_expand: float = 0.20
    min_face_px: int = 80
    min_face_conf: float = 0.60
    overlap_iou_threshold: float = 0.45
    min_luma_change_apply: float = 0.002
    max_luma_change: float = 0.12
    max_luma_change_reject: float = 0.22
    match_local_luminance: bool = True
    require_eye_evidence: bool = True
    eye_check_max_face_px: int = 10000
    min_eye_count: int = 1
    small_face_eye_gate_px: int = 145
    small_face_min_eye_count: int = 2
    min_detail_lap_var: float = 7.5e-05
    detail_check_max_face_px: int = 150
    save_face_previews: bool = False
    previews_subdir: str = "_face_previews"
    model_path: Path | None = None


@dataclass(frozen=True)
class FaceDetection:
    x: int
    y: int
    w: int
    h: int
    confidence: float


@dataclass(frozen=True)
class FaceEnhanceResult:
    image_rgb: np.ndarray
    metadata: dict[str, Any]


class FaceEnhanceBackend(Protocol):
    name: str

    def is_available(self) -> tuple[bool, str | None]:
        ...

    def enhance_face(
        self,
        face_rgb_f32: np.ndarray,
        *,
        strength: float,
        fidelity: float,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        ...


class FaceDetector(Protocol):
    name: str

    def is_available(self) -> tuple[bool, str | None]:
        ...

    def detect(self, image_rgb_f32: np.ndarray, *, min_face_px: int) -> list[FaceDetection]:
        ...


class OpenCvHaarFaceDetector:
    name = "opencv-haar"

    def __init__(self) -> None:
        self._warning: str | None = None
        self._cv2: Any | None = None
        self._cascade: Any | None = None
        self._version: str | None = None

        try:
            import cv2  # type: ignore
        except Exception as exc:  # noqa: BLE001 - optional dependency.
            self._warning = f"opencv-unavailable: {exc}"
            return

        cascade_candidates = [
            Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml",
            Path(cv2.data.haarcascades) / "haarcascade_frontalface_alt2.xml",
        ]
        cascade_path = next((p for p in cascade_candidates if p.exists()), None)
        if cascade_path is None:
            self._warning = "opencv-haarcascade-not-found"
            return

        cascade = cv2.CascadeClassifier(str(cascade_path))
        if cascade.empty():
            self._warning = f"opencv-haarcascade-load-failed: {cascade_path}"
            return

        self._cv2 = cv2
        self._cascade = cascade
        self._version = str(getattr(cv2, "__version__", "unknown"))

    def is_available(self) -> tuple[bool, str | None]:
        return (self._cv2 is not None and self._cascade is not None), self._warning

    @property
    def version(self) -> str | None:
        return self._version

    def detect(self, image_rgb_f32: np.ndarray, *, min_face_px: int) -> list[FaceDetection]:
        available, warning = self.is_available()
        if not available:
            raise RuntimeError(warning or "detector-unavailable")

        gray = rgb_to_gray_u8(image_rgb_f32)
        min_side = max(16, int(min_face_px))

        detections: list[FaceDetection] = []
        detect_fn3 = getattr(self._cascade, "detectMultiScale3", None)
        if callable(detect_fn3):
            faces, _reject_levels, level_weights = self._cascade.detectMultiScale3(
                gray,
                scaleFactor=1.10,
                minNeighbors=4,
                minSize=(min_side, min_side),
                outputRejectLevels=True,
            )
            if faces is not None and len(faces) > 0:
                for (x, y, w, h), weight in zip(faces, level_weights, strict=False):
                    conf = sigmoid_confidence(float(weight))
                    detections.append(FaceDetection(int(x), int(y), int(w), int(h), conf))
                return detections

        faces = self._cascade.detectMultiScale(
            gray,
            scaleFactor=1.10,
            minNeighbors=4,
            minSize=(min_side, min_side),
        )
        if faces is None or len(faces) == 0:
            return []

        for x, y, w, h in faces:
            detections.append(FaceDetection(int(x), int(y), int(w), int(h), 1.0))
        return detections


class GfpganBackend:
    name = "gfpgan"

    def __init__(self, *, model_path: Path | None) -> None:
        self._warning: str | None = None
        self._restorer: Any | None = None
        self._model_path = model_path
        self._model_candidates = resolve_gfpgan_model_candidates(model_path)
        self._model_source: str | None = None
        self._model_resolution: str | None = None
        self._version: str | None = None
        self._init_attempted = False
        self._init_failed = False

    def is_available(self) -> tuple[bool, str | None]:
        if self._restorer is not None:
            return True, None
        if self._init_failed:
            return False, self._warning
        if self._init_attempted:
            return False, self._warning
        self._init_attempted = True

        if not self._model_candidates:
            self._init_failed = True
            self._warning = "gfpgan-model-unresolved"
            return False, self._warning

        attempt_errors: list[str] = []
        for model_source, model_resolution in self._model_candidates:
            ok, warning = self._try_initialize_with_source(
                model_source=model_source,
                model_resolution=model_resolution,
            )
            if ok:
                return True, None
            if warning is not None:
                attempt_errors.append(warning)
                LOGGER.warning(
                    "Face enhancement: GFPGAN initialization failed for %s candidate '%s': %s",
                    model_resolution,
                    model_source,
                    warning,
                )

        self._init_failed = True
        if attempt_errors:
            self._warning = attempt_errors[-1]
            if len(attempt_errors) > 1:
                tail = " | ".join(attempt_errors[-3:])
                self._warning = f"gfpgan-init-failed-all: {tail}"
        else:
            self._warning = "gfpgan-model-unresolved"
        return False, self._warning

    def _try_initialize_with_source(self, *, model_source: str, model_resolution: str) -> tuple[bool, str | None]:
        source_for_init = model_source
        if not is_url_source(model_source):
            resolved_path = Path(model_source).expanduser().resolve()
            if not resolved_path.exists():
                return False, f"gfpgan-model-not-found: {resolved_path}"
            source_for_init = str(resolved_path)

        if self._model_path is None and model_resolution == "auto-url":
            LOGGER.info("Face enhancement: using default GFPGAN model URL for auto-download.")

        ensure_gfpgan_import_compatibility()
        try:
            from gfpgan import GFPGANer  # type: ignore
        except Exception as exc:  # noqa: BLE001
            return False, f"gfpgan-import-failed: {exc}"

        try:
            restorer = GFPGANer(
                model_path=str(source_for_init),
                upscale=1,
                arch="clean",
                channel_multiplier=2,
                bg_upsampler=None,
            )
        except Exception as exc:  # noqa: BLE001
            return False, f"gfpgan-init-failed[{model_resolution}]: {exc}"

        self._restorer = restorer
        self._version = get_module_version("gfpgan")
        self._model_source = source_for_init
        self._model_resolution = model_resolution
        return True, None

    @property
    def version(self) -> str | None:
        return self._version

    @property
    def model_source(self) -> str | None:
        return self._model_source

    @property
    def model_resolution(self) -> str | None:
        return self._model_resolution

    def enhance_face(
        self,
        face_rgb_f32: np.ndarray,
        *,
        strength: float,
        fidelity: float,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        ok, warning = self.is_available()
        if not ok:
            raise RuntimeError(warning or "gfpgan-backend-unavailable")

        # GFPGAN expects BGR uint8.
        face_bgr_u8 = rgb_f32_to_bgr_u8(face_rgb_f32)
        safe_strength = float(np.clip(strength, 0.0, 1.0))
        # GFPGAN weight is interpolation with original; 1.0 is conservative.
        gfpgan_weight = float(np.clip(1.0 - (0.6 * safe_strength), 0.0, 1.0))

        _cropped, _restored, restored_bgr = self._restorer.enhance(
            face_bgr_u8,
            has_aligned=False,
            only_center_face=True,
            paste_back=True,
            weight=gfpgan_weight,
        )
        restored_rgb = bgr_u8_to_rgb_f32(restored_bgr)
        return restored_rgb, {
            "backend": self.name,
            "backend_version": self._version,
            "model_source": self._model_source,
            "model_resolution": self._model_resolution,
            "backend_weight": gfpgan_weight,
            "fidelity": float(np.clip(fidelity, 0.0, 1.0)),
        }


class CodeFormerBackend:
    name = "codeformer"

    def is_available(self) -> tuple[bool, str | None]:
        return False, "codeformer-backend-not-implemented"

    def enhance_face(
        self,
        face_rgb_f32: np.ndarray,
        *,
        strength: float,
        fidelity: float,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        raise RuntimeError("codeformer-backend-not-implemented")


class FaceEnhancementStage:
    name = "face-enhance"

    def __init__(self, config: FaceEnhancementConfig) -> None:
        self.config = config
        self.detector = OpenCvHaarFaceDetector()
        self.backend: FaceEnhanceBackend = create_backend(config)

    def process(self, context: ImageContext) -> ImageContext:
        if context.image_f32 is None:
            raise RuntimeError("Face enhancement stage expected image data.")

        image_type = str(context.metadata.get("image_type", "true-color"))
        is_color_image = image_type == "true-color"

        if self.config.mode == "off":
            context.metadata.update(face_enhance_disabled_metadata(config=self.config))
            return context

        source = np.clip(context.image_f32, 0.0, 1.0).astype(np.float32, copy=False)
        output = source.copy()

        det_ok, det_warning = self.detector.is_available()
        backend_ok, backend_warning = self.backend.is_available()

        metadata: dict[str, Any] = {
            "face_enhance_requested": True,
            "face_enhance_enabled": True,
            "face_enhance_applied": False,
            "face_enhance_backend": self.config.backend,
            "face_enhance_backend_available": bool(backend_ok),
            "face_enhance_backend_warning": backend_warning,
            "face_enhance_detector": self.detector.name,
            "face_enhance_detector_version": getattr(self.detector, "version", None),
            "face_enhance_detector_available": bool(det_ok),
            "face_enhance_detector_warning": det_warning,
            "face_enhance_alignment_method": "none",
            "face_enhance_strength": float(np.clip(self.config.strength, 0.0, 1.0)),
            "face_enhance_codeformer_fidelity": float(np.clip(self.config.codeformer_fidelity, 0.0, 1.0)),
            "face_enhance_blend": float(np.clip(self.config.blend, 0.0, 1.0)),
            "face_enhance_feather": float(np.clip(self.config.feather, 0.0, 1.0)),
            "face_enhance_min_face_px": int(max(1, self.config.min_face_px)),
            "face_enhance_min_face_conf": float(np.clip(self.config.min_face_conf, 0.0, 1.0)),
            "face_enhance_match_local_luminance": bool(self.config.match_local_luminance),
            "face_enhance_require_eye_evidence": bool(self.config.require_eye_evidence),
            "face_enhance_eye_check_max_face_px": int(max(1, self.config.eye_check_max_face_px)),
            "face_enhance_min_eye_count": int(max(0, self.config.min_eye_count)),
            "face_enhance_small_face_eye_gate_px": int(max(1, self.config.small_face_eye_gate_px)),
            "face_enhance_small_face_min_eye_count": int(max(0, self.config.small_face_min_eye_count)),
            "face_enhance_min_detail_lap_var": float(max(0.0, self.config.min_detail_lap_var)),
            "face_enhance_detail_check_max_face_px": int(max(1, self.config.detail_check_max_face_px)),
            "face_enhance_min_luma_change_apply": float(max(0.0, self.config.min_luma_change_apply)),
            "face_enhance_max_luma_change": float(max(1e-6, self.config.max_luma_change)),
            "face_enhance_max_luma_change_reject": float(
                max(self.config.min_luma_change_apply + 1e-6, self.config.max_luma_change_reject)
            ),
            "face_enhance_save_previews": bool(self.config.save_face_previews),
            "face_enhance_model_path": (str(self.config.model_path) if self.config.model_path else None),
            "face_enhance_model_source": getattr(self.backend, "model_source", None),
            "face_enhance_model_resolution": getattr(self.backend, "model_resolution", None),
            "face_enhance_faces_detected": 0,
            "face_enhance_detected_faces": [],
            "face_enhance_faces_processed": 0,
            "face_enhance_faces_skipped": 0,
            "face_enhance_processed_faces": [],
            "face_enhance_skipped_faces": [],
            "face_enhance_near_grayscale_neutralized": bool(not is_color_image),
            "face_enhance_preview_dir": None,
        }

        eye_detector_ok, eye_detector_warning = eye_detector_available()
        metadata["face_enhance_eye_detector_available"] = bool(eye_detector_ok)
        metadata["face_enhance_eye_detector_warning"] = eye_detector_warning

        if not det_ok:
            metadata["face_enhance_skipped_reason"] = "detector-unavailable"
            context.metadata.update(metadata)
            return context

        if not backend_ok:
            metadata["face_enhance_skipped_reason"] = "backend-unavailable"
            context.metadata.update(metadata)
            return context

        detections = self.detector.detect(source, min_face_px=max(16, int(self.config.min_face_px // 2)))
        raw_faces_meta = [face_to_meta(fd) for fd in detections]
        metadata["face_enhance_faces_detected"] = int(len(detections))
        metadata["face_enhance_detected_faces"] = raw_faces_meta
        if len(detections) == 0:
            metadata["face_enhance_skipped_reason"] = "no-faces-detected"
            context.metadata.update(metadata)
            return context

        selected, skipped = select_faces(
            detections,
            min_face_px=int(max(1, self.config.min_face_px)),
            min_face_conf=float(np.clip(self.config.min_face_conf, 0.0, 1.0)),
            overlap_iou_threshold=float(np.clip(self.config.overlap_iou_threshold, 0.0, 1.0)),
        )

        processed_faces: list[dict[str, Any]] = []
        skipped_faces: list[dict[str, Any]] = skipped.copy()

        preview_root: Path | None = None
        if self.config.save_face_previews:
            preview_root = context.output_path.parent / self.config.previews_subdir / context.output_path.stem
            preview_root.mkdir(parents=True, exist_ok=True)

        for idx, detection in enumerate(selected):
            expanded_bbox = expand_bbox(
                detection=detection,
                image_shape=source.shape,
                expand_ratio=float(max(0.0, self.config.crop_expand)),
            )
            x0, y0, x1, y1 = expanded_bbox
            if x1 <= x0 or y1 <= y0:
                skipped_faces.append({"index": idx, "reason": "invalid-expanded-bbox", **face_to_meta(detection)})
                continue

            original_patch = source[y0:y1, x0:x1, :]
            aligned_patch, align_meta = align_face_patch(original_patch)
            min_side = min(original_patch.shape[0], original_patch.shape[1])

            detail_lap_var, detail_grad_p90 = patch_detail_proxies(original_patch)
            if (
                min_side <= int(max(1, self.config.detail_check_max_face_px))
                and detail_lap_var < float(max(0.0, self.config.min_detail_lap_var))
            ):
                skipped_faces.append(
                    {
                        "index": idx,
                        "reason": "low-detail-face",
                        "detail_lap_var": float(detail_lap_var),
                        "detail_grad_p90": float(detail_grad_p90),
                        **face_to_meta(detection),
                    }
                )
                continue

            eye_count = None
            if (
                bool(self.config.require_eye_evidence)
                and min_side <= int(max(1, self.config.eye_check_max_face_px))
                and eye_detector_ok
            ):
                eye_count = detect_eye_count(original_patch)
                min_eyes_required = int(max(0, self.config.min_eye_count))
                if min_side <= int(max(1, self.config.small_face_eye_gate_px)):
                    min_eyes_required = max(min_eyes_required, int(max(0, self.config.small_face_min_eye_count)))
                if eye_count < min_eyes_required:
                    skipped_faces.append(
                        {
                            "index": idx,
                            "reason": "low-eye-evidence",
                            "min_eyes_required": int(min_eyes_required),
                            "eye_count": int(eye_count),
                            "detail_lap_var": float(detail_lap_var),
                            "detail_grad_p90": float(detail_grad_p90),
                            **face_to_meta(detection),
                        }
                    )
                    continue

            try:
                enhanced_patch, backend_meta = self.backend.enhance_face(
                    aligned_patch,
                    strength=float(np.clip(self.config.strength, 0.0, 1.0)),
                    fidelity=float(np.clip(self.config.codeformer_fidelity, 0.0, 1.0)),
                )
            except Exception as exc:  # noqa: BLE001 - per-face fallback.
                skipped_faces.append({
                    "index": idx,
                    "reason": "backend-error",
                    "error": str(exc),
                    **face_to_meta(detection),
                })
                continue

            if enhanced_patch.shape[:2] != original_patch.shape[:2]:
                enhanced_patch = resize_patch(enhanced_patch, original_patch.shape[1], original_patch.shape[0])

            if self.config.match_local_luminance:
                enhanced_patch = match_patch_luminance(enhanced_patch, original_patch)

            raw_change = patch_luma_mean_abs_diff(original_patch, enhanced_patch)
            min_apply_change = float(max(0.0, self.config.min_luma_change_apply))
            if raw_change < min_apply_change:
                skipped_faces.append(
                    {
                        "index": idx,
                        "reason": "low-expected-benefit",
                        "change_metric_luma_abs_before": float(raw_change),
                        **face_to_meta(detection),
                    }
                )
                continue

            reject_change = float(max(min_apply_change + 1e-6, self.config.max_luma_change_reject))
            if raw_change > reject_change:
                skipped_faces.append(
                    {
                        "index": idx,
                        "reason": "high-change-risk",
                        "change_metric_luma_abs_before": float(raw_change),
                        **face_to_meta(detection),
                    }
                )
                continue

            max_change = float(max(1e-6, self.config.max_luma_change))
            if raw_change > max_change:
                scale = max_change / raw_change
                enhanced_patch = original_patch + ((enhanced_patch - original_patch) * scale)
                clamped_change = patch_luma_mean_abs_diff(original_patch, enhanced_patch)
            else:
                clamped_change = raw_change

            if not is_color_image:
                enhanced_patch = neutralize_patch(enhanced_patch)

            mask = soft_rect_mask(
                height=original_patch.shape[0],
                width=original_patch.shape[1],
                feather=float(np.clip(self.config.feather, 0.0, 0.95)),
            )
            blend_alpha = float(np.clip(self.config.blend, 0.0, 1.0)) * mask

            current_patch = output[y0:y1, x0:x1, :]
            blended_patch = (current_patch * (1.0 - blend_alpha[:, :, None])) + (enhanced_patch * blend_alpha[:, :, None])
            if not is_color_image:
                blended_patch = neutralize_patch(blended_patch)

            output[y0:y1, x0:x1, :] = np.clip(blended_patch, 0.0, 1.0)

            face_meta: dict[str, Any] = {
                "index": idx,
                **face_to_meta(detection),
                "expanded_bbox": [x0, y0, x1 - x0, y1 - y0],
                "alignment_method": str(align_meta.get("alignment_method", "none")),
                "change_metric_luma_abs_before": float(raw_change),
                "change_metric_luma_abs_after": float(clamped_change),
                "detail_lap_var": float(detail_lap_var),
                "detail_grad_p90": float(detail_grad_p90),
                "eye_count": (int(eye_count) if eye_count is not None else None),
                "backend_meta": backend_meta,
            }
            processed_faces.append(face_meta)

            if preview_root is not None:
                save_face_preview(preview_root, idx, "before", original_patch)
                save_face_preview(preview_root, idx, "enhanced", enhanced_patch)
                save_face_preview(preview_root, idx, "blended", np.clip(blended_patch, 0.0, 1.0))

        output = np.clip(output, 0.0, 1.0).astype(np.float32, copy=False)

        metadata.update(
            {
                "face_enhance_applied": len(processed_faces) > 0,
                "face_enhance_faces_processed": int(len(processed_faces)),
                "face_enhance_faces_skipped": int(len(skipped_faces)),
                "face_enhance_processed_faces": processed_faces,
                "face_enhance_skipped_faces": skipped_faces,
                "face_enhance_near_grayscale_neutralized": bool(not is_color_image),
                "face_enhance_preview_dir": str(preview_root) if preview_root is not None else None,
            }
        )
        if len(processed_faces) == 0 and len(skipped_faces) > 0:
            metadata["face_enhance_skipped_reason"] = "all-faces-skipped"

        context.image_f32 = output
        context.metadata.update(metadata)
        return context


def create_backend(config: FaceEnhancementConfig) -> FaceEnhanceBackend:
    if config.backend == "gfpgan":
        return GfpganBackend(model_path=config.model_path)
    if config.backend == "codeformer":
        return CodeFormerBackend()
    raise ValueError(f"Unsupported face enhancement backend: {config.backend}")


def face_enhance_disabled_metadata(*, config: FaceEnhancementConfig) -> dict[str, Any]:
    return {
        "face_enhance_requested": False,
        "face_enhance_enabled": False,
        "face_enhance_applied": False,
        "face_enhance_backend": config.backend,
        "face_enhance_faces_detected": 0,
        "face_enhance_faces_processed": 0,
        "face_enhance_faces_skipped": 0,
        "face_enhance_skipped_reason": "face-enhance-off",
        "face_enhance_blend": float(np.clip(config.blend, 0.0, 1.0)),
        "face_enhance_feather": float(np.clip(config.feather, 0.0, 1.0)),
        "face_enhance_strength": float(np.clip(config.strength, 0.0, 1.0)),
        "face_enhance_codeformer_fidelity": float(np.clip(config.codeformer_fidelity, 0.0, 1.0)),
        "face_enhance_require_eye_evidence": bool(config.require_eye_evidence),
        "face_enhance_eye_check_max_face_px": int(max(1, config.eye_check_max_face_px)),
        "face_enhance_min_eye_count": int(max(0, config.min_eye_count)),
        "face_enhance_small_face_eye_gate_px": int(max(1, config.small_face_eye_gate_px)),
        "face_enhance_small_face_min_eye_count": int(max(0, config.small_face_min_eye_count)),
        "face_enhance_min_detail_lap_var": float(max(0.0, config.min_detail_lap_var)),
        "face_enhance_detail_check_max_face_px": int(max(1, config.detail_check_max_face_px)),
        "face_enhance_min_luma_change_apply": float(max(0.0, config.min_luma_change_apply)),
        "face_enhance_max_luma_change": float(max(1e-6, config.max_luma_change)),
        "face_enhance_max_luma_change_reject": float(max(config.min_luma_change_apply + 1e-6, config.max_luma_change_reject)),
        "face_enhance_save_previews": bool(config.save_face_previews),
        "face_enhance_model_path": (str(config.model_path) if config.model_path else None),
        "face_enhance_model_source": (str(config.model_path) if config.model_path else None),
        "face_enhance_model_resolution": ("cli" if config.model_path else "disabled"),
    }


def select_faces(
    detections: list[FaceDetection],
    *,
    min_face_px: int,
    min_face_conf: float,
    overlap_iou_threshold: float,
) -> tuple[list[FaceDetection], list[dict[str, Any]]]:
    skipped: list[dict[str, Any]] = []
    selected: list[FaceDetection] = []

    sorted_dets = sorted(detections, key=lambda d: (d.w * d.h), reverse=True)
    for det in sorted_dets:
        if det.w < min_face_px or det.h < min_face_px:
            skipped.append({"reason": "too-small", **face_to_meta(det)})
            continue
        if det.confidence < min_face_conf:
            skipped.append({"reason": "low-confidence", **face_to_meta(det)})
            continue

        overlaps = [iou(det, exist) for exist in selected]
        if overlaps and max(overlaps) >= overlap_iou_threshold:
            skipped.append({"reason": "overlap", **face_to_meta(det)})
            continue
        selected.append(det)

    return selected, skipped


def expand_bbox(*, detection: FaceDetection, image_shape: tuple[int, int, int], expand_ratio: float) -> tuple[int, int, int, int]:
    h, w = image_shape[:2]
    ex = int(round(detection.w * expand_ratio))
    ey = int(round(detection.h * expand_ratio))

    x0 = max(0, detection.x - ex)
    y0 = max(0, detection.y - int(round(ey * 1.10)))
    x1 = min(w, detection.x + detection.w + ex)
    y1 = min(h, detection.y + detection.h + int(round(ey * 0.90)))
    return x0, y0, x1, y1


def align_face_patch(face_patch: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    # Landmark-based alignment can be added later without changing stage API.
    return face_patch.astype(np.float32, copy=False), {"alignment_method": "none"}


def match_patch_luminance(enhanced: np.ndarray, reference: np.ndarray) -> np.ndarray:
    y_e, cb_e, cr_e = rgb_to_ycbcr(np.clip(enhanced, 0.0, 1.0).astype(np.float32, copy=False))
    y_r, _, _ = rgb_to_ycbcr(np.clip(reference, 0.0, 1.0).astype(np.float32, copy=False))

    mean_e = float(np.mean(y_e))
    std_e = float(np.std(y_e))
    mean_r = float(np.mean(y_r))
    std_r = float(np.std(y_r))

    if std_e < 1e-6:
        y_adj = np.full_like(y_e, mean_r, dtype=np.float32)
    else:
        gain = np.clip(std_r / std_e, 0.65, 1.35)
        y_adj = ((y_e - mean_e) * gain) + mean_r

    y_adj = np.clip(y_adj, 0.0, 1.0)
    return np.clip(ycbcr_to_rgb(y_adj, cb_e, cr_e), 0.0, 1.0).astype(np.float32, copy=False)


def neutralize_patch(patch: np.ndarray) -> np.ndarray:
    y = np.mean(patch, axis=2)
    return np.repeat(y[:, :, None], 3, axis=2).astype(np.float32, copy=False)


def patch_luma_mean_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    ya = rgb_to_luma(np.clip(a, 0.0, 1.0).astype(np.float32, copy=False))
    yb = rgb_to_luma(np.clip(b, 0.0, 1.0).astype(np.float32, copy=False))
    return float(np.mean(np.abs(ya - yb)))


def soft_rect_mask(*, height: int, width: int, feather: float) -> np.ndarray:
    feather = float(np.clip(feather, 0.0, 0.95))
    if feather <= 1e-6:
        return np.ones((height, width), dtype=np.float32)

    pad = max(1, int(round(min(height, width) * feather)))
    base = np.zeros((height, width), dtype=np.float32)
    y0, y1 = pad, max(pad + 1, height - pad)
    x0, x1 = pad, max(pad + 1, width - pad)
    base[y0:y1, x0:x1] = 1.0

    sigma = max(1.0, float(pad) * 0.55)
    blurred = gaussian_blur(base, sigma=sigma)
    max_val = float(np.max(blurred))
    if max_val > 1e-8:
        blurred /= max_val
    return np.clip(blurred, 0.0, 1.0).astype(np.float32, copy=False)


def resize_patch(patch: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    u8 = np.clip(patch * 255.0, 0.0, 255.0).astype(np.uint8)
    pil = Image.fromarray(u8, mode="RGB")
    resized = pil.resize((target_w, target_h), resample=Image.Resampling.BICUBIC)
    return (np.asarray(resized).astype(np.float32) / 255.0).astype(np.float32, copy=False)


def save_face_preview(root: Path, face_idx: int, label: str, patch_rgb_f32: np.ndarray) -> None:
    patch_u8 = np.clip(patch_rgb_f32 * 255.0, 0.0, 255.0).astype(np.uint8)
    Image.fromarray(patch_u8, mode="RGB").save(root / f"face_{face_idx:02d}_{label}.png")


def face_to_meta(face: FaceDetection) -> dict[str, Any]:
    return {
        "bbox": [int(face.x), int(face.y), int(face.w), int(face.h)],
        "confidence": float(face.confidence),
    }


def iou(a: FaceDetection, b: FaceDetection) -> float:
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


def sigmoid_confidence(weight: float) -> float:
    x = float(np.clip(weight, -16.0, 16.0))
    return float(1.0 / (1.0 + math.exp(-x)))


_EYE_DETECTOR_TRIED = False
_EYE_DETECTOR_WARNING: str | None = None
_EYE_DETECTOR_CASCADE: Any | None = None
_EYE_DETECTOR_CV2: Any | None = None


def eye_detector_available() -> tuple[bool, str | None]:
    global _EYE_DETECTOR_TRIED, _EYE_DETECTOR_WARNING, _EYE_DETECTOR_CASCADE, _EYE_DETECTOR_CV2
    if _EYE_DETECTOR_TRIED:
        return (_EYE_DETECTOR_CASCADE is not None and _EYE_DETECTOR_CV2 is not None), _EYE_DETECTOR_WARNING
    _EYE_DETECTOR_TRIED = True

    try:
        import cv2  # type: ignore
    except Exception as exc:  # noqa: BLE001 - optional dependency.
        _EYE_DETECTOR_WARNING = f"eye-detector-opencv-unavailable: {exc}"
        return False, _EYE_DETECTOR_WARNING

    candidates = [
        Path(cv2.data.haarcascades) / "haarcascade_eye_tree_eyeglasses.xml",
        Path(cv2.data.haarcascades) / "haarcascade_eye.xml",
    ]
    cascade_path = next((p for p in candidates if p.exists()), None)
    if cascade_path is None:
        _EYE_DETECTOR_WARNING = "eye-detector-haarcascade-not-found"
        return False, _EYE_DETECTOR_WARNING

    cascade = cv2.CascadeClassifier(str(cascade_path))
    if cascade.empty():
        _EYE_DETECTOR_WARNING = f"eye-detector-haarcascade-load-failed: {cascade_path}"
        return False, _EYE_DETECTOR_WARNING

    _EYE_DETECTOR_CV2 = cv2
    _EYE_DETECTOR_CASCADE = cascade
    _EYE_DETECTOR_WARNING = None
    return True, None


def detect_eye_count(face_patch_rgb_f32: np.ndarray) -> int:
    ok, _warning = eye_detector_available()
    if not ok or _EYE_DETECTOR_CASCADE is None:
        return 0

    gray = rgb_to_gray_u8(face_patch_rgb_f32)
    h, w = gray.shape
    if h < 16 or w < 16:
        return 0

    # Search in the upper face to reduce false detections from mouths/chins.
    upper = gray[: max(16, int(round(h * 0.68))), :]
    min_eye = max(8, int(round(min(h, w) * 0.11)))
    eyes = _EYE_DETECTOR_CASCADE.detectMultiScale(
        upper,
        scaleFactor=1.08,
        minNeighbors=3,
        minSize=(min_eye, min_eye),
    )
    if eyes is None:
        return 0
    return int(len(eyes))


def patch_detail_proxies(patch_rgb_f32: np.ndarray) -> tuple[float, float]:
    y = rgb_to_luma(np.clip(patch_rgb_f32, 0.0, 1.0).astype(np.float32, copy=False))
    if y.shape[0] < 3 or y.shape[1] < 3:
        return 0.0, 0.0
    gy, gx = np.gradient(y)
    gyy, _ = np.gradient(gy)
    _, gxx = np.gradient(gx)
    lap = gxx + gyy
    grad_mag = np.sqrt((gx * gx) + (gy * gy))
    return float(np.var(lap)), float(np.percentile(grad_mag, 90.0))


def rgb_to_gray_u8(rgb_f32: np.ndarray) -> np.ndarray:
    y = rgb_to_luma(np.clip(rgb_f32, 0.0, 1.0).astype(np.float32, copy=False))
    return np.clip(y * 255.0, 0.0, 255.0).astype(np.uint8)


def rgb_to_luma(rgb: np.ndarray) -> np.ndarray:
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]
    return ((0.2126 * r) + (0.7152 * g) + (0.0722 * b)).astype(np.float32, copy=False)


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


def rgb_f32_to_bgr_u8(rgb_f32: np.ndarray) -> np.ndarray:
    rgb_u8 = np.clip(rgb_f32 * 255.0, 0.0, 255.0).astype(np.uint8)
    return rgb_u8[:, :, ::-1]


def bgr_u8_to_rgb_f32(bgr_u8: np.ndarray) -> np.ndarray:
    rgb_u8 = bgr_u8[:, :, ::-1]
    return (rgb_u8.astype(np.float32) / 255.0).astype(np.float32, copy=False)


def get_module_version(module_name: str) -> str | None:
    try:
        import importlib.metadata as ilm

        return ilm.version(module_name)
    except Exception:  # noqa: BLE001
        return None


def resolve_gfpgan_model_candidates(config_model_path: Path | None) -> list[tuple[str, str]]:
    if config_model_path is not None:
        return [(str(config_model_path.expanduser()), "cli")]

    candidates: list[tuple[str, str]] = []
    seen: set[str] = set()
    for candidate in discover_local_gfpgan_model_candidates():
        if not candidate.exists():
            continue
        resolved = str(candidate.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        candidates.append((resolved, "auto-local"))

    if GFPGAN_DEFAULT_MODEL_URL not in seen:
        candidates.append((GFPGAN_DEFAULT_MODEL_URL, "auto-url"))
    return candidates


def discover_local_gfpgan_model_candidates() -> list[Path]:
    import importlib.util

    model_dirs: list[Path] = []
    project_root = Path(__file__).resolve().parent.parent
    model_dirs.append(project_root / "gfpgan" / "weights")
    model_dirs.append(Path.cwd() / "gfpgan" / "weights")
    model_dirs.append(Path.cwd() / "weights")

    try:
        spec = importlib.util.find_spec("gfpgan")
    except Exception:  # noqa: BLE001
        spec = None
    if spec is not None:
        if spec.origin:
            model_dirs.append(Path(spec.origin).resolve().parent / "weights")
        if spec.submodule_search_locations:
            for location in spec.submodule_search_locations:
                model_dirs.append(Path(location).resolve() / "weights")

    candidates: list[Path] = []
    seen: set[str] = set()
    for model_dir in model_dirs:
        for model_name in GFPGAN_MODEL_FILENAMES:
            candidate = model_dir / model_name
            key = str(candidate)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(candidate)
    return candidates


def is_url_source(value: str) -> bool:
    return value.startswith("https://") or value.startswith("http://")


def ensure_gfpgan_import_compatibility() -> None:
    try:
        import torchvision.transforms.functional_tensor  # type: ignore # noqa: F401

        return
    except Exception:
        pass

    try:
        import sys
        import types
        import torchvision.transforms.functional as functional  # type: ignore
    except Exception:
        return

    shim = types.ModuleType("torchvision.transforms.functional_tensor")
    for attr in ("rgb_to_grayscale",):
        if hasattr(functional, attr):
            setattr(shim, attr, getattr(functional, attr))
    sys.modules.setdefault("torchvision.transforms.functional_tensor", shim)
