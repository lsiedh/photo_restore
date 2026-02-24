from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .classification import ClassificationConfig, ImageTypeClassificationStage
from .denoise import DenoiseConfig, DenoiseStage
from .discovery import discover_images
from .face_enhance import FaceEnhancementConfig, FaceEnhancementStage
from .flatfield import FlatFieldConfig, FlatFieldStage
from .jpeg_out import JpegExportConfig, JpegExportError, export_jpg_with_cap
from .metadata import JsonlMetadataWriter, SidecarMetadataWriter
from .models import ImageContext
from .normalize import NormalizationConfig, NormalizationStage
from .pipeline import ProcessingPipeline
from .redeye import RedEyeConfig, RedEyeStage
from .sharpen import SharpenConfig, SharpenStage
from .tonal import TonalNormalizationConfig, TonalNormalizationStage
from .white_balance import WhiteBalanceConfig, WhiteBalanceStage

LOGGER = logging.getLogger("restore_batch")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="restore-batch",
        description="Batch restoration pipeline for scanned photo images.",
    )
    parser.add_argument("input_dir", type=Path, help="Directory containing input images.")
    parser.add_argument("output_dir", type=Path, help="Directory where restored images will be written.")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan input subdirectories.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing normalized outputs.",
    )
    parser.add_argument(
        "--metadata-file",
        type=Path,
        default=None,
        help="Path for JSONL metadata log (default: <output_dir>/normalization_metadata.jsonl).",
    )
    parser.add_argument(
        "--alpha-background",
        choices=("white", "black"),
        default="white",
        help="Background used when flattening alpha channels.",
    )
    parser.add_argument(
        "--strict-icc",
        action="store_true",
        help="Fail an image when embedded ICC conversion fails instead of assuming sRGB.",
    )
    parser.add_argument(
        "--flatfield",
        choices=("on", "off"),
        default="on",
        help="Enable conservative global flat-field luminance normalization before white balance.",
    )
    parser.add_argument(
        "--flatfield-radius",
        type=str,
        default="320px",
        help="Flat-field blur radius in pixels (e.g. 320px) or percent of min dimension (e.g. 18%%).",
    )
    parser.add_argument(
        "--output-format",
        choices=("jpg",),
        default="jpg",
        help="Output format for restored images.",
    )
    parser.add_argument(
        "--jpg-max-mb",
        type=float,
        default=3.0,
        help="Maximum final JPG size in MiB per image.",
    )
    parser.add_argument(
        "--jpg-quality-max",
        type=int,
        default=92,
        help="Maximum JPG quality to try during size-cap search.",
    )
    parser.add_argument(
        "--jpg-quality-min",
        type=int,
        default=62,
        help="Minimum JPG quality to try before downscaling.",
    )
    parser.add_argument(
        "--jpg-quality-step",
        type=int,
        default=4,
        help="JPG quality decrement step during size-cap search.",
    )
    parser.add_argument(
        "--jpg-downscale-step",
        type=float,
        default=0.90,
        help="Per-iteration downscale factor when JPG is still over size cap.",
    )
    parser.add_argument(
        "--jpg-min-side",
        type=int,
        default=320,
        help="Minimum allowed image side while downscaling for JPG size cap.",
    )
    parser.add_argument(
        "--classification-threshold",
        type=float,
        default=0.0012,
        help="Threshold for near-grayscale vs true-color classification score.",
    )
    parser.add_argument(
        "--classification-sepia-gate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Treat monochrome-with-cast (sepia-like) scans as near-grayscale.",
    )
    parser.add_argument(
        "--classification-sepia-hue-var-max",
        type=float,
        default=0.01,
        help="Max weighted hue circular variance for sepia-like monochrome gate.",
    )
    parser.add_argument(
        "--classification-sepia-corr-min",
        type=float,
        default=0.995,
        help="Minimum luminance/channel correlation for sepia-like monochrome gate.",
    )
    parser.add_argument(
        "--classification-sepia-sat-mean-max",
        type=float,
        default=0.22,
        help="Maximum mean saturation for sepia-like monochrome gate.",
    )
    parser.add_argument(
        "--classification-sepia-sat-p90-max",
        type=float,
        default=0.35,
        help="Maximum saturation p90 for sepia-like monochrome gate.",
    )
    parser.add_argument(
        "--white-balance",
        choices=("none", "gray-world", "shades-of-gray", "consensus"),
        default="shades-of-gray",
        help="White balance method (applied only to true-color images).",
    )
    parser.add_argument(
        "--wb-strength",
        type=float,
        default=0.6,
        help="Blend strength for white balance correction (0..1 recommended).",
    )
    parser.add_argument(
        "--wb-max-gain",
        type=float,
        default=1.25,
        help="Maximum per-channel gain clamp for white balance.",
    )
    parser.add_argument(
        "--wb-shades-of-gray-p",
        type=float,
        default=6.0,
        help="Minkowski exponent for shades-of-gray method.",
    )
    parser.add_argument(
        "--wb-white-patch-percentile",
        type=float,
        default=None,
        help="Legacy compatibility alias: maps to robust fallback high-percentile exclusion when provided.",
    )
    parser.add_argument(
        "--wb-gray-edge-sigma",
        type=float,
        default=1.0,
        help="Gaussian pre-smoothing sigma used by robust gray-edge fallback estimation.",
    )
    parser.add_argument(
        "--wb-border-band",
        type=float,
        default=0.08,
        help="Border band fraction used for border-first white reference candidates.",
    )
    parser.add_argument(
        "--wb-border-texture-max",
        type=float,
        default=0.020,
        help="Maximum luminance-gradient texture allowed for border white-reference candidates.",
    )
    parser.add_argument(
        "--wb-border-neutrality-max",
        type=float,
        default=0.045,
        help="Maximum mean chroma magnitude allowed for border white-reference candidates.",
    )
    parser.add_argument(
        "--wb-border-clip-max",
        type=float,
        default=0.004,
        help="Maximum clipped-pixel fraction allowed for border white-reference candidates.",
    )
    parser.add_argument(
        "--wb-border-warm-bias-max",
        type=float,
        default=0.035,
        help="Maximum warm bias (R-B) allowed for border white-reference candidates.",
    )
    parser.add_argument(
        "--wb-min-valid-pixels",
        type=int,
        default=1024,
        help="Minimum valid pixels for robust WB illuminant estimation before fallback sampling.",
    )
    parser.add_argument(
        "--wb-confidence-reduce-threshold",
        type=float,
        default=0.45,
        help="WB confidence threshold below which correction strength is reduced.",
    )
    parser.add_argument(
        "--wb-confidence-skip-threshold",
        type=float,
        default=0.08,
        help="WB confidence threshold below which correction is skipped.",
    )
    parser.add_argument(
        "--chroma-bias-correction",
        choices=("on", "off"),
        default="on",
        help="Enable conservative low-frequency chroma bias removal after global white balance.",
    )
    parser.add_argument(
        "--chroma-bias-radius",
        type=str,
        default="16%",
        help="Chroma-bias blur radius in pixels or percent (e.g. 16%%).",
    )
    parser.add_argument(
        "--chroma-bias-cap",
        type=float,
        default=0.020,
        help="Maximum absolute chroma-bias subtraction per opponent channel.",
    )
    parser.add_argument(
        "--tonal-black-percentile",
        type=float,
        default=0.5,
        help="Black-point percentile for global tonal normalization.",
    )
    parser.add_argument(
        "--tonal-white-percentile",
        type=float,
        default=99.5,
        help="White-point percentile for global tonal normalization.",
    )
    parser.add_argument(
        "--tonal-strength",
        type=float,
        default=0.65,
        help="Global tonal normalization strength (0..1 recommended).",
    )
    parser.add_argument(
        "--tonal-contrast",
        type=float,
        default=0.25,
        help="Gentle global midtone contrast amount (0..1 recommended).",
    )
    parser.add_argument(
        "--tonal-near-gray-white-percentile",
        type=float,
        default=99.2,
        help="Near-grayscale-only white-point percentile for stronger B/W anchoring.",
    )
    parser.add_argument(
        "--tonal-near-gray-contrast",
        type=float,
        default=0.30,
        help="Near-grayscale-only midtone contrast amount.",
    )
    parser.add_argument(
        "--tonal-near-gray-min-strength",
        type=float,
        default=0.50,
        help="Minimum effective tonal strength floor for near-grayscale workflow.",
    )
    parser.add_argument(
        "--tonal-border-trim",
        type=float,
        default=0.04,
        help="Border trim fraction used for robust tonal stats.",
    )
    parser.add_argument(
        "--tonal-dynamic-range-good",
        type=float,
        default=0.82,
        help="Dynamic range threshold above which tonal strength is reduced.",
    )
    parser.add_argument(
        "--tonal-max-clip-fraction",
        type=float,
        default=0.001,
        help="Maximum tolerated pre-clip pixel fraction during tonal correction.",
    )
    parser.add_argument(
        "--denoise",
        choices=("none", "edge-aware-luma"),
        default="edge-aware-luma",
        help="Denoising method for Stage 4.",
    )
    parser.add_argument(
        "--denoise-strength",
        type=float,
        default=0.22,
        help="Base denoising strength (conservative defaults recommended).",
    )
    parser.add_argument(
        "--denoise-chroma-strength",
        type=float,
        default=0.08,
        help="Chroma denoising strength for color images.",
    )
    parser.add_argument(
        "--auto-strength",
        action="store_true",
        help="Automatically adjust denoise strength from a noise proxy.",
    )
    parser.add_argument(
        "--sharpen",
        choices=("none", "edge-aware-unsharp"),
        default="edge-aware-unsharp",
        help="Sharpening method for Stage 5.",
    )
    parser.add_argument(
        "--sharpen-amount",
        type=float,
        default=0.30,
        help="Global sharpening amount (conservative default for 300 dpi scans).",
    )
    parser.add_argument(
        "--sharpen-radius",
        type=float,
        default=1.20,
        help="Sharpening blur radius in pixels.",
    )
    parser.add_argument(
        "--sharpen-threshold",
        type=float,
        default=0.02,
        help="Detail threshold for halo-safe sharpening (luminance units).",
    )
    parser.add_argument(
        "--face-sharpen",
        action="store_true",
        help="Enable conservative face-local sharpening boost (optional).",
    )
    parser.add_argument(
        "--face-sharpen-boost",
        type=float,
        default=1.15,
        help="Face-local sharpening multiplier when face sharpening is enabled.",
    )
    parser.add_argument(
        "--redeye",
        choices=("off", "on"),
        default="off",
        help="Optional red-eye reduction stage (color images only).",
    )
    parser.add_argument(
        "--redeye-strength",
        type=float,
        default=0.70,
        help="Red-eye correction blend strength (0..1).",
    )
    parser.add_argument(
        "--redeye-red-ratio",
        type=float,
        default=1.65,
        help="Red dominance ratio threshold for red-eye masking.",
    )
    parser.add_argument(
        "--redeye-min-red",
        type=float,
        default=0.20,
        help="Minimum red channel level to consider red-eye candidates.",
    )
    parser.add_argument(
        "--redeye-min-red-excess",
        type=float,
        default=0.12,
        help="Minimum absolute red-over-nonred margin to qualify red-eye pixels.",
    )
    parser.add_argument(
        "--redeye-min-eye-px",
        type=int,
        default=12,
        help="Minimum eye box size (pixels) for red-eye processing.",
    )
    parser.add_argument(
        "--redeye-min-mask-px",
        type=int,
        default=8,
        help="Minimum red-eye mask pixels required before applying correction.",
    )
    parser.add_argument(
        "--redeye-max-mask-fraction",
        type=float,
        default=0.12,
        help="Maximum eye-region mask fraction to prevent overcorrection.",
    )
    parser.add_argument(
        "--redeye-feather-sigma",
        type=float,
        default=1.20,
        help="Mask feathering sigma for red-eye blending.",
    )
    parser.add_argument(
        "--redeye-darken-factor",
        type=float,
        default=0.55,
        help="Darkening factor applied to red channel in masked regions.",
    )
    parser.add_argument(
        "--face-enhance-backend",
        choices=("gfpgan", "codeformer"),
        default="gfpgan",
        help="Face enhancement backend for Stage 6.",
    )
    parser.add_argument(
        "--face-enhance-strength",
        type=float,
        default=0.35,
        help="Conservative face enhancement strength (0..1).",
    )
    parser.add_argument(
        "--face-codeformer-fidelity",
        type=float,
        default=0.70,
        help="CodeFormer fidelity parameter (used when backend is codeformer).",
    )
    parser.add_argument(
        "--face-model-path",
        type=Path,
        default=None,
        help="Path to the face enhancement model weights file (optional; auto-resolve/download is used when omitted).",
    )
    parser.add_argument(
        "--face-blend",
        type=float,
        default=0.60,
        help="Blend strength for enhanced face patch reintegration.",
    )
    parser.add_argument(
        "--face-feather",
        type=float,
        default=0.15,
        help="Feather amount for face blending mask.",
    )
    parser.add_argument(
        "--face-crop-expand",
        type=float,
        default=0.20,
        help="Relative margin used to expand detected face crops before enhancement.",
    )
    parser.add_argument(
        "--min-face-px",
        type=int,
        default=80,
        help="Minimum detected face width/height in pixels for enhancement.",
    )
    parser.add_argument(
        "--min-face-conf",
        type=float,
        default=0.60,
        help="Minimum face detector confidence for enhancement eligibility.",
    )
    parser.add_argument(
        "--face-overlap-iou",
        type=float,
        default=0.45,
        help="IoU threshold for overlap rejection between detected faces.",
    )
    parser.add_argument(
        "--face-min-change-apply",
        type=float,
        default=0.002,
        help="Minimum expected luminance-change metric required to apply enhancement.",
    )
    parser.add_argument(
        "--face-max-change",
        type=float,
        default=0.12,
        help="Maximum allowed luminance-change metric after clamping enhanced output.",
    )
    parser.add_argument(
        "--face-max-change-reject",
        type=float,
        default=0.22,
        help="Reject enhancement when estimated luminance-change metric exceeds this safety limit.",
    )
    parser.add_argument(
        "--face-match-local-luminance",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Match local luminance stats before blending enhanced faces.",
    )
    parser.add_argument(
        "--face-require-eye-evidence",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require eye evidence for smaller faces before enhancement to avoid hallucinated features.",
    )
    parser.add_argument(
        "--face-eye-check-max-px",
        type=int,
        default=10000,
        help="Apply eye-evidence gating only to faces up to this size (pixels).",
    )
    parser.add_argument(
        "--face-min-eye-count",
        type=int,
        default=1,
        help="Minimum detected eyes required when eye-evidence gating is active.",
    )
    parser.add_argument(
        "--face-small-eye-gate-px",
        type=int,
        default=145,
        help="Faces up to this size require stronger eye evidence to avoid fabricated features.",
    )
    parser.add_argument(
        "--face-small-min-eye-count",
        type=int,
        default=2,
        help="Minimum eye detections required for small-face eye gating.",
    )
    parser.add_argument(
        "--face-min-detail-lapvar",
        type=float,
        default=7.5e-05,
        help="Minimum face detail proxy (Laplacian variance) for enhancement eligibility.",
    )
    parser.add_argument(
        "--face-detail-check-max-px",
        type=int,
        default=150,
        help="Apply minimum-detail gating only to faces up to this size (pixels).",
    )
    parser.add_argument(
        "--save-face-previews",
        action="store_true",
        help="Save per-face before/enhanced/blended preview crops under output directory.",
    )
    parser.add_argument(
        "--debug-stats",
        action="store_true",
        help="Print per-image debug stats for classification, white balance, tonal, denoise, sharpening, and face enhancement tuning.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return parser.parse_args(argv)


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def alpha_background_rgb(name: str) -> tuple[int, int, int]:
    return (255, 255, 255) if name == "white" else (0, 0, 0)


def output_suffix(output_format: str) -> str:
    return ".jpg"


def sidecar_output_path(output_path: Path) -> Path:
    return output_path.with_suffix(f"{output_path.suffix}.json")


def should_write_sidecar(record: dict[str, Any]) -> bool:
    return str(record.get("status", "")).lower() == "ok"


def build_output_plan(
    *,
    input_paths: list[Path],
    input_root: Path,
    output_root: Path,
    output_format: str,
) -> tuple[dict[Path, Path], dict[Path, bool]]:
    stem_groups: dict[tuple[Path, str], list[Path]] = {}
    for path in input_paths:
        rel = path.relative_to(input_root)
        key = (rel.parent, rel.stem)
        stem_groups.setdefault(key, []).append(path)

    duplicate_keys = {key for key, grouped in stem_groups.items() if len(grouped) > 1}
    suffix = output_suffix(output_format)
    outputs: dict[Path, Path] = {}
    collision_resolved: dict[Path, bool] = {}

    for path in input_paths:
        rel = path.relative_to(input_root)
        key = (rel.parent, rel.stem)
        stem = rel.stem
        resolved = False
        if key in duplicate_keys:
            resolved = True
            source_ext = path.suffix.lower().lstrip(".") or "img"
            stem = f"{stem}__{source_ext}"
        outputs[path] = output_root / rel.parent / f"{stem}{suffix}"
        collision_resolved[path] = resolved

    return outputs, collision_resolved


def print_debug_stats(*, input_path: Path, metadata: dict[str, Any]) -> None:
    mean_rgb = metadata.get("stats_mean_rgb", [None, None, None])
    chroma_variance = metadata.get("classification_chroma_variance")
    image_type = metadata.get("image_type")
    wb_method = metadata.get("white_balance_method")
    wb_cli_mode = metadata.get("white_balance_cli_mode")
    wb_effective_mode = metadata.get("white_balance_effective_mode")
    wb_mode_mapped = metadata.get("white_balance_mode_mapped")
    wb_mode_mapping_reason = metadata.get("white_balance_mode_mapping_reason")
    wb_reference_method = metadata.get("white_balance_reference_method")
    wb_border_used = metadata.get("white_balance_border_used")
    wb_border_candidate_count = metadata.get("white_balance_border_candidate_count")
    wb_border_reject_reasons = metadata.get("white_balance_border_reject_reasons")
    estimated_illuminant = metadata.get("white_balance_estimated_illuminant", [None, None, None])
    applied_scales = metadata.get("white_balance_channel_scales_applied", [None, None, None])
    wb_transform_type = metadata.get("white_balance_global_transform_type")
    wb_transform_values = metadata.get("white_balance_global_transform_values")
    wb_confidence = metadata.get("white_balance_confidence")
    wb_strength_eff = metadata.get("white_balance_strength_effective")
    wb_sampling_mode = metadata.get("white_balance_sampling_mode")
    wb_percentile_low = metadata.get("white_balance_percentile_exclusion_low")
    wb_percentile_high = metadata.get("white_balance_percentile_exclusion_high")
    wb_clipped_exclusion = metadata.get("white_balance_clipped_exclusion_threshold")
    wb_consensus_agreement = metadata.get("white_balance_consensus_agreement")
    wb_estimator_weights = metadata.get("white_balance_estimator_weights")
    wb_guardrails = metadata.get("white_balance_guardrail_triggers")
    chroma_bias_applied = metadata.get("chroma_bias_correction_applied")
    chroma_bias_radius = metadata.get("chroma_bias_radius_spec")
    chroma_bias_cap = metadata.get("chroma_bias_cap")
    flatfield_applied = metadata.get("flatfield_applied")
    flatfield_radius = metadata.get("flatfield_radius_spec")
    black_p = metadata.get("tonal_black_percentile")
    white_p = metadata.get("tonal_white_percentile")
    near_gray_white_target = metadata.get("tonal_near_grayscale_white_percentile_target")
    near_gray_contrast_target = metadata.get("tonal_near_grayscale_contrast_target")
    near_gray_min_strength_target = metadata.get("tonal_near_grayscale_min_effective_strength_target")
    near_gray_strength_floor_applied = metadata.get("tonal_near_grayscale_strength_floor_applied")
    black_point = metadata.get("tonal_black_point")
    white_point = metadata.get("tonal_white_point")
    tonal_workflow = metadata.get("tonal_workflow")
    tonal_mapping = metadata.get("tonal_luminance_mapping")
    clip_low = metadata.get("tonal_clip_fraction_low_preclip")
    clip_high = metadata.get("tonal_clip_fraction_high_preclip")
    luma_before = metadata.get("tonal_luminance_mean_before")
    luma_after = metadata.get("tonal_luminance_mean_after")
    luma_p5_before = metadata.get("tonal_luminance_p5_before")
    luma_p5_after = metadata.get("tonal_luminance_p5_after")
    luma_p95_before = metadata.get("tonal_luminance_p95_before")
    luma_p95_after = metadata.get("tonal_luminance_p95_after")
    tonal_fallback = metadata.get("tonal_guardrail_fallback_used")
    tonal_fallback_mode = metadata.get("tonal_guardrail_fallback_mode")
    denoise_method = metadata.get("denoise_method")
    denoise_strength = metadata.get("denoise_strength_chosen")
    denoise_noise_before = metadata.get("denoise_noise_proxy_before")
    denoise_noise_after = metadata.get("denoise_noise_proxy_after")
    denoise_sharp_before = metadata.get("denoise_sharpness_proxy_before")
    denoise_sharp_after = metadata.get("denoise_sharpness_proxy_after")
    sharpen_method = metadata.get("sharpen_method")
    sharpen_amount = metadata.get("sharpen_amount")
    sharpen_radius = metadata.get("sharpen_radius")
    sharpen_threshold = metadata.get("sharpen_threshold")
    sharpen_sharp_before = metadata.get("sharpen_sharpness_proxy_before")
    sharpen_sharp_after = metadata.get("sharpen_sharpness_proxy_after")
    sharpen_faces = metadata.get("sharpen_faces_detected")
    redeye_method = metadata.get("redeye_method")
    redeye_applied = metadata.get("redeye_applied")
    redeye_skip_reason = metadata.get("redeye_skipped_reason")
    redeye_faces = metadata.get("redeye_faces_detected")
    redeye_eyes_detected = metadata.get("redeye_eyes_detected")
    redeye_eyes_processed = metadata.get("redeye_eyes_processed")
    redeye_eyes_skipped = metadata.get("redeye_eyes_skipped")
    redeye_pixels = metadata.get("redeye_pixels_corrected")
    face_enhance_backend = metadata.get("face_enhance_backend")
    face_enhance_applied = metadata.get("face_enhance_applied")
    face_enhance_skip_reason = metadata.get("face_enhance_skipped_reason")
    face_enhance_strength = metadata.get("face_enhance_strength")
    face_enhance_blend = metadata.get("face_enhance_blend")
    face_enhance_feather = metadata.get("face_enhance_feather")
    face_enhance_faces_detected = metadata.get("face_enhance_faces_detected")
    face_enhance_faces_processed = metadata.get("face_enhance_faces_processed")
    face_enhance_faces_skipped = metadata.get("face_enhance_faces_skipped")
    face_enhance_backend_warning = metadata.get("face_enhance_backend_warning")
    face_enhance_detector_warning = metadata.get("face_enhance_detector_warning")
    print(
        (
            f"[debug] {input_path.name} | "
            f"image_type={image_type} | "
            f"flatfield=(applied={flatfield_applied},radius={flatfield_radius}) | "
            f"wb_method={wb_method} | "
            f"wb_mode_map=(cli={wb_cli_mode},effective={wb_effective_mode},mapped={wb_mode_mapped},reason={wb_mode_mapping_reason}) | "
            f"wb_reference=(method={wb_reference_method},border_used={wb_border_used},border_candidates={wb_border_candidate_count},border_rejects={wb_border_reject_reasons}) | "
            f"mean_rgb={mean_rgb} | "
            f"chroma_variance={chroma_variance} | "
            f"estimated_illuminant={estimated_illuminant} | "
            f"applied_scaling={applied_scales} | "
            f"wb_transform=(type={wb_transform_type},values={wb_transform_values}) | "
            f"wb_confidence={wb_confidence} | "
            f"wb_strength_effective={wb_strength_eff} | "
            f"wb_sampling_mode={wb_sampling_mode} | "
            f"wb_fallback_mask=(p_low={wb_percentile_low},p_high={wb_percentile_high},clip_thr={wb_clipped_exclusion}) | "
            f"wb_consensus_agreement={wb_consensus_agreement} | "
            f"wb_estimator_weights={wb_estimator_weights} | "
            f"wb_guardrails={wb_guardrails} | "
            f"chroma_bias=(applied={chroma_bias_applied},radius={chroma_bias_radius},cap={chroma_bias_cap}) | "
            f"tonal_workflow={tonal_workflow} | "
            f"tonal_mapping={tonal_mapping} | "
            f"bw_percentiles=({black_p},{white_p}) | "
            f"near_gray_targets=(white_p={near_gray_white_target},contrast={near_gray_contrast_target},min_strength={near_gray_min_strength_target},floor_applied={near_gray_strength_floor_applied}) | "
            f"bw_points=({black_point},{white_point}) | "
            f"clip_fractions=(low={clip_low},high={clip_high}) | "
            f"luma_mean_before_after=({luma_before},{luma_after}) | "
            f"luma_p5_before_after=({luma_p5_before},{luma_p5_after}) | "
            f"luma_p95_before_after=({luma_p95_before},{luma_p95_after}) | "
            f"tonal_fallback=({tonal_fallback},{tonal_fallback_mode}) | "
            f"denoise_method={denoise_method} | "
            f"denoise_strength={denoise_strength} | "
            f"denoise_noise_before_after=({denoise_noise_before},{denoise_noise_after}) | "
            f"denoise_sharp_before_after=({denoise_sharp_before},{denoise_sharp_after}) | "
            f"sharpen_method={sharpen_method} | "
            f"sharpen_params=(amount={sharpen_amount},radius={sharpen_radius},threshold={sharpen_threshold}) | "
            f"sharpen_sharp_before_after=({sharpen_sharp_before},{sharpen_sharp_after}) | "
            f"sharpen_faces_detected={sharpen_faces} | "
            f"redeye=(method={redeye_method},applied={redeye_applied},skip={redeye_skip_reason},faces={redeye_faces},eyes={redeye_eyes_detected}/{redeye_eyes_processed}/{redeye_eyes_skipped},pixels={redeye_pixels}) | "
            f"face_enhance_backend={face_enhance_backend} | "
            f"face_enhance_applied={face_enhance_applied} | "
            f"face_enhance_skip_reason={face_enhance_skip_reason} | "
            f"face_enhance_params=(strength={face_enhance_strength},blend={face_enhance_blend},feather={face_enhance_feather}) | "
            f"face_enhance_faces=(detected={face_enhance_faces_detected},processed={face_enhance_faces_processed},skipped={face_enhance_faces_skipped}) | "
            f"face_enhance_warnings=(backend={face_enhance_backend_warning},detector={face_enhance_detector_warning})"
        ),
        flush=True,
    )


def process_batch(args: argparse.Namespace) -> int:
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    metadata_file = args.metadata_file.resolve() if args.metadata_file else (output_dir / "normalization_metadata.jsonl")

    if not input_dir.exists() or not input_dir.is_dir():
        LOGGER.error("Input directory does not exist or is not a directory: %s", input_dir)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_writer = JsonlMetadataWriter(metadata_file)
    sidecar_writer = SidecarMetadataWriter()

    normalization_stage = NormalizationStage(
        NormalizationConfig(
            alpha_background_rgb=alpha_background_rgb(args.alpha_background),
            strict_icc=args.strict_icc,
        )
    )
    flatfield_stage = FlatFieldStage(
        FlatFieldConfig(
            mode=args.flatfield,
            radius_spec=args.flatfield_radius,
        )
    )
    classification_stage = ImageTypeClassificationStage(
        ClassificationConfig(
            threshold=args.classification_threshold,
            sepia_gate_enabled=args.classification_sepia_gate,
            sepia_hue_circular_var_max=args.classification_sepia_hue_var_max,
            sepia_channel_corr_min=args.classification_sepia_corr_min,
            sepia_sat_mean_max=args.classification_sepia_sat_mean_max,
            sepia_sat_p90_max=args.classification_sepia_sat_p90_max,
        )
    )
    white_balance_stage = WhiteBalanceStage(
        WhiteBalanceConfig(
            method=args.white_balance,
            strength=args.wb_strength,
            max_gain=args.wb_max_gain,
            shades_of_gray_p=args.wb_shades_of_gray_p,
            gray_edge_sigma=args.wb_gray_edge_sigma,
            min_valid_pixels=args.wb_min_valid_pixels,
            confidence_reduce_threshold=args.wb_confidence_reduce_threshold,
            confidence_skip_threshold=args.wb_confidence_skip_threshold,
            border_band=args.wb_border_band,
            border_texture_max=args.wb_border_texture_max,
            border_neutrality_max=args.wb_border_neutrality_max,
            border_clip_max=args.wb_border_clip_max,
            border_warm_bias_max=args.wb_border_warm_bias_max,
            fallback_percentile_high=(98.0 if args.wb_white_patch_percentile is None else args.wb_white_patch_percentile),
            chroma_bias_correction=args.chroma_bias_correction,
            chroma_bias_radius_spec=args.chroma_bias_radius,
            chroma_bias_cap=args.chroma_bias_cap,
        )
    )
    tonal_stage = TonalNormalizationStage(
        TonalNormalizationConfig(
            black_percentile=args.tonal_black_percentile,
            white_percentile=args.tonal_white_percentile,
            border_trim_fraction=args.tonal_border_trim,
            strength=args.tonal_strength,
            contrast_strength=args.tonal_contrast,
            near_grayscale_contrast_strength=args.tonal_near_gray_contrast,
            near_grayscale_white_percentile=args.tonal_near_gray_white_percentile,
            near_grayscale_min_effective_strength=args.tonal_near_gray_min_strength,
            dynamic_range_good=args.tonal_dynamic_range_good,
            clip_fraction_threshold=args.tonal_max_clip_fraction,
        )
    )
    denoise_stage = DenoiseStage(
        DenoiseConfig(
            method=args.denoise,
            strength=args.denoise_strength,
            chroma_strength=args.denoise_chroma_strength,
            auto_strength=args.auto_strength,
        )
    )
    sharpen_stage = SharpenStage(
        SharpenConfig(
            method=args.sharpen,
            amount=args.sharpen_amount,
            radius=args.sharpen_radius,
            threshold=args.sharpen_threshold,
            face_sharpen_enabled=args.face_sharpen,
            face_boost=args.face_sharpen_boost,
        )
    )
    redeye_stage = RedEyeStage(
        RedEyeConfig(
            mode=args.redeye,
            strength=args.redeye_strength,
            red_ratio=args.redeye_red_ratio,
            min_red=args.redeye_min_red,
            min_red_excess=args.redeye_min_red_excess,
            min_eye_px=args.redeye_min_eye_px,
            min_mask_px=args.redeye_min_mask_px,
            max_mask_fraction=args.redeye_max_mask_fraction,
            feather_sigma=args.redeye_feather_sigma,
            darken_factor=args.redeye_darken_factor,
        )
    )
    face_model_path = args.face_model_path.resolve() if args.face_model_path is not None else None
    face_enhance_stage = FaceEnhancementStage(
        FaceEnhancementConfig(
            mode="on",
            backend=args.face_enhance_backend,
            strength=args.face_enhance_strength,
            codeformer_fidelity=args.face_codeformer_fidelity,
            blend=args.face_blend,
            feather=args.face_feather,
            crop_expand=args.face_crop_expand,
            min_face_px=args.min_face_px,
            min_face_conf=args.min_face_conf,
            overlap_iou_threshold=args.face_overlap_iou,
            min_luma_change_apply=args.face_min_change_apply,
            max_luma_change=args.face_max_change,
            max_luma_change_reject=args.face_max_change_reject,
            match_local_luminance=args.face_match_local_luminance,
            require_eye_evidence=args.face_require_eye_evidence,
            eye_check_max_face_px=args.face_eye_check_max_px,
            min_eye_count=args.face_min_eye_count,
            small_face_eye_gate_px=args.face_small_eye_gate_px,
            small_face_min_eye_count=args.face_small_min_eye_count,
            min_detail_lap_var=args.face_min_detail_lapvar,
            detail_check_max_face_px=args.face_detail_check_max_px,
            save_face_previews=args.save_face_previews,
            model_path=face_model_path,
        )
    )
    stages = [
        normalization_stage,
        flatfield_stage,
        classification_stage,
        white_balance_stage,
        tonal_stage,
        denoise_stage,
        sharpen_stage,
        redeye_stage,
        face_enhance_stage,
    ]
    stage_names = [stage.name for stage in stages]
    pipeline = ProcessingPipeline(stages)
    images = discover_images(input_dir, recursive=args.recursive)
    output_paths, collision_flags = build_output_plan(
        input_paths=images,
        input_root=input_dir,
        output_root=output_dir,
        output_format=args.output_format,
    )
    LOGGER.info("Discovered %d input images in %s", len(images), input_dir)

    success_count = 0
    skipped_count = 0
    error_count = 0
    face_sharpen_disabled_count = 0
    face_sharpen_disabled_reason: str | None = None
    face_enhance_processed_image_count = 0
    face_enhance_processed_faces_total = 0
    face_enhance_skipped_faces_total = 0
    face_enhance_backend_unavailable_count = 0
    face_enhance_detector_unavailable_count = 0
    face_enhance_all_skipped_count = 0
    face_enhance_no_faces_count = 0
    redeye_applied_image_count = 0
    redeye_faces_detected_total = 0
    redeye_eyes_detected_total = 0
    redeye_eyes_processed_total = 0
    redeye_eyes_skipped_total = 0
    jpg_export_config = JpegExportConfig(
        max_mb=args.jpg_max_mb,
        quality_max=args.jpg_quality_max,
        quality_min=args.jpg_quality_min,
        quality_step=args.jpg_quality_step,
        downscale_step=args.jpg_downscale_step,
        min_side=args.jpg_min_side,
    )

    for input_path in images:
        started = time.perf_counter()
        output_path = output_paths[input_path]
        record: dict[str, Any] = {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "input_path": str(input_path),
            "output_path": str(output_path),
            "stage": "restore-pipeline",
            "pipeline_stages": stage_names,
            "output_name_collision_resolved": collision_flags[input_path],
        }
        context: ImageContext | None = None

        try:
            if output_path.exists() and not args.overwrite:
                skipped_count += 1
                record.update({"status": "skipped", "reason": "output-exists"})
                LOGGER.info("Skipping existing output: %s", output_path)
                metadata_writer.write(record)
                continue

            context = ImageContext(input_path=input_path, output_path=output_path)
            context = pipeline.run(context)
            if context.image_f32 is None:
                raise RuntimeError("Normalization stage did not produce image data.")

            conversion_meta = export_jpg_with_cap(
                image_srgb_f32=context.image_f32,
                output_path=output_path,
                config=jpg_export_config,
            )
            jpg_dims = conversion_meta.get("jpg_export_dims")
            if isinstance(jpg_dims, list) and len(jpg_dims) == 2:
                output_size = [int(jpg_dims[0]), int(jpg_dims[1])]
            else:
                output_size = [int(context.image_f32.shape[1]), int(context.image_f32.shape[0])]

            elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
            record.update(context.metadata)
            record.update(conversion_meta)
            record.update(
                {
                    "status": "ok",
                    "elapsed_ms": elapsed_ms,
                    "output_size": output_size,
                }
            )
            metadata_writer.write(record)
            if should_write_sidecar(record):
                sidecar_writer.write(sidecar_path=sidecar_output_path(output_path), record=record)
            if args.debug_stats:
                print_debug_stats(input_path=input_path, metadata=record)
            if args.face_sharpen and record.get("sharpen_face_warning"):
                face_sharpen_disabled_count += 1
                face_sharpen_disabled_reason = str(record.get("sharpen_face_warning"))
            face_enhance_processed_faces = int(record.get("face_enhance_faces_processed", 0))
            face_enhance_skipped_faces = int(record.get("face_enhance_faces_skipped", 0))
            face_enhance_processed_faces_total += face_enhance_processed_faces
            face_enhance_skipped_faces_total += face_enhance_skipped_faces
            if face_enhance_processed_faces > 0:
                face_enhance_processed_image_count += 1
            redeye_faces_detected_total += int(record.get("redeye_faces_detected", 0))
            redeye_eyes_detected_total += int(record.get("redeye_eyes_detected", 0))
            redeye_eyes_processed_total += int(record.get("redeye_eyes_processed", 0))
            redeye_eyes_skipped_total += int(record.get("redeye_eyes_skipped", 0))
            if bool(record.get("redeye_applied")):
                redeye_applied_image_count += 1

            face_enhance_skip_reason = str(record.get("face_enhance_skipped_reason", ""))
            if face_enhance_skip_reason == "backend-unavailable":
                face_enhance_backend_unavailable_count += 1
            elif face_enhance_skip_reason == "detector-unavailable":
                face_enhance_detector_unavailable_count += 1
            elif face_enhance_skip_reason == "all-faces-skipped":
                face_enhance_all_skipped_count += 1
            elif face_enhance_skip_reason == "no-faces-detected":
                face_enhance_no_faces_count += 1
            success_count += 1
            LOGGER.info("Processed %s -> %s", input_path.name, output_path)
        except JpegExportError as exc:
            error_count += 1
            elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
            if context is not None:
                record.update(context.metadata)
            record.update(exc.metadata)
            record.update({"status": "error", "error": str(exc), "elapsed_ms": elapsed_ms})
            metadata_writer.write(record)
            LOGGER.error("Failed JPG export for image: %s (%s)", input_path, exc)
        except Exception as exc:  # noqa: BLE001 - batch should continue after failures.
            error_count += 1
            elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
            record.update({"status": "error", "error": str(exc), "elapsed_ms": elapsed_ms})
            metadata_writer.write(record)
            LOGGER.exception("Failed to process image: %s", input_path)

    LOGGER.info(
        "Completed pipeline. ok=%d skipped=%d errors=%d metadata=%s",
        success_count,
        skipped_count,
        error_count,
        metadata_file,
    )
    if args.face_sharpen and face_sharpen_disabled_reason:
        LOGGER.warning(
            "Face sharpening summary: disabled on %d/%d processed images (%s).",
            face_sharpen_disabled_count,
            success_count,
            face_sharpen_disabled_reason,
        )
    LOGGER.info(
        "Red-eye summary: applied_images=%d/%d faces_detected=%d eyes_detected=%d eyes_processed=%d eyes_skipped=%d",
        redeye_applied_image_count,
        success_count,
        redeye_faces_detected_total,
        redeye_eyes_detected_total,
        redeye_eyes_processed_total,
        redeye_eyes_skipped_total,
    )
    LOGGER.info(
        "Face enhancement summary: processed_images=%d/%d processed_faces=%d skipped_faces=%d no_faces_detected=%d all_faces_skipped=%d backend_unavailable=%d detector_unavailable=%d",
        face_enhance_processed_image_count,
        success_count,
        face_enhance_processed_faces_total,
        face_enhance_skipped_faces_total,
        face_enhance_no_faces_count,
        face_enhance_all_skipped_count,
        face_enhance_backend_unavailable_count,
        face_enhance_detector_unavailable_count,
    )
    if success_count > 0 and face_enhance_backend_unavailable_count == success_count:
        LOGGER.warning(
            "Face enhancement was disabled for the full run because backend '%s' was unavailable. Install backend dependencies, ensure network/model availability, or set --face-model-path.",
            args.face_enhance_backend,
        )
    if success_count > 0 and face_enhance_detector_unavailable_count == success_count:
        LOGGER.warning(
            "Face enhancement could not run in this batch because the face detector was unavailable.",
        )
    return 1 if error_count else 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level)
    return process_batch(args)


if __name__ == "__main__":
    sys.exit(main())
