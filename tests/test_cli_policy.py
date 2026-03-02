from __future__ import annotations

from pathlib import Path

import pytest

from restore_batch.cli import parse_args, should_write_sidecar


def test_should_write_sidecar_only_for_ok_records() -> None:
    assert should_write_sidecar({"status": "ok"}) is True
    assert should_write_sidecar({"status": "OK"}) is True
    assert should_write_sidecar({"status": "skipped"}) is False
    assert should_write_sidecar({"status": "error"}) is False
    assert should_write_sidecar({}) is False


def test_wb_white_patch_percentile_cli_alias_parses_and_defaults_to_none() -> None:
    args_default = parse_args(["/tmp/in", "/tmp/out"])
    assert isinstance(args_default.input_dir, Path)
    assert isinstance(args_default.output_dir, Path)
    assert args_default.wb_white_patch_percentile is None
    assert args_default.output_format == "jpg"

    args_override = parse_args(["/tmp/in", "/tmp/out", "--wb-white-patch-percentile", "97.5"])
    assert args_override.wb_white_patch_percentile == 97.5


def test_wb_cast_and_skin_saturation_cli_defaults() -> None:
    args = parse_args(["/tmp/in", "/tmp/out"])
    assert args.wb_cast_removal_mode == "conservative"
    assert args.wb_skin_saturation_auto == "on"
    assert args.wb_skin_sat_target_low == 0.16
    assert args.wb_skin_sat_target_high == 0.48
    assert args.wb_skin_sat_adjust_limit == 0.20


def test_output_format_rejects_png_and_tiff() -> None:
    with pytest.raises(SystemExit):
        parse_args(["/tmp/in", "/tmp/out", "--output-format", "png"])
    with pytest.raises(SystemExit):
        parse_args(["/tmp/in", "/tmp/out", "--output-format", "tiff"])


def test_dust_cleanup_cli_defaults() -> None:
    args = parse_args(["/tmp/in", "/tmp/out"])
    assert args.dust_clean == "on"
    assert args.dust_response_sigma == 1.15
    assert args.dust_response_sigma_wide == 2.20
    assert args.dust_mad_multiplier == 5.0
    assert args.dust_min_contrast == 0.020
    assert args.dust_texture_percentile == 62.0
    assert args.dust_min_component_px == 3
    assert args.dust_max_component_px == 220
    assert args.dust_max_component_aspect == 3.5
    assert args.dust_min_component_fill == 0.08
    assert args.dust_max_mask_fraction == 0.015
    assert args.dust_inpaint_radius == 2.20
    assert args.dust_save_mask_preview is False


def test_dust_cleanup_cli_rejects_invalid_mode() -> None:
    with pytest.raises(SystemExit):
        parse_args(["/tmp/in", "/tmp/out", "--dust-clean", "maybe"])


def test_openai_repair_flags_are_rejected() -> None:
    with pytest.raises(SystemExit):
        parse_args(["/tmp/in", "/tmp/out", "--openai-repair", "on"])
