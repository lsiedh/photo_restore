from __future__ import annotations

from pathlib import Path

import pytest

from restore_batch.cli import parse_args, should_write_sidecar
from restore_batch.openai_repair import DEFAULT_OPENAI_REPAIR_PROMPT_BW, DEFAULT_OPENAI_REPAIR_PROMPT_COLOR


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


def test_output_format_rejects_png_and_tiff() -> None:
    with pytest.raises(SystemExit):
        parse_args(["/tmp/in", "/tmp/out", "--output-format", "png"])
    with pytest.raises(SystemExit):
        parse_args(["/tmp/in", "/tmp/out", "--output-format", "tiff"])


def test_openai_repair_cli_defaults() -> None:
    args = parse_args(["/tmp/in", "/tmp/out"])
    assert args.openai_repair == "off"
    assert args.openai_repair_mask == "none"
    assert args.openai_repair_aggressiveness == "conservative"
    assert args.openai_repair_model == "gpt-image-1"
    assert args.openai_repair_quality == "medium"
    assert args.openai_repair_input_fidelity == "high"
    assert args.openai_repair_size == "auto"
    assert args.openai_repair_prompt_color == DEFAULT_OPENAI_REPAIR_PROMPT_COLOR
    assert args.openai_repair_prompt_bw == DEFAULT_OPENAI_REPAIR_PROMPT_BW
    assert args.openai_repair_prompt_override is None
    assert args.openai_repair_failure == "fail-closed"
    assert args.openai_repair_min_mask_px == 96
    assert args.openai_repair_min_mask_fraction == 0.0005
    assert args.openai_repair_max_mask_fraction == 0.20
    assert args.openai_repair_save_mask_preview is False


def test_openai_repair_cli_overrides() -> None:
    args = parse_args(
        [
            "/tmp/in",
            "/tmp/out",
            "--openai-repair",
            "on",
            "--openai-repair-mask",
            "none",
            "--openai-repair-aggressiveness",
            "aggressive",
            "--openai-repair-model",
            "gpt-image-1",
            "--openai-repair-quality",
            "high",
            "--openai-repair-input-fidelity",
            "low",
            "--openai-repair-size",
            "1536x1024",
            "--openai-repair-prompt-color",
            "color prompt",
            "--openai-repair-prompt-bw",
            "bw prompt",
            "--openai-repair-prompt-override",
            "override prompt",
            "--openai-repair-failure",
            "fail-closed",
            "--openai-repair-min-mask-px",
            "32",
            "--openai-repair-min-mask-fraction",
            "0.002",
            "--openai-repair-max-mask-fraction",
            "0.4",
            "--openai-repair-save-mask-preview",
        ]
    )
    assert args.openai_repair == "on"
    assert args.openai_repair_mask == "none"
    assert args.openai_repair_aggressiveness == "aggressive"
    assert args.openai_repair_model == "gpt-image-1"
    assert args.openai_repair_quality == "high"
    assert args.openai_repair_input_fidelity == "low"
    assert args.openai_repair_size == "1536x1024"
    assert args.openai_repair_prompt_color == "color prompt"
    assert args.openai_repair_prompt_bw == "bw prompt"
    assert args.openai_repair_prompt_override == "override prompt"
    assert args.openai_repair_failure == "fail-closed"
    assert args.openai_repair_min_mask_px == 32
    assert args.openai_repair_min_mask_fraction == 0.002
    assert args.openai_repair_max_mask_fraction == 0.4
    assert args.openai_repair_save_mask_preview is True


def test_openai_repair_cli_rejects_invalid_choices() -> None:
    with pytest.raises(SystemExit):
        parse_args(["/tmp/in", "/tmp/out", "--openai-repair", "maybe"])
    with pytest.raises(SystemExit):
        parse_args(["/tmp/in", "/tmp/out", "--openai-repair-mask", "manual"])
    with pytest.raises(SystemExit):
        parse_args(["/tmp/in", "/tmp/out", "--openai-repair-aggressiveness", "extreme"])
    with pytest.raises(SystemExit):
        parse_args(["/tmp/in", "/tmp/out", "--openai-repair-quality", "ultra"])
    with pytest.raises(SystemExit):
        parse_args(["/tmp/in", "/tmp/out", "--openai-repair-input-fidelity", "auto"])
    with pytest.raises(SystemExit):
        parse_args(["/tmp/in", "/tmp/out", "--openai-repair-size", "2048x2048"])
    with pytest.raises(SystemExit):
        parse_args(["/tmp/in", "/tmp/out", "--openai-repair-failure", "retry"])
