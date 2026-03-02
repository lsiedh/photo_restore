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


def test_output_format_rejects_png_and_tiff() -> None:
    with pytest.raises(SystemExit):
        parse_args(["/tmp/in", "/tmp/out", "--output-format", "png"])
    with pytest.raises(SystemExit):
        parse_args(["/tmp/in", "/tmp/out", "--output-format", "tiff"])


def test_openai_repair_flags_are_rejected() -> None:
    with pytest.raises(SystemExit):
        parse_args(["/tmp/in", "/tmp/out", "--openai-repair", "on"])
