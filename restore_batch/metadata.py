from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class JsonlMetadataWriter:
    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, sort_keys=True, ensure_ascii=True)
        with self.output_path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.write("\n")


class SidecarMetadataWriter:
    def write(self, *, sidecar_path: Path, record: dict[str, Any]) -> None:
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        with sidecar_path.open("w", encoding="utf-8") as handle:
            json.dump(record, handle, sort_keys=True, ensure_ascii=True, indent=2)
            handle.write("\n")
