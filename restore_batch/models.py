from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class ImageContext:
    input_path: Path
    output_path: Path
    image_f32: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
