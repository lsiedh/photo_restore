from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _chunk(tag: bytes, payload: bytes) -> bytes:
    body = tag + payload
    crc = zlib.crc32(body) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + body + struct.pack(">I", crc)


def write_rgb_u16_png(path: Path, array: np.ndarray, compression_level: int = 6) -> None:
    if array.dtype != np.uint16:
        raise ValueError("Expected uint16 RGB array.")
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("Expected shape (height, width, 3).")

    height, width, _ = array.shape
    path.parent.mkdir(parents=True, exist_ok=True)

    ihdr = struct.pack(">IIBBBBB", width, height, 16, 2, 0, 0, 0)
    compressor = zlib.compressobj(level=compression_level)
    row_bytes = array.astype(">u2", copy=False).view(np.uint8).reshape(height, width * 6)

    with path.open("wb") as handle:
        handle.write(PNG_SIGNATURE)
        handle.write(_chunk(b"IHDR", ihdr))

        compressed_buffer = bytearray()
        for row in row_bytes:
            compressed_buffer.extend(compressor.compress(b"\x00" + row.tobytes()))
            if len(compressed_buffer) >= 1024 * 1024:
                handle.write(_chunk(b"IDAT", bytes(compressed_buffer)))
                compressed_buffer.clear()

        compressed_buffer.extend(compressor.flush())
        if compressed_buffer:
            handle.write(_chunk(b"IDAT", bytes(compressed_buffer)))
        handle.write(_chunk(b"IEND", b""))
