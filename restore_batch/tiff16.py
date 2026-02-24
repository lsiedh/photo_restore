from __future__ import annotations

import struct
from pathlib import Path

import numpy as np


def encode_rgb_u16_tiff_bytes(array: np.ndarray) -> bytes:
    if array.dtype != np.uint16:
        raise ValueError("Expected uint16 RGB array.")
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("Expected shape (height, width, 3).")

    height, width, _ = array.shape
    payload = np.ascontiguousarray(array.astype("<u2", copy=False)).tobytes()
    strip_byte_count = len(payload)

    # Baseline TIFF tags for uncompressed chunky RGB.
    entries: list[tuple[int, int, int, int]] = []
    entry_count = 10
    ifd_offset = 8
    ifd_size = 2 + (entry_count * 12) + 4
    bits_per_sample_offset = ifd_offset + ifd_size
    bits_per_sample_data = struct.pack("<HHH", 16, 16, 16)
    strip_offset = bits_per_sample_offset + len(bits_per_sample_data)

    entries.append((256, 4, 1, width))  # ImageWidth (LONG)
    entries.append((257, 4, 1, height))  # ImageLength (LONG)
    entries.append((258, 3, 3, bits_per_sample_offset))  # BitsPerSample (SHORT x3)
    entries.append((259, 3, 1, 1))  # Compression = none
    entries.append((262, 3, 1, 2))  # PhotometricInterpretation = RGB
    entries.append((273, 4, 1, strip_offset))  # StripOffsets
    entries.append((277, 3, 1, 3))  # SamplesPerPixel = 3
    entries.append((278, 4, 1, height))  # RowsPerStrip
    entries.append((279, 4, 1, strip_byte_count))  # StripByteCounts
    entries.append((284, 3, 1, 1))  # PlanarConfiguration = chunky

    out = bytearray()
    out.extend(b"II")  # Little endian
    out.extend(struct.pack("<H", 42))
    out.extend(struct.pack("<I", ifd_offset))

    out.extend(struct.pack("<H", entry_count))
    for tag, typ, count, value in entries:
        out.extend(struct.pack("<HHII", tag, typ, count, value))
    out.extend(struct.pack("<I", 0))  # next IFD offset

    out.extend(bits_per_sample_data)
    out.extend(payload)
    return bytes(out)


def write_rgb_u16_tiff(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = encode_rgb_u16_tiff_bytes(array)
    with path.open("wb") as handle:
        handle.write(payload)
