from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from inferforge.merger.core.gguf_reader import GGML_F32, NativeGGUFReader, load_gguf_native


def _write_f32_gguf(path: Path, name: str, matrix: np.ndarray) -> None:
    rows, cols = matrix.shape
    payload = np.ascontiguousarray(matrix.astype(np.float32, copy=False)).tobytes()
    with path.open("wb") as handle:
        handle.write(b"GGUF")
        handle.write(struct.pack("<I", 3))
        handle.write(struct.pack("<Q", 1))
        handle.write(struct.pack("<Q", 1))
        key = b"general.architecture"
        handle.write(struct.pack("<Q", len(key)))
        handle.write(key)
        handle.write(struct.pack("<I", 8))
        value = b"llama"
        handle.write(struct.pack("<Q", len(value)))
        handle.write(value)
        tname = name.encode("utf-8")
        handle.write(struct.pack("<Q", len(tname)))
        handle.write(tname)
        handle.write(struct.pack("<I", 2))
        handle.write(struct.pack("<Q", cols))
        handle.write(struct.pack("<Q", rows))
        handle.write(struct.pack("<I", GGML_F32))
        handle.write(struct.pack("<Q", 0))
        pos = handle.tell()
        align = 32
        pad = 0 if pos % align == 0 else align - (pos % align)
        handle.write(b"\x00" * pad)
        handle.write(payload)


def test_native_gguf_reader_loads_f32(tmp_path: Path):
    path = tmp_path / "tiny.gguf"
    matrix = np.arange(12, dtype=np.float32).reshape(3, 4)
    _write_f32_gguf(path, "token_embd.weight", matrix)
    weights, arch, _tok = load_gguf_native(path)
    assert "token_embd.weight" in weights
    loaded = weights["token_embd.weight"]
    assert loaded.shape == (3, 4)
    assert np.allclose(loaded, matrix)
    assert arch.get("architecture") == "llama"


def test_native_gguf_reader_context_manager(tmp_path: Path):
    path = tmp_path / "tiny.gguf"
    matrix = np.ones((2, 2), dtype=np.float32)
    _write_f32_gguf(path, "output.weight", matrix)
    with NativeGGUFReader(path) as reader:
        loaded = reader.load_weights()
    assert "output.weight" in loaded
    assert np.allclose(loaded["output.weight"], matrix)
