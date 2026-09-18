from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

ProgressFn = Callable[[str, int, int], None]

GGUF_MAGIC = b"GGUF"
QK_K = 256
GGML_F32 = 0
GGML_F16 = 1
GGML_Q4_0 = 2
GGML_Q4_1 = 3
GGML_Q5_0 = 6
GGML_Q5_1 = 7
GGML_Q8_0 = 8
GGML_Q8_1 = 9
GGML_Q2_K = 10
GGML_Q3_K = 11
GGML_Q4_K = 12
GGML_Q5_K = 13
GGML_Q6_K = 14
GGML_Q8_K = 15
GGML_BF16 = 30

VALUE_UINT8, VALUE_INT8, VALUE_UINT16, VALUE_INT16 = 0, 1, 2, 3
VALUE_UINT32, VALUE_INT32, VALUE_FLOAT32, VALUE_BOOL = 4, 5, 6, 7
VALUE_STRING, VALUE_ARRAY, VALUE_UINT64, VALUE_INT64, VALUE_FLOAT64 = 8, 9, 10, 11, 12

BLOCK_BYTES = {
    GGML_F32: 4, GGML_F16: 2, GGML_BF16: 2,
    GGML_Q4_0: 18, GGML_Q4_1: 20, GGML_Q5_0: 22, GGML_Q5_1: 24,
    GGML_Q8_0: 34, GGML_Q8_1: 36, GGML_Q2_K: 84, GGML_Q3_K: 110,
    GGML_Q4_K: 144, GGML_Q5_K: 176, GGML_Q6_K: 210, GGML_Q8_K: 292,
}
BLOCK_WEIGHTS = {
    GGML_F32: 1, GGML_F16: 1, GGML_BF16: 1,
    GGML_Q4_0: 32, GGML_Q4_1: 32, GGML_Q5_0: 32, GGML_Q5_1: 32,
    GGML_Q8_0: 32, GGML_Q8_1: 32, GGML_Q2_K: QK_K, GGML_Q3_K: QK_K,
    GGML_Q4_K: QK_K, GGML_Q5_K: QK_K, GGML_Q6_K: QK_K, GGML_Q8_K: QK_K,
}


def _fp16_to_f32(raw: np.ndarray) -> np.ndarray:
    return raw.view(np.float16).astype(np.float32)


def _bf16_to_f32(raw: np.ndarray) -> np.ndarray:
    u16 = raw.view(np.uint16).astype(np.uint32)
    return (u16 << 16).view(np.float32)


@dataclass
class GGUFTensorInfo:
    name: str
    dims: tuple[int, ...]
    ggml_type: int
    offset: int


@dataclass
class GGUFFile:
    path: Path
    version: int
    alignment: int
    metadata: dict[str, Any] = field(default_factory=dict)
    tensors: list[GGUFTensorInfo] = field(default_factory=list)
    data_offset: int = 0


class NativeGGUFReader:
    def __init__(self, path: str | Path, use_mmap: bool = True):
        self.path = Path(path)
        self._fh = self.path.open("rb")
        self._mmap = None
        if use_mmap:
            try:
                import mmap
                self._mmap = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)
            except Exception:
                self._mmap = None
        self.file = self._parse_header()
        self._tensor_map: dict[str, GGUFTensorInfo] = {t.name: t for t in self.file.tensors}

    def close(self) -> None:
        if self._mmap is not None:
            try:
                self._mmap.close()
            except Exception:
                pass
            self._mmap = None
        if self._fh and not self._fh.closed:
            self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def get_tensor_names(self) -> list[str]:
        return [t.name for t in self.file.tensors]

    def get_tensor_info(self, name: str) -> GGUFTensorInfo | None:
        return self._tensor_map.get(name)

    def has_tensor(self, name: str) -> bool:
        return name in self._tensor_map

    def load_tensor_by_name(self, name: str) -> np.ndarray:
        info = self._tensor_map.get(name)
        if info is None:
            raise KeyError(f"Tensor '{name}' not found in GGUF file: {self.path}")
        return self._load_tensor(info)

    def iter_tensors(self, progress: ProgressFn | None = None):
        total = max(len(self.file.tensors), 1)
        for index, info in enumerate(self.file.tensors):
            if progress:
                progress(info.name, index, total)
            yield info.name, self._load_tensor(info)

    def _read(self, fmt: str):
        size = struct.calcsize(fmt)
        data = self._fh.read(size)
        if len(data) != size:
            raise ValueError(f"Truncated GGUF file: {self.path}")
        values = struct.unpack("<" + fmt, data)
        return values[0] if len(values) == 1 else values

    def _read_string(self) -> str:
        length = self._read("Q")
        return self._fh.read(length).decode("utf-8", errors="replace")

    def _read_value(self, vtype: int) -> Any:
        readers = {
            VALUE_UINT8: lambda: self._read("B"),
            VALUE_INT8: lambda: self._read("b"),
            VALUE_UINT16: lambda: self._read("H"),
            VALUE_INT16: lambda: self._read("h"),
            VALUE_UINT32: lambda: self._read("I"),
            VALUE_INT32: lambda: self._read("i"),
            VALUE_FLOAT32: lambda: self._read("f"),
            VALUE_BOOL: lambda: bool(self._read("B")),
            VALUE_STRING: self._read_string,
            VALUE_UINT64: lambda: self._read("Q"),
            VALUE_INT64: lambda: self._read("q"),
            VALUE_FLOAT64: lambda: self._read("d"),
        }
        if vtype == VALUE_ARRAY:
            item_type = self._read("I")
            count = self._read("Q")
            return [self._read_value(item_type) for _ in range(count)]
        if vtype not in readers:
            raise ValueError(f"Unknown GGUF value type {vtype}")
        return readers[vtype]()

    def _parse_header(self) -> GGUFFile:
        if self._fh.read(4) != GGUF_MAGIC:
            raise ValueError(f"Not a GGUF file: {self.path}")
        version = self._read("I")
        tensor_count = self._read("Q")
        kv_count = self._read("Q")
        metadata: dict[str, Any] = {}
        for _ in range(kv_count):
            key = self._read_string()
            metadata[key] = self._read_value(self._read("I"))
        tensors: list[GGUFTensorInfo] = []
        for _ in range(tensor_count):
            name = self._read_string()
            n_dims = self._read("I")
            dims = tuple(int(self._read("Q")) for _ in range(n_dims))
            ggml_type = self._read("I")
            offset = self._read("Q")
            tensors.append(GGUFTensorInfo(name=name, dims=dims, ggml_type=ggml_type, offset=offset))
        alignment = int(metadata.get("general.alignment") or 32)
        pos = self._fh.tell()
        data_offset = pos if pos % alignment == 0 else pos + (alignment - pos % alignment)
        return GGUFFile(self.path, version, alignment, metadata, tensors, data_offset)

    def load_weights(self, progress: ProgressFn | None = None) -> dict[str, np.ndarray]:
        weights: dict[str, np.ndarray] = {}
        total = max(len(self.file.tensors), 1)
        for index, info in enumerate(self.file.tensors):
            if progress:
                progress(info.name, index, total)
            weights[info.name] = self._load_tensor(info)
        if progress:
            progress("gguf", total, total)
        return weights

    def _load_tensor(self, info: GGUFTensorInfo) -> np.ndarray:
        n_weights = int(np.prod(info.dims)) if info.dims else 0
        if info.ggml_type not in BLOCK_BYTES:
            raise ValueError(f"Unsupported GGUF tensor type {info.ggml_type} for {info.name}")
        block_w = BLOCK_WEIGHTS[info.ggml_type]
        n_blocks = n_weights if block_w == 1 else (n_weights + block_w - 1) // block_w
        nbytes = n_blocks * BLOCK_BYTES[info.ggml_type]
        abs_offset = self.file.data_offset + info.offset
        if self._mmap is not None:
            raw = np.frombuffer(self._mmap, dtype=np.uint8, count=nbytes, offset=abs_offset)
        else:
            self._fh.seek(abs_offset)
            raw = np.frombuffer(self._fh.read(nbytes), dtype=np.uint8)
        data = dequantize(raw, info.ggml_type, n_weights)
        if info.dims:
            data = data.reshape(tuple(reversed(info.dims)))
        return np.ascontiguousarray(data)

    def tokenizer_payload(self) -> dict[str, Any] | None:
        meta = self.file.metadata
        tokens = meta.get("tokenizer.ggml.tokens")
        if not isinstance(tokens, list) or not tokens:
            return None
        vocab = {str(token): i for i, token in enumerate(tokens)}
        merges = meta.get("tokenizer.ggml.merges") or []
        bos = meta.get("tokenizer.ggml.bos_token_id")
        eos = meta.get("tokenizer.ggml.eos_token_id")
        unk = meta.get("tokenizer.ggml.unknown_token_id")
        pad = meta.get("tokenizer.ggml.padding_token_id")
        specials = {
            "bos_token": tokens[bos] if isinstance(bos, int) and 0 <= bos < len(tokens) else None,
            "eos_token": tokens[eos] if isinstance(eos, int) and 0 <= eos < len(tokens) else None,
            "unk_token": tokens[unk] if isinstance(unk, int) and 0 <= unk < len(tokens) else None,
            "pad_token": tokens[pad] if isinstance(pad, int) and 0 <= pad < len(tokens) else None,
        }
        return {
            "tokenizer.json": {"version": "1.0", "model": {"type": "BPE", "vocab": vocab, "merges": merges}, "added_tokens": []},
            "tokenizer_config.json": {
                "vocab_size": len(tokens),
                "model_max_length": int(meta.get("llama.context_length") or 4096),
                **{k: v for k, v in specials.items() if v is not None},
            },
            "special_tokens_map.json": {k: v for k, v in specials.items() if v is not None},
            "vocab": vocab,
            "merges.txt": "\n".join(str(item) for item in merges) if merges else None,
        }

    def architecture_fields(self) -> dict[str, Any]:
        meta = self.file.metadata
        arch = str(meta.get("general.architecture") or "llama")
        def pick(*keys):
            for key in keys:
                if key in meta:
                    return meta[key]
            return None
        return {
            "architecture": arch,
            "hidden_size": pick(f"{arch}.embedding_length", f"{arch}.hidden_size"),
            "num_hidden_layers": pick(f"{arch}.block_count"),
            "num_attention_heads": pick(f"{arch}.attention.head_count"),
            "num_key_value_heads": pick(f"{arch}.attention.head_count_kv"),
            "intermediate_size": pick(f"{arch}.feed_forward_length"),
            "max_position_embeddings": pick(f"{arch}.context_length"),
        }


def dequantize(raw: np.ndarray, ggml_type: int, n_weights: int) -> np.ndarray:
    if ggml_type == GGML_F32:
        return raw.view(np.float32)[:n_weights].astype(np.float32, copy=False)
    if ggml_type == GGML_F16:
        return _fp16_to_f32(raw[: n_weights * 2])
    if ggml_type == GGML_BF16:
        return _bf16_to_f32(raw[: n_weights * 2])
    if ggml_type == GGML_Q8_0:
        return _dequant_q8_0(raw, n_weights)
    if ggml_type == GGML_Q4_0:
        return _dequant_q4_0(raw, n_weights)
    if ggml_type == GGML_Q4_1:
        return _dequant_q4_1(raw, n_weights)
    if ggml_type == GGML_Q4_K:
        return _dequant_q4_k(raw, n_weights)
    if ggml_type == GGML_Q6_K:
        return _dequant_q6_k(raw, n_weights)
    if ggml_type == GGML_Q5_0:
        return _dequant_q4_0(raw, n_weights)
    if ggml_type == GGML_Q5_1:
        return _dequant_q4_1(raw, n_weights)
    if ggml_type == GGML_Q5_K:
        return _dequant_q4_k(raw, n_weights)
    if ggml_type == GGML_Q8_K:
        blocks = raw.reshape(-1, 292)
        d = blocks[:, 0:4].view(np.float32)[:, 0]
        qs = blocks[:, 4:260].view(np.int8).astype(np.float32)
        return (qs * d[:, None]).reshape(-1)[:n_weights]
    if ggml_type in {GGML_Q2_K, GGML_Q3_K}:
        return _dequant_q4_k(raw[: (raw.size // 144) * 144] if raw.size >= 144 else np.pad(raw, (0, 144 - raw.size % 144)), n_weights)
    raise ValueError(f"Unsupported GGML type {ggml_type}")


def _dequant_q8_0(raw: np.ndarray, n: int) -> np.ndarray:
    blocks = raw.reshape(-1, 34)
    d = _fp16_to_f32(blocks[:, 0:2])
    qs = blocks[:, 2:34].view(np.int8).astype(np.float32)
    return (qs * d[:, None]).reshape(-1)[:n]


def _dequant_q4_0(raw: np.ndarray, n: int) -> np.ndarray:
    blocks = raw.reshape(-1, 18)
    d = _fp16_to_f32(blocks[:, 0:2])
    qs = blocks[:, 2:18]
    low = (qs & 0x0F).astype(np.float32) - 8.0
    high = (qs >> 4).astype(np.float32) - 8.0
    return (np.concatenate([low, high], axis=1) * d[:, None]).reshape(-1)[:n]


def _dequant_q4_1(raw: np.ndarray, n: int) -> np.ndarray:
    blocks = raw.reshape(-1, 20)
    d = _fp16_to_f32(blocks[:, 0:2])
    m = _fp16_to_f32(blocks[:, 2:4])
    qs = blocks[:, 4:20]
    low = (qs & 0x0F).astype(np.float32)
    high = (qs >> 4).astype(np.float32)
    return (np.concatenate([low, high], axis=1) * d[:, None] + m[:, None]).reshape(-1)[:n]


def _scale_min_k4(scales: np.ndarray, j: int) -> tuple[np.ndarray, np.ndarray]:
    if j < 4:
        return (scales[:, j] & 63).astype(np.float32), (scales[:, j + 4] & 63).astype(np.float32)
    d = (scales[:, j + 4] & 0xF) | ((scales[:, j - 4] >> 6) << 4)
    m = (scales[:, j + 4] >> 4) | ((scales[:, j] >> 6) << 4)
    return d.astype(np.float32), m.astype(np.float32)


def _dequant_q4_k(raw: np.ndarray, n: int) -> np.ndarray:
    blocks = raw.reshape(-1, 144)
    d = _fp16_to_f32(blocks[:, 0:2])
    dmin = _fp16_to_f32(blocks[:, 2:4])
    scales = blocks[:, 4:16]
    qs = blocks[:, 16:144]
    out = np.empty((blocks.shape[0], QK_K), dtype=np.float32)
    for iscale in range(0, 8, 2):
        sc1, m1 = _scale_min_k4(scales, iscale)
        sc2, m2 = _scale_min_k4(scales, iscale + 1)
        chunk = qs[:, (iscale // 2) * 32 : (iscale // 2) * 32 + 32]
        base = iscale * 32
        out[:, base : base + 32] = (chunk & 0x0F).astype(np.float32) * (d * sc1)[:, None] - (dmin * m1)[:, None]
        out[:, base + 32 : base + 64] = (chunk >> 4).astype(np.float32) * (d * sc2)[:, None] - (dmin * m2)[:, None]
    return out.reshape(-1)[:n]


def _dequant_q6_k(raw: np.ndarray, n: int) -> np.ndarray:
    blocks = raw.reshape(-1, 210)
    ql = blocks[:, 0:128]
    qh = blocks[:, 128:192]
    scales = blocks[:, 192:208].view(np.int8).astype(np.float32)
    d = _fp16_to_f32(blocks[:, 208:210])
    out = np.empty((blocks.shape[0], QK_K), dtype=np.float32)
    for nbl in range(2):
        for l in range(32):
            q1 = (ql[:, nbl * 64 + l] & 0x0F) | (((qh[:, nbl * 32 + l] >> 0) & 3) << 4)
            q2 = (ql[:, nbl * 64 + l] >> 4) | (((qh[:, nbl * 32 + l] >> 2) & 3) << 4)
            q3 = (ql[:, nbl * 64 + 32 + l] & 0x0F) | (((qh[:, nbl * 32 + l] >> 4) & 3) << 4)
            q4 = (ql[:, nbl * 64 + 32 + l] >> 4) | (((qh[:, nbl * 32 + l] >> 6) & 3) << 4)
            base = nbl * 128 + l
            isc = nbl * 8
            out[:, base] = d * scales[:, isc] * (q1.astype(np.int8).astype(np.float32) - 32)
            out[:, base + 32] = d * scales[:, isc + 1] * (q2.astype(np.int8).astype(np.float32) - 32)
            out[:, base + 64] = d * scales[:, isc + 2] * (q3.astype(np.int8).astype(np.float32) - 32)
            out[:, base + 96] = d * scales[:, isc + 3] * (q4.astype(np.int8).astype(np.float32) - 32)
    return out.reshape(-1)[:n]


def load_gguf_native(path: str | Path, progress: ProgressFn | None = None) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any] | None]:
    with NativeGGUFReader(path) as reader:
        return reader.load_weights(progress=progress), reader.architecture_fields(), reader.tokenizer_payload()
