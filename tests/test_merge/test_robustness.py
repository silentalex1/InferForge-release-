from __future__ import annotations

import json
from pathlib import Path
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from inferforge.merger.alignment.cka_analyzer import DynamicSVD
from inferforge.merger.alignment.procrustes import ProcrustesAligner
from inferforge.merger.core.loader import resolve_weight_path
from inferforge.merger.core.tensor_utils import TensorUtils
from inferforge.merger.core.tokenizer_aligner import TokenizerAligner
from inferforge.merger.execution.fisher_mask import FisherImportanceMask


def test_dynamic_svd_bfloat16():
    # 2D tensor in bfloat16
    x = torch.randn(64, 32, dtype=torch.bfloat16)
    compressed = DynamicSVD.compress_via_svd(x, target_dim=16)
    assert compressed.shape == (64, 16)
    assert compressed.dtype == torch.bfloat16

    expanded = DynamicSVD.expand_via_svd(compressed, target_dim=32)
    assert expanded.shape == (64, 32)
    assert expanded.dtype == torch.bfloat16


def test_procrustes_and_cka_bfloat16():
    a = torch.randn(32, 32, dtype=torch.bfloat16)
    b = torch.randn(32, 32, dtype=torch.bfloat16)

    rot, aligned = ProcrustesAligner.orthogonal_procrustes(a, b)
    assert aligned.shape == (32, 32)
    assert aligned.dtype == torch.bfloat16
    assert rot.shape == (32, 32)

    perm_weights, col_ind = ProcrustesAligner.Hungarian_weight_matching(a, b)
    assert perm_weights.shape == (32, 32)
    assert perm_weights.dtype == torch.bfloat16
    assert len(col_ind) == 32

    cka_score = ProcrustesAligner.central_kernel_alignment(a, b)
    assert isinstance(cka_score, float)
    assert 0.0 <= cka_score <= 1.0


def test_fisher_importance_mask_mismatched_shapes():
    fisher = FisherImportanceMask(importance_threshold=0.5)
    # Mismatched 2D tensors in bfloat16
    w_a = torch.randn(64, 32, dtype=torch.bfloat16)
    w_b = torch.randn(32, 16, dtype=torch.bfloat16)

    merged = fisher.adaptive_merge(w_a, w_b, merge_ratio=0.5)
    assert merged.shape == (32, 16)
    assert merged.dtype == torch.bfloat16


def test_tensor_utils_norm_scale_fill():
    # 1D norm scale tensor: e.g. layernorm weight
    w = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
    padded = TensorUtils.match_shape(w, shape=(6,), is_norm_scale=True)
    assert padded.shape == (6,)
    # Newly padded elements must be 1.0 (identity for scales)
    assert torch.allclose(padded, torch.ones(6, dtype=torch.float32))


def test_tokenizer_aligner_fast(tmp_path: Path):
    tok1_path = tmp_path / "tok1.json"
    tok2_path = tmp_path / "tok2.json"

    tok1_path.write_text(json.dumps({"vocab": {"<pad>": 0, "hello": 1, "world": 2}}), encoding="utf-8")
    tok2_path.write_text(json.dumps({"vocab": {"<pad>": 0, "world": 1, "inferforge": 2}}), encoding="utf-8")

    aligner = TokenizerAligner(tokenizer_paths=[tok1_path, tok2_path])
    aligner.load_tokenizers()
    vocabs = aligner.extract_vocabularies()
    master = aligner.build_master_vocabulary(vocabs)

    assert set(master) >= {"<pad>", "hello", "world", "inferforge"}
    remaps = aligner.create_remap_dictionaries(vocabs)
    assert len(remaps) == 2


def test_resolve_weight_path_ollama_fallback(tmp_path: Path):
    model_dir = tmp_path / "custom_meta"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "Modelfile").write_text("FROM llama3.1:8b\nPARAMETER temperature 0.7", encoding="utf-8")
    (model_dir / "config.json").write_text('{"model_type": "llama"}', encoding="utf-8")

    resolved = resolve_weight_path(str(model_dir))
    assert resolved.exists()
    assert "blobs" in str(resolved)
