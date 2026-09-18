from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from inferforge.core.registry import ModelRecord
from inferforge.merger.core.loader import detect_format, detect_model_architecture, load_model_weights
from inferforge.merger.core.weight_blender import MergeConfig, MergeStrategy
from inferforge.merger.pipeline import MergePipeline


def _write_model(path: Path, scale: float) -> Path:
    from safetensors.torch import save_file

    path.mkdir(parents=True, exist_ok=True)
    weights = {
        "model.embed_tokens.weight": torch.ones(32, 8) * scale,
        "model.layers.0.self_attn.q_proj.weight": torch.ones(8, 8) * scale,
        "model.layers.0.mlp.gate_proj.weight": torch.ones(16, 8) * scale,
        "lm_head.weight": torch.ones(32, 8) * scale,
    }
    save_file(weights, str(path / "model.safetensors"))
    (path / "config.json").write_text('{"model_type": "llama", "hidden_size": 8, "vocab_size": 32}', encoding="utf-8")
    return path


def test_detect_format_and_architecture(tmp_path: Path):
    model_dir = _write_model(tmp_path / "m", 1.0)
    assert detect_format(model_dir) == "safetensors"
    weights = load_model_weights(model_dir)
    assert "model.embed_tokens.weight" in weights
    assert detect_model_architecture(weights) == "llama"


def test_pipeline_merges_real_safetensors(tmp_path: Path):
    a = _write_model(tmp_path / "a", 1.0)
    b = _write_model(tmp_path / "b", 3.0)
    records = [
        ModelRecord(name="a", path=str(a), format="safetensors", backend="huggingface"),
        ModelRecord(name="b", path=str(b), format="safetensors", backend="huggingface"),
    ]
    out = tmp_path / "merged"
    pipeline = MergePipeline(
        config=MergeConfig(strategy=MergeStrategy.SIMPLE_AVERAGE, normalize=False, precision="float32"),
        enable_svd=False,
    )
    result = pipeline.run(records, out)
    assert (result / "model.safetensors").exists()
    assert (result / "config.json").exists()
    merged = load_model_weights(result)
    embed = merged["model.embed_tokens.weight"]
    assert torch.allclose(embed, torch.ones_like(embed) * 2.0, atol=1e-4)
    assert not torch.allclose(embed, torch.randn_like(embed))
