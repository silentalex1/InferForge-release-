from __future__ import annotations

import json
from pathlib import Path
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from safetensors.torch import save_file
from inferforge.core.registry import ModelRecord
from inferforge.merger.core.loader import LazyModelReader, estimate_merge_memory
from inferforge.merger.core.weight_blender import MergeConfig, MergeStrategy
from inferforge.merger.pipeline import MergePipeline


def _create_mock_safetensors(dir_path: Path, prefix: str = "model", scale: float = 1.0) -> Path:
    dir_path.mkdir(parents=True, exist_ok=True)
    tensors = {
        "model.embed_tokens.weight": torch.ones(16, 8) * scale,
        "model.layers.0.self_attn.q_proj.weight": torch.ones(8, 8) * scale,
        "model.layers.0.mlp.down_proj.weight": torch.ones(8, 16) * scale,
        "model.layers.1.self_attn.q_proj.weight": torch.ones(8, 8) * scale,
        "lm_head.weight": torch.ones(16, 8) * scale,
    }
    file_path = dir_path / f"{prefix}.safetensors"
    save_file(tensors, str(file_path))
    (dir_path / "config.json").write_text('{"model_type": "llama", "hidden_size": 8, "vocab_size": 16}', encoding="utf-8")
    return dir_path


def test_lazy_model_reader(tmp_path: Path):
    model_dir = _create_mock_safetensors(tmp_path / "m1", scale=2.5)
    with LazyModelReader(model_dir) as reader:
        names = reader.get_tensor_names()
        assert "model.embed_tokens.weight" in names
        assert "model.layers.0.self_attn.q_proj.weight" in names
        assert reader.has_tensor("lm_head.weight")
        t = reader.get_tensor("lm_head.weight")
        assert torch.allclose(t, torch.ones(16, 8) * 2.5)


def test_estimate_merge_memory(tmp_path: Path):
    m1 = _create_mock_safetensors(tmp_path / "m1", scale=1.0)
    m2 = _create_mock_safetensors(tmp_path / "m2", scale=2.0)
    rec1 = ModelRecord(name="m1", path=str(m1), format="safetensors")
    rec2 = ModelRecord(name="m2", path=str(m2), format="safetensors")
    
    info = estimate_merge_memory([rec1, rec2], strategy="dare_ties", layer_wise=True, output_dir=tmp_path / "out")
    assert "available_ram_gb" in info
    assert "streaming_ram_gb" in info
    assert info["disk_ok"] is True
    assert info["streaming_ram_gb"] <= info["in_memory_ram_gb"]


def test_streaming_merge_pipeline(tmp_path: Path):
    m1 = _create_mock_safetensors(tmp_path / "m1", scale=1.0)
    m2 = _create_mock_safetensors(tmp_path / "m2", scale=3.0)
    rec1 = ModelRecord(name="m1", path=str(m1), format="safetensors", backend="huggingface")
    rec2 = ModelRecord(name="m2", path=str(m2), format="safetensors", backend="huggingface")
    
    out_dir = tmp_path / "merged_out"
    pipeline = MergePipeline(
        config=MergeConfig(strategy=MergeStrategy.SIMPLE_AVERAGE, precision="float32"),
        lazy_load=True,
    )
    result = pipeline.run([rec1, rec2], out_dir)
    assert (result / "model.safetensors").exists()
    assert (result / "config.json").exists()
    assert (result / "merge_manifest.json").exists()

    with LazyModelReader(result) as reader:
        embed = reader.get_tensor("model.embed_tokens.weight")
        assert torch.allclose(embed, torch.ones(16, 8) * 2.0)


def test_checkpoint_resume(tmp_path: Path):
    out_dir = tmp_path / "resume_out"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / ".merge_checkpoint.json"
    ckpt.write_text(json.dumps({
        "completed_keys": ["model.embed_tokens.weight"],
        "timestamp": 123456
    }), encoding="utf-8")

    m1 = _create_mock_safetensors(tmp_path / "m1", scale=1.0)
    m2 = _create_mock_safetensors(tmp_path / "m2", scale=2.0)
    rec1 = ModelRecord(name="m1", path=str(m1), format="safetensors", backend="huggingface")
    rec2 = ModelRecord(name="m2", path=str(m2), format="safetensors", backend="huggingface")

    pipeline = MergePipeline(
        config=MergeConfig(strategy=MergeStrategy.SIMPLE_AVERAGE, precision="float32"),
        lazy_load=True,
        resume=True,
    )
    result = pipeline.run([rec1, rec2], out_dir)
    assert (result / "model.safetensors").exists()
    # Checkpoint should be cleaned up on completion
    assert not ckpt.exists()
