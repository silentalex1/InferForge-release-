from __future__ import annotations

import math

import pytest

from inferforge.nexara.alignment import AlignmentConfig, compute_alignment_loss
from inferforge.nexara.architectures import ArchConfig, build_arch_config, count_parameters
from inferforge.nexara.compiler import NexaraCompiler
from inferforge.nexara.engine import NexaraEngine
from inferforge.nexara.scaling import (
    RECIPE_CATALOG,
    SCALE_ORDER,
    chinchilla_tokens,
    fit_recipe_to_hardware,
    get_recipe,
    list_recipes,
    plan_training,
    resolve_scale,
)
from inferforge.nexara.universal_trainer import (
    detect_record_schema,
    record_to_text,
    tokenize_records,
    CharTokenizer,
)


@pytest.mark.unit
class TestScaling:
    def test_catalog_covers_nano_to_gpt4o(self):
        for name in SCALE_ORDER:
            assert name in RECIPE_CATALOG
        assert get_recipe("nano").params < get_recipe("tiny").params
        assert get_recipe("gpt4o").params > get_recipe("large").params
        assert get_recipe("gpt4").use_moe
        assert "dpo" in get_recipe("gpt4o").stages

    def test_aliases(self):
        assert resolve_scale("gpt-4o") == "gpt4o"
        assert resolve_scale("llama3-8b") == "large"
        assert resolve_scale(None, params=1_000_000) in RECIPE_CATALOG

    def test_chinchilla_and_plan(self):
        r = get_recipe("tiny")
        assert chinchilla_tokens(r.params) >= r.params * 10
        plan = plan_training("tiny", vram_gb=0, ram_gb=16, gpu_count=0, gpu_available=False)
        assert plan["fitted"]["fitted_scale"] in RECIPE_CATALOG
        assert plan["stages"]
        assert "honest_gap" in plan

    def test_fit_downscales_without_gpu(self):
        recipe = get_recipe("gpt4o")
        fit = fit_recipe_to_hardware(recipe, vram_gb=0, ram_gb=8, gpu_count=0, gpu_available=False)
        assert fit["fitted_scale"] in {"nano", "micro", "tiny"}
        assert fit["mode"] in {"scratch_cpu", "qlora"}

    def test_list_recipes(self):
        rows = list_recipes()
        assert len(rows) == len(SCALE_ORDER)
        assert rows[0]["name"] == "nano"
        assert rows[-1]["name"] == "gpt4o"


@pytest.mark.unit
class TestArchitectures:
    def test_arch_from_recipe(self):
        cfg = build_arch_config("nano")
        assert isinstance(cfg, ArchConfig)
        assert cfg.n_layer >= 2
        assert cfg.n_embd >= 64

    def test_scratch_forward_if_torch(self):
        torch = pytest.importorskip("torch")
        from inferforge.nexara.architectures import build_scratch_model
        model = build_scratch_model("nano")
        ids = torch.randint(0, model.cfg.vocab_size, (2, 16))
        out = model(ids, labels=ids)
        assert out["loss"] is not None
        assert out["logits"].shape[-1] == model.cfg.vocab_size
        stats = count_parameters(model)
        assert stats["total"] > 0
        assert stats["trainable"] == stats["total"]


@pytest.mark.unit
class TestAlignment:
    def test_dpo_loss_prefers_chosen(self):
        torch = pytest.importorskip("torch")
        chosen = torch.tensor([-1.0, -0.5])
        rejected = torch.tensor([-3.0, -2.5])
        ref_c = torch.zeros(2)
        ref_r = torch.zeros(2)
        loss, metrics = compute_alignment_loss("dpo", chosen, rejected, ref_c, ref_r)
        assert torch.isfinite(loss)
        assert metrics["rewards/accuracy"] == 1.0

    def test_methods_run(self):
        torch = pytest.importorskip("torch")
        c = torch.tensor([-1.0, -0.2])
        r = torch.tensor([-2.0, -1.5])
        z = torch.zeros(2)
        nll = torch.tensor(1.2)
        for method in ("dpo", "orpo", "ipo", "simpo", "cpo", "slic"):
            loss, _ = compute_alignment_loss(
                method, c, r, z, z, chosen_nll=nll, policy_chosen_avg=c, policy_rejected_avg=r,
                config=AlignmentConfig(method=method),
            )
            assert torch.isfinite(loss)


@pytest.mark.unit
class TestDataAdapter:
    def test_schema_and_render(self):
        assert detect_record_schema({"prompt": "p", "chosen": "a", "rejected": "b"}) == "preference"
        assert detect_record_schema({"instruction": "x", "output": "y"}) == "alpaca"
        text = record_to_text({"input": "hi", "output": "hello"})
        assert "hi" in text and "hello" in text

    def test_char_tokenizer_pack(self):
        tok = CharTokenizer(["abc def ghi"])
        packed = tokenize_records([{"text": "abc def ghi " * 20}], tok, seq_len=16, packing=True)
        assert packed["input_ids"]
        assert len(packed["input_ids"][0]) == 16
        assert len(packed["labels"][0]) == 16


@pytest.mark.unit
class TestEngineAndCompiler:
    def test_compile_ready(self, sample_nexara_code, temp_dir):
        engine = NexaraEngine()
        result = engine.compile_and_train(sample_nexara_code, temp_dir)
        assert result["status"] == "ready_for_training"
        assert result["models_count"] == 1
        assert (temp_dir / "train_nexara.py").exists()
        content = (temp_dir / "train_nexara.py").read_text(encoding="utf-8")
        assert "import torch" in content
        assert "Trainer" in content
        assert "UniversalTrainer" in content

    def test_compiler_script(self, sample_nexara_code, temp_dir):
        compiler = NexaraCompiler()
        compiled = compiler.compile(sample_nexara_code, temp_dir)
        path = compiler.generate_python_code(compiled, temp_dir / "train.py")
        text = path.read_text(encoding="utf-8")
        assert "import torch" in text
        assert "Trainer" in text

    def test_list_and_plan(self):
        engine = NexaraEngine()
        rows = engine.list_recipes()
        assert any(r["name"] == "gpt4o" for r in rows)
        plan = engine.plan_training("nano")
        assert plan["target"] == "nano"

    def test_parser_base_alias(self, sample_nexara_code):
        engine = NexaraEngine()
        ok, errors = engine.validate_code(sample_nexara_code)
        assert ok, errors
