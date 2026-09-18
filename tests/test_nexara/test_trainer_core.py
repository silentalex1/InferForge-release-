"""Regression tests for the UniversalTrainer core loop.

These guard the bugs found in the original implementation: gradient accumulation silently
training 1/N of the data, LR schedules never completing, EOS being masked out when
pad == eos, prompts being trained on during SFT, unbounded checkpoints and NaN hangs.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from inferforge.nexara.universal_trainer import (
    CharTokenizer,
    UniversalTrainConfig,
    UniversalTrainer,
    _prune_checkpoints,
    record_to_prompt_completion,
    tokenize_records,
)

torch = pytest.importorskip("torch")

from inferforge.nexara.universal_trainer import build_scheduler  # noqa: E402


def _write_jsonl(path: Path, n: int = 24) -> Path:
    rows = [{"text": f"the quick brown fox {i} jumps over the lazy dog " * 4} for i in range(n)]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return path


def _cfg(tmp_path: Path, data: Path, **overrides) -> UniversalTrainConfig:
    base = dict(
        output_dir=str(tmp_path / "run"),
        from_scratch=True,
        scale="nano",
        data=str(data),
        seq_len=32,
        batch_size=4,
        epochs=1,
        logging_steps=1,
        eval_steps=10_000,
        save_steps=10_000,
        warmup_ratio=0.1,
        packing=True,
    )
    base.update(overrides)
    return UniversalTrainConfig(**base)


# --------------------------------------------------------------------------- data layer


@pytest.mark.unit
def test_prompt_completion_split_alpaca():
    prompt, completion = record_to_prompt_completion({"instruction": "Say hi", "output": "hello"})
    assert prompt.endswith("<|im_start|>assistant\n")
    assert completion.startswith("hello")


@pytest.mark.unit
def test_prompt_completion_split_chat_uses_last_assistant_turn():
    rec = {"messages": [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "c"},
        {"role": "assistant", "content": "FINAL"},
    ]}
    prompt, completion = record_to_prompt_completion(rec)
    assert "FINAL" not in prompt
    assert completion.startswith("FINAL")


@pytest.mark.unit
def test_completions_only_masks_prompt_tokens():
    tok = CharTokenizer(["abcdefghijklmnopqrstuvwxyz <|im_start|>end\n"])
    rec = {"prompt": "question: ", "completion": "answer"}
    out = tokenize_records([rec], tok, seq_len=64, packing=False, completions_only=True)
    labels = out["labels"][0]
    n_prompt = len(tok.encode("question: ", add_eos=False))
    assert all(lbl == -100 for lbl in labels[:n_prompt])
    assert any(lbl != -100 for lbl in labels[n_prompt:])
    # without masking every real token is a label
    out2 = tokenize_records([rec], tok, seq_len=64, packing=False, completions_only=False)
    real = sum(out2["attention_mask"][0])
    assert sum(1 for lbl in out2["labels"][0] if lbl != -100) == real


@pytest.mark.unit
def test_eos_is_learned_when_pad_equals_eos():
    tok = CharTokenizer(["hello world"])
    tok.pad_token_id = tok.eos_token_id  # gpt2-style tokenizer
    out = tokenize_records([{"text": "hello"}], tok, seq_len=16, packing=False)
    ids, mask, labels = out["input_ids"][0], out["attention_mask"][0], out["labels"][0]
    eos_pos = len(tok.encode("hello", add_eos=False))
    assert ids[eos_pos] == tok.eos_token_id
    assert mask[eos_pos] == 1, "EOS must be attended, not treated as padding"
    assert labels[eos_pos] == tok.eos_token_id, "EOS must be a training target"
    assert mask[eos_pos + 1] == 0 and labels[eos_pos + 1] == -100


# --------------------------------------------------------------------------- schedule


@pytest.mark.unit
def test_scheduler_warms_up_and_decays_to_floor():
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([p], lr=1.0)
    sched = build_scheduler(opt, steps=100, warmup=10, kind="cosine", min_lr_ratio=0.1)
    lrs = []
    for _ in range(100):
        lrs.append(sched.get_last_lr()[0])
        opt.step()
        sched.step()
    assert lrs[0] == pytest.approx(0.1)  # first step is not wasted at ~0
    assert max(lrs) == pytest.approx(1.0)
    assert lrs[-1] == pytest.approx(0.1, abs=1e-3)


@pytest.mark.unit
def test_prune_checkpoints_keeps_newest(tmp_path: Path):
    for step in (10, 20, 30, 40):
        (tmp_path / f"checkpoint-{step}").mkdir()
    (tmp_path / "best").mkdir()
    _prune_checkpoints(tmp_path, 2)
    remaining = sorted(p.name for p in tmp_path.iterdir())
    assert remaining == ["best", "checkpoint-30", "checkpoint-40"]


# --------------------------------------------------------------------------- training loop


@pytest.mark.unit
def test_gradient_accumulation_sees_all_data(tmp_path: Path):
    data = _write_jsonl(tmp_path / "data.jsonl")
    r1 = UniversalTrainer(_cfg(tmp_path / "a", data, gradient_accumulation_steps=1), hardware={"ram": 16}).train()
    r4 = UniversalTrainer(_cfg(tmp_path / "b", data, gradient_accumulation_steps=4), hardware={"ram": 16}).train()
    assert r1["status"] == "completed" and r4["status"] == "completed"
    # Same data, same epoch count -> identical number of samples regardless of accumulation.
    assert r4["samples_seen"] == r1["samples_seen"]
    assert r4["micro_steps"] == r1["micro_steps"]
    # Optimizer steps shrink by ~accum, and the schedule runs to completion in both cases.
    assert r4["steps"] <= -(-r1["steps"] // 4)
    assert r4["steps"] == r4["max_steps"]
    assert r1["epochs_completed"] == pytest.approx(1.0)
    assert r4["epochs_completed"] == pytest.approx(1.0)
    peak = r1["resolved"]["learning_rate"]
    assert r1["train_history"][-1]["lr"] < 0.2 * peak
    assert r4["train_history"][-1]["lr"] < 0.2 * peak


@pytest.mark.unit
def test_nan_loss_aborts_instead_of_hanging(tmp_path: Path, monkeypatch):
    data = _write_jsonl(tmp_path / "data.jsonl")
    trainer = UniversalTrainer(_cfg(tmp_path / "nan", data, max_nan_skips=3), hardware={"ram": 16})

    def nan_loss(self, model, batch):
        return torch.tensor(float("nan"), requires_grad=True)

    monkeypatch.setattr(UniversalTrainer, "_forward_loss", nan_loss)
    result = trainer.train()
    assert result["status"] == "diverged"
    assert result["steps"] == 0
    assert result["monitor"]["nan_skips"] >= 3


@pytest.mark.unit
def test_save_total_limit_enforced_during_training(tmp_path: Path):
    data = _write_jsonl(tmp_path / "data.jsonl", n=32)
    cfg = _cfg(tmp_path / "ckpt", data, save_steps=1, save_total_limit=2, gradient_accumulation_steps=1)
    result = UniversalTrainer(cfg, hardware={"ram": 16}).train()
    assert result["steps"] >= 3
    ckpts = [p for p in Path(cfg.output_dir).glob("checkpoint-*") if p.is_dir()]
    assert len(ckpts) <= 2


# --------------------------------------------------------------------------- agent data


@pytest.mark.unit
def test_agent_dataset_is_deterministic_and_valid():
    from inferforge.training.agent_dataset import build_agent_dataset, validate_agent_records
    a = build_agent_dataset(40)
    b = build_agent_dataset(40)
    assert a == b  # deterministic seed
    valid, issues = validate_agent_records(a)
    assert not issues
    assert len(valid) == len(a)
    # tool calls in assistant turns must parse with the real agent parser
    from inferforge.agent.tools import parse_tool_calls
    tool_turns = sum(
        len(parse_tool_calls(m["content"]))
        for r in valid for m in r["messages"] if m["role"] == "assistant"
    )
    assert tool_turns >= len(valid) // 2  # ~3/4 of records contain at least one call


@pytest.mark.unit
def test_agent_validator_rejects_unknown_and_malformed_calls():
    from inferforge.training.agent_dataset import validate_agent_records
    recs = [
        {"messages": [
            {"role": "user", "content": "do it"},
            {"role": "assistant", "content": '```json\n{"name": "nuke_everything"}\n```'},
        ]},
        {"messages": [
            {"role": "user", "content": "read it"},
            {"role": "assistant", "content": '```json\n{"name": "read_file"}\n```'},  # missing path
        ]},
    ]
    valid, issues = validate_agent_records(recs)
    assert valid == []
    assert len(issues) == 2


@pytest.mark.unit
def test_agent_validator_converts_legacy_io_records():
    from inferforge.training.agent_dataset import validate_agent_records
    valid, issues = validate_agent_records([{"input": "hi", "output": "hello"}])
    assert len(valid) == 1 and not issues
    assert valid[0]["messages"][0]["role"] == "user"


@pytest.mark.unit
def test_agent_preference_dataset_has_valid_schema():
    from inferforge.training.agent_dataset import build_agent_preference_dataset
    from inferforge.nexara.universal_trainer import detect_record_schema
    from inferforge.agent.tools import parse_tool_calls
    prefs = build_agent_preference_dataset(30)
    assert len(prefs) == 30
    assert all(detect_record_schema(p) == "preference" for p in prefs)
    for p in prefs:
        assert p["prompt"].endswith("<|im_start|>assistant\n")
        assert p["chosen"] != p["rejected"]
    # most chosen answers should be valid tool calls; rejected ones should NOT be better
    chosen_calls = sum(bool(parse_tool_calls(p["chosen"])) for p in prefs)
    rejected_calls = sum(bool(parse_tool_calls(p["rejected"])) for p in prefs)
    assert chosen_calls > rejected_calls


@pytest.mark.unit
def test_agent_eval_scorer():
    from inferforge.training.agent_eval import _score_case
    good = [{"name": "read_file", "path": "data.csv"}]
    assert _score_case(good, "read_file", ["path"], {"path": "data.csv"})["args_ok"] is True
    assert _score_case(good, "create_file", ["path"], {})["correct_tool"] is False
    assert _score_case([{"name": "read_file"}], "read_file", ["path"], {})["args_ok"] is False
    # no-tool cases: emitting a call is wrong, staying quiet is right
    assert _score_case([], None, [], {})["correct_tool"] is True
    assert _score_case(good, None, [], {})["correct_tool"] is False


@pytest.mark.unit
def test_advanced_engine_accumulation_steps_once_per_window():
    torch = pytest.importorskip("torch")
    from inferforge.nexara.advanced_training import AdvancedTrainingConfig, AdvancedTrainingEngine
    cfg = AdvancedTrainingConfig(
        gradient_accumulation_steps=4, mixed_precision=False,
        use_mixup=False, use_cutmix=False, focal_loss_gamma=0.0,
        contrastive_temperature=0.0,
    )
    eng = AdvancedTrainingEngine(cfg)
    eng.initialize_model(torch.nn.Linear(4, 4))
    eng.initialize_optimizer()

    steps = {"n": 0}
    orig_step = eng.optimizer.step
    eng.optimizer.step = lambda *a, **k: (steps.__setitem__("n", steps["n"] + 1), orig_step(*a, **k))[1]

    batch = torch.randn(2, 4)
    targets = torch.randint(0, 4, (2,))
    for _ in range(3):
        eng.train_step(batch, targets)
    assert steps["n"] == 0  # no step mid-window (old bug stepped on micro 0)
    eng.train_step(batch, targets)
    assert steps["n"] == 1
    assert eng.global_step == 1
