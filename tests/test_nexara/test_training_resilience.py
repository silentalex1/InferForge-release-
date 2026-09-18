from __future__ import annotations

from pathlib import Path

import pytest

from inferforge.nexara.mixed_precision import MixedPrecisionEngine, PrecisionConfig, PrecisionMode
from inferforge.nexara.training_monitor import TrainingMonitor


def test_mixed_precision_detects_nan_overflow():
    engine = MixedPrecisionEngine(PrecisionConfig(mode=PrecisionMode.FP16))
    torch = pytest.importorskip("torch")
    grads = {"w": torch.tensor([1.0, float("nan")])}
    assert engine.check_overflow(grads) is True
    assert engine.overflow_count == 1


def test_mixed_precision_finite_grads_are_ok():
    engine = MixedPrecisionEngine(PrecisionConfig(mode=PrecisionMode.BF16))
    torch = pytest.importorskip("torch")
    grads = {"w": torch.tensor([0.1, -0.2, 0.3])}
    assert engine.check_overflow(grads) is False


def test_training_monitor_writes_log(tmp_path: Path):
    monitor = TrainingMonitor(tmp_path)
    monitor.record_nan_skip()
    snapshot = monitor.log(10, loss=1.25, lr=0.001, epoch=0.5)
    assert snapshot.loss == 1.25
    assert snapshot.nan_skips == 1
    assert (tmp_path / "training_log.jsonl").exists()
    summary = monitor.summary()
    assert summary["nan_skips"] == 1
    assert summary["last_loss"] == 1.25
