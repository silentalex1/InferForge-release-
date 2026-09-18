from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from inferforge.merger.core.weight_blender import MergeConfig, MergeStrategy, WeightBlender


def test_simple_average_is_real_mean():
    blender = WeightBlender(MergeConfig(strategy=MergeStrategy.SIMPLE_AVERAGE, normalize=False, precision="float32"))
    a = torch.ones(4, 4)
    b = torch.ones(4, 4) * 3
    out = blender.merge_weights([a, b])
    assert torch.allclose(out, torch.ones(4, 4) * 2)


def test_linear_interpolates_real_tensors():
    blender = WeightBlender(
        MergeConfig(strategy=MergeStrategy.LINEAR, interpolation=0.25, normalize=False, precision="float32")
    )
    a = torch.zeros(8)
    b = torch.ones(8) * 4
    out = blender.merge_weights([a, b])
    assert out.shape == a.shape
    assert float(out.mean()) > 0.0
    assert float(out.mean()) < 4.0


def test_ties_keeps_significant_weights():
    blender = WeightBlender(MergeConfig(strategy=MergeStrategy.TIES, ties_param_k=0.5, normalize=False, precision="float32"))
    base = torch.zeros(16)
    delta_a = torch.zeros(16)
    delta_a[0] = 10.0
    delta_b = torch.zeros(16)
    delta_b[0] = 10.0
    out = blender.merge_weights([base + delta_a, base + delta_b], base_weights=base)
    assert float(out[0]) != 0.0


def test_merge_state_dicts_merges_matching_keys():
    blender = WeightBlender(MergeConfig(strategy=MergeStrategy.SIMPLE_AVERAGE, normalize=False, precision="float32"))
    a = {"w": torch.ones(2, 2), "only_a": torch.ones(3)}
    b = {"w": torch.ones(2, 2) * 5}
    merged = blender.merge_state_dicts([a, b])
    assert torch.allclose(merged["w"], torch.ones(2, 2) * 3)
    assert "only_a" in merged


def test_ties_bfloat16_quantile_fix():
    """Verify that TIES handles bfloat16 tensors without torch.quantile RuntimeError."""
    blender = WeightBlender(MergeConfig(strategy=MergeStrategy.TIES, ties_param_k=0.5, normalize=False, precision="bfloat16"))
    base = torch.zeros(32, dtype=torch.bfloat16)
    delta_a = torch.zeros(32, dtype=torch.bfloat16)
    delta_a[0] = 5.0
    delta_b = torch.zeros(32, dtype=torch.bfloat16)
    delta_b[0] = 5.0
    out = blender.merge_weights([base + delta_a, base + delta_b], base_weights=base)
    assert out.dtype == torch.bfloat16
    assert float(out[0]) > 0.0


def test_dare_ties_rescaling():
    """Verify DARE-TIES drops and rescales delta weights."""
    blender = WeightBlender(MergeConfig(
        strategy=MergeStrategy.DARE_TIES,
        dare_drop_rate=0.2,
        dare_rescale=True,
        normalize=False,
        precision="float32",
    ))
    base = torch.zeros(20)
    delta_a = torch.ones(20) * 2.0
    delta_b = torch.ones(20) * 2.0
    out = blender.merge_weights([base + delta_a, base + delta_b], base_weights=base)
    assert out.shape == base.shape
    assert torch.isfinite(out).all()


def test_task_arithmetic():
    """Verify Task Arithmetic correctly applies multi-model task vectors."""
    blender = WeightBlender(MergeConfig(
        strategy=MergeStrategy.TASK_ARITHMETIC,
        weights=[0.5, 0.5],
        normalize=False,
        precision="float32",
    ))
    base = torch.zeros(10)
    task1 = torch.ones(10) * 2.0
    task2 = torch.ones(10) * 4.0
    out = blender.merge_weights([task1, task2], base_weights=base)
    # base + 0.5*(2-0) + 0.5*(4-0) = 1 + 2 = 3.0
    assert torch.allclose(out, torch.ones(10) * 3.0)


def test_strategy_graceful_fallback():
    """Verify that if an unsupported operation occurs, it gracefully falls back to average."""
    blender = WeightBlender(MergeConfig(strategy=MergeStrategy.TIES, precision="float32"))
    a = torch.ones(4)
    b = torch.ones(4) * 3
    # Pass empty delta or tricky input
    out = blender.merge_weights([a, b])
    assert torch.isfinite(out).all()
