from __future__ import annotations

from pathlib import Path

import pytest

from inferforge.nexara.forever_factory import (
    CYCLE_SIZE,
    iter_cycle_examples,
    load_cycle_records,
    write_cycle_dataset,
)
from inferforge.nexara.forever_loop import ForeverConfig, ForeverRetrainLoop


@pytest.mark.unit
class TestForeverFactory:
    def test_cycle_size_default(self):
        assert CYCLE_SIZE == 250_000

    def test_unique_quality_examples(self):
        items = list(iter_cycle_examples(cycle=0, n=200, seed=1))
        assert len(items) == 200
        hashes = {x["hash"] for x in items}
        assert len(hashes) == 200
        assert all(x["quality"] >= 0.55 for x in items)
        assert all("input" in x and "output" in x for x in items)
        domains = {x["domain"] for x in items}
        assert len(domains) >= 6

    def test_write_and_load(self, temp_dir: Path):
        manifest = write_cycle_dataset(temp_dir, cycle=3, n=25, seed=2)
        assert manifest["examples"] == 25
        assert manifest["avg_quality"] >= 0.55
        recs = load_cycle_records(manifest["path"])
        assert len(recs) == 25
        assert (Path(manifest["path"]) / "manifest.json").exists()


@pytest.mark.unit
class TestForeverLoop:
    def test_select_train_n_cpu(self):
        loop = ForeverRetrainLoop(ForeverConfig(cycle_size=250_000, scale="tiny"))
        n = loop._select_train_n({"mode": "scratch_cpu", "micro_batch_size": 1})
        assert 1 <= n <= 8192

    def test_replay_mix(self, temp_dir: Path):
        m0 = write_cycle_dataset(temp_dir, cycle=0, n=20, seed=0)
        m1 = write_cycle_dataset(temp_dir, cycle=1, n=20, seed=1)
        fresh = load_cycle_records(m1["path"])
        loop = ForeverRetrainLoop(ForeverConfig(replay_fraction=0.25))
        mixed = loop._mix_replay(fresh, Path(m0["path"]))
        assert len(mixed) > len(fresh)

    def test_stop_keeps_complete_shards_only(self, temp_dir: Path):
        from inferforge.nexara.safe_io import StopFlag, cleanup_partials
        stop = StopFlag()
        n = 0

        def halt_after():
            nonlocal n
            n += 1
            return n > 40

        manifest = write_cycle_dataset(temp_dir, cycle=9, n=500, seed=3, stop=halt_after)
        cleanup_partials(temp_dir)
        partials = list(Path(manifest["path"]).glob("*.partial"))
        assert partials == []
        shards = list(Path(manifest["path"]).glob("shard_*.jsonl"))
        for s in shards:
            assert s.stat().st_size > 0

    def test_preference_pairs_present(self):
        items = list(iter_cycle_examples(cycle=0, n=80, seed=4))
        prefs = [x for x in items if x.get("chosen") and x.get("rejected")]
        assert prefs
