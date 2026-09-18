from __future__ import annotations

from inferforge.cli import forge, run, run_cmd


def test_run_cmd_is_exported():
    assert callable(run_cmd)
    assert callable(run)
    assert callable(forge)
    assert run_cmd is not None
    assert getattr(run_cmd, "name", "run") == "run"


def test_forge_registers_run():
    command = forge.commands.get("run")
    assert command is not None
    assert command.name == "run"
