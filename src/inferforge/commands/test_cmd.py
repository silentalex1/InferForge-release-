from __future__ import annotations

import json
import time
from pathlib import Path

import click
from platformdirs import user_data_dir
from rich.console import Console
from rich.table import Table

from inferforge.core.registry import Registry

console = Console()

CODING_BENCHMARK = [
    {"id": "c1", "prompt": "Write a function that checks if a string is a palindrome.", "expect_keywords": ["def", "return", "=="]},
    {"id": "c2", "prompt": "Implement binary search on a sorted list.", "expect_keywords": ["def", "while", "mid"]},
    {"id": "c3", "prompt": "Write a function to reverse a linked list.", "expect_keywords": ["def", "next", "None"]},
    {"id": "c4", "prompt": "Explain the difference between a list and a tuple in Python.", "expect_keywords": ["mutable", "tuple"]},
    {"id": "c5", "prompt": "Write a decorator that times function execution.", "expect_keywords": ["def", "import time", "wrapper"]},
]


def _results_dir() -> Path:
    d = Path(user_data_dir("inferforge")) / "test_results"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _score_response(prompt: str, response: str) -> dict:
    test = next((t for t in CODING_BENCHMARK if t["prompt"] == prompt), None)
    lowered = response.lower()
    keywords = test["expect_keywords"] if test else []
    matched = sum(1 for kw in keywords if kw.lower() in lowered)
    total = len(keywords)
    if total == 0:
        return {"passed": bool(response.strip()), "matched": 0, "total": 0}
    return {"passed": matched >= max(total // 2, 1), "matched": matched, "total": total}


@click.group("test")
def test_command():
    """Model testing and quality regression framework."""
    pass


@test_command.command("run")
@click.argument("model")
@click.option("--benchmark", "-b", type=click.Choice(["coding"]), default="coding", help="Built-in benchmark suite")
@click.option("--custom", default=None, type=click.Path(exists=True), help="Path to a custom tests JSON file")
@click.option("--save", "-s", "save_label", default=None, help="Label to save results under for later comparison")
@click.option("--agent", "agent_eval", is_flag=True, help="Score tool-calling ability (parseable/correct calls on held-out agent tasks) instead of the keyword benchmark.")
def run_tests(model: str, benchmark: str, custom: str | None, save_label: str | None, agent_eval: bool):
    """Run quality tests against a model."""
    if agent_eval:
        _run_agent_eval(model, save_label)
        return

    if custom:
        try:
            cases = json.loads(Path(custom).read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            console.print("[red]Custom test file is not valid JSON.[/]")
            return
        if isinstance(cases, dict):
            cases = cases.get("tests", [])
        for case in cases:
            case.setdefault("expect_keywords", [])
    else:
        cases = [dict(c) for c in CODING_BENCHMARK]

    console.print(f"[bold cyan]Testing {model}[/] with {len(cases)} cases\n")

    reg = Registry()
    record = reg.get(model)
    if not record:
        console.print(f"[red]Model '{model}' not found. Run 'forge list' first.[/]")
        return

    engine = None
    try:
        from inferforge.engine import ChatMessage, get_router

        engine = get_router().resolve(record)
    except Exception as exc:
        console.print(f"[red]Could not load model:[/] {exc}")
        return

    table = Table(title=f"Results: {model}")
    table.add_column("Case", style="cyan")
    table.add_column("Keywords Matched", style="yellow")
    table.add_column("Status", style="green bold")
    passed = 0
    details = []
    try:
        for case in cases:
            prompt_text = case.get("prompt", "")
            try:
                response = engine.chat([ChatMessage(role="user", content=prompt_text)])
            except Exception:
                response = ""
            result = _score_response(prompt_text, response)
            status = "[green]PASS[/]" if result["passed"] else "[red]FAIL[/]"
            if result["passed"]:
                passed += 1
            table.add_row(case.get("id", "?"), f"{result['matched']}/{result['total']}", status)
            details.append({"case": case.get("id", "?"), **result})
    finally:
        try:
            engine.close()
        except Exception:
            pass
    console.print(table)

    score = passed / len(cases) * 100 if cases else 0
    console.print(f"\n[bold]Score:[/] {score:.0f}% ({passed}/{len(cases)} passed)")

    if save_label:
        record = {
            "model": model,
            "label": save_label,
            "timestamp": time.time(),
            "score": score,
            "details": details,
        }
        out = _results_dir() / f"{model.replace('/', '_').replace(':', '_')}_{save_label}.json"
        out.write_text(json.dumps(record, indent=2), encoding="utf-8")
        console.print(f"[green]Saved as '{save_label}' ->[/] {out}")


def _run_agent_eval(model: str, save_label: str | None) -> None:
    """Score a model's tool-calling ability with the real agent parser."""
    reg = Registry()
    record = reg.get(model)
    if not record:
        console.print(f"[red]Model '{model}' not found. Run 'forge list' first.[/]")
        return
    model_id = record.meta.get("hf_model_id") or record.path or record.name

    try:
        from inferforge.training.agent_eval import eval_agent_model
    except ImportError as exc:
        console.print(f"[red]Agent eval requires torch + transformers:[/] {exc}")
        return

    console.print(f"[bold cyan]Agent eval: {model}[/] ({model_id})")
    try:
        report = eval_agent_model(model_id)
    except Exception as exc:
        console.print(f"[red]Could not load model:[/] {exc}")
        return

    summary = report["summary"]
    table = Table(title=f"Agent tool-calling: {model}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="yellow")
    table.add_row("emitted a call (tool cases)", f"{summary['tool_call_rate']:.0%}")
    table.add_row("correct tool", f"{summary['correct_tool_rate']:.0%}")
    table.add_row("correct args", f"{summary['arg_accuracy']:.0%}")
    table.add_row("no-tool discipline", f"{summary['no_tool_discipline']:.0%}")
    table.add_row("overall score", f"{summary['score']:.0%}", style="bold green")
    console.print(table)

    fails = [r for r in report["cases"] if not (r["correct_tool"] and r["args_ok"])]
    if fails:
        console.print(f"\n[dim]{len(fails)} failing cases (first 5):[/]")
        for r in fails[:5]:
            console.print(f"  [red]✗[/] {r['task']}  ->  {r['output'][:70]!r}")

    if save_label:
        out = _results_dir() / f"{model.replace('/', '_').replace(':', '_')}_{save_label}_agent.json"
        out.write_text(json.dumps({"model": model, "label": save_label, "timestamp": time.time(),
                                   "score": summary["score"] * 100, "summary": summary,
                                   "details": report["cases"]}, indent=2), encoding="utf-8")
        console.print(f"[green]Saved as '{save_label}' ->[/] {out}")


@test_command.command("compare")
@click.argument("baseline_model")
@click.argument("candidate_model")
@click.option("--baseline-label", default=None, help="Saved label of baseline results")
@click.option("--candidate-label", default=None, help="Saved label of candidate results")
def compare_results(baseline_model: str, candidate_model: str, baseline_label: str | None, candidate_label: str | None):
    """Compare saved test results between two models."""
    def load(model, label):
        pattern = f"{model.replace('/', '_').replace(':', '_')}_{label or '*'}.json"
        files = sorted(_results_dir().glob(pattern))
        if not files:
            return None
        return json.loads(files[-1].read_text(encoding="utf-8"))

    base = load(baseline_model, baseline_label)
    cand = load(candidate_model, candidate_label)
    if not base or not cand:
        console.print("[yellow]No saved results found. Run 'forge test run <model> --save <label>' first.[/]")
        return

    table = Table(title="Quality Comparison")
    table.add_column("Model", style="cyan")
    table.add_column("Score", style="yellow bold")
    verdict = ""
    if cand["score"] > base["score"]:
        verdict = f"[green]{candidate_model} improved by {cand['score'] - base['score']:.0f} points[/]"
    elif cand["score"] < base["score"]:
        verdict = f"[red]{candidate_model} regressed by {base['score'] - cand['score']:.0f} points[/]"
    else:
        verdict = "[dim]Scores are equal[/]"
    table.add_row(base["model"], f"{base['score']:.0f}%")
    table.add_row(cand["model"], f"{cand['score']:.0f}%")
    console.print(table)
    console.print(verdict)
