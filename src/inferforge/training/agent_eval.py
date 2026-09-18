"""Evaluate a model's agent tool-calling ability.

Loads a HF model (hub id or local dir), prompts it with held-out agent requests,
parses output with the real ``parse_tool_calls`` parser, and scores:

- tool_call_rate: did it emit at least one parseable tool call
- correct_tool_rate: did it call the *right* tool
- arg_accuracy: did the call include all required args (with correct values where checkable)
- overall score

Usage::

    from inferforge.training.agent_eval import eval_agent_model
    report = eval_agent_model("path/to/final")
    print(report["summary"])
"""
from __future__ import annotations

from typing import Any

from inferforge.agent.tools import NAME_ALIASES, parse_tool_calls

from .agent_dataset import AGENT_SYSTEM_PROMPT

# Held-out prompts the generator does NOT produce verbatim.
# (task, expected_tool_or_None, required_args, expected_arg_values)
_EVAL_CASES: list[tuple[str, str | None, list[str], dict[str, str]]] = [
    ("create a file named report.txt with a short status line", "create_file", ["path"], {"path": "report.txt"}),
    ("make notes.md and put a todo list in it", "create_file", ["path"], {"path": "notes.md"}),
    ("can you list all the files in the folder", "list_dir", ["path"], {}),
    ("show me the files in this directory", "list_dir", ["path"], {}),
    ("what's inside data.csv?", "read_file", ["path"], {"path": "data.csv"}),
    ("open the readme file please", "read_file", ["path"], {}),
    ("check how much disk space is free", "check_storage", [], {}),
    ("how much storage do I have left?", "check_storage", [], {}),
    ("run the tests for me", "run_command", ["command"], {}),
    ("can you show my git status?", "run_command", ["command"], {"command": "git status"}),
    ("fetch the homepage at https://example.com", "web_request", ["url"], {"url": "https://example.com"}),
    ("get https://api.weather.dev/now and tell me the temperature", "web_request", ["url"], {"url": "https://api.weather.dev/now"}),
    ("remove old_log.txt", "delete_file", ["path"], {"path": "old_log.txt"}),
    # no-tool questions: the model must NOT call a tool
    ("what is a closure in python?", None, [], {}),
    ("explain what a mutex does", None, [], {}),
    ("thanks, that's all I needed", None, [], {}),
]

REQUIRED_ARGS: dict[str, list[str]] = {
    "create_file": ["path", "content"],
    "edit_file": ["path", "old", "new"],
    "delete_file": ["path"],
    "read_file": ["path"],
    "open_file": ["path"],
    "list_dir": ["path"],
    "run_command": ["command"],
    "web_request": ["url"],
    "check_storage": [],
}


def _canonical(name: str) -> str:
    return NAME_ALIASES.get(name, name)


def _score_case(calls: list[dict], expected: str | None, required: list[str], expected_args: dict[str, str]) -> dict[str, Any]:
    if expected is None:
        # no-tool case: success = no tool call at all
        return {
            "emitted_call": bool(calls),
            "correct_tool": not calls,
            "args_ok": not calls,
        }
    if not calls:
        return {"emitted_call": False, "correct_tool": False, "args_ok": False}
    # use the first call (models sometimes emit several)
    call = calls[0]
    name = _canonical(str(call.get("name", "")))
    correct_tool = name == expected
    req = REQUIRED_ARGS.get(expected, required)
    has_args = all(k in call and call[k] not in (None, "") for k in req)
    values_ok = all(
        str(call.get(k, "")).strip().rstrip("/.") == v.strip().rstrip("/.")
        or v in str(call.get(k, ""))
        for k, v in expected_args.items()
    )
    return {
        "emitted_call": True,
        "correct_tool": correct_tool,
        "args_ok": correct_tool and has_args and values_ok,
        "called": name,
    }


def _prompt_for(task: str) -> str:
    return (
        f"<|im_start|>system\n{AGENT_SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{task}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def eval_agent_model(
    model_id: str,
    *,
    max_new_tokens: int = 160,
    device: str | None = None,
    temperature: float = 0.0,
    verbose: bool = False,
) -> dict[str, Any]:
    """Generate answers to held-out agent tasks and score tool-call correctness."""
    import torch  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    rows: list[dict[str, Any]] = []
    for task, expected, required, expected_args in _EVAL_CASES:
        ids = tokenizer(_prompt_for(task), return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            out = model.generate(
                ids,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                pad_token_id=pad_id,
            )
        text = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=False)
        # stop at end-of-turn markers
        for marker in ("<|im_end|>", "<|im_start|>"):
            if marker in text:
                text = text.split(marker)[0]
        calls = parse_tool_calls(text)
        score = _score_case(calls, expected, required, expected_args)
        rows.append({"task": task, "expected": expected, "output": text.strip()[:400], **score})
        if verbose:
            print(f"[{score.get('called', '—'):>12}] {task}  ->  {text.strip()[:80]!r}")

    n = len(rows)
    tool_cases = [r for r in rows if r["expected"] is not None]
    summary = {
        "cases": n,
        "tool_call_rate": sum(r["emitted_call"] for r in tool_cases) / max(len(tool_cases), 1),
        "correct_tool_rate": sum(r["correct_tool"] for r in tool_cases) / max(len(tool_cases), 1),
        "arg_accuracy": sum(r["args_ok"] for r in tool_cases) / max(len(tool_cases), 1),
        "no_tool_discipline": sum(not r["emitted_call"] for r in rows if r["expected"] is None)
        / max(sum(1 for r in rows if r["expected"] is None), 1),
    }
    summary["score"] = (
        0.4 * summary["correct_tool_rate"]
        + 0.3 * summary["arg_accuracy"]
        + 0.3 * summary["no_tool_discipline"]
    )
    return {"summary": summary, "cases": rows}


def compare_models(model_ids: list[str], **kwargs: Any) -> dict[str, dict[str, Any]]:
    """Score several models side by side (e.g. base vs fine-tuned)."""
    return {mid: eval_agent_model(mid, **kwargs) for mid in model_ids}
