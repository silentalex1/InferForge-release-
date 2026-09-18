"""Synthetic agent (tool-calling) SFT dataset.

Every example is a chat record ``{"messages": [...]}`` whose assistant turns emit tool calls in
the exact wire format understood by :func:`inferforge.agent.tools.parse_tool_calls`::

    ```json
    {"name": "read_file", "path": "src/app.py"}
    ```

Traces are multi-turn: user request -> assistant tool call -> tool result (as a ``tool`` turn)
-> assistant final answer. Roughly a quarter of the examples are *negative* (pure Q&A) so the
model also learns when **not** to call a tool.
"""
from __future__ import annotations

import json
import random
import re
from typing import Any

from inferforge.agent.tools import NAME_ALIASES, TOOL_NAMES

AGENT_SYSTEM_PROMPT = """You are a local coding agent running inside InferForge.
You can act on the user's workspace with tools. When an action is needed, emit exactly one tool
call as a ```json fence and nothing else, then wait for the result:

```json
{"name": "create_file", "path": "hello.py", "content": "print(1)\\n"}
```

Available tools: create_file(path, content), edit_file(path, old, new), delete_file(path),
read_file(path), open_file(path), list_dir(path), run_command(command),
web_request(url, method), check_storage().

Rules:
- Use a tool whenever the user wants something read, changed, run, or fetched.
- After a tool result arrives, answer the user concisely in markdown. Do not repeat the call.
- For questions that need no workspace access, answer directly with no tool call.
- Never invent file contents or command output; read or run first.
"""

CANONICAL_TOOLS = sorted(TOOL_NAMES - set(NAME_ALIASES))

_FILES = [
    "main.py", "src/app.py", "src/utils/helpers.py", "README.md", "tests/test_api.py",
    "config.yaml", "package.json", "Dockerfile", "scripts/deploy.sh", "notes.txt",
]
_DIRS = [".", "src", "tests", "scripts", "docs"]
_COMMANDS = [
    ("run the tests", "pytest -q", "12 passed in 0.84s"),
    ("check the git status", "git status --short", " M src/app.py\n?? notes.txt"),
    ("install the dependencies", "pip install -r requirements.txt", "Successfully installed requests-2.32.3"),
    ("lint the project", "ruff check .", "All checks passed!"),
    ("show the python version", "python --version", "Python 3.12.4"),
    ("build the docker image", "docker build -t app .", "Successfully tagged app:latest"),
    ("list the last three commits", "git log --oneline -3", "a1b2c3d fix login\n9f8e7d6 add tests\n1234abc init"),
]
_URLS = [
    "https://api.github.com/repos/python/cpython",
    "https://httpbin.org/get",
    "https://api.example.com/v1/status",
]
_QA = [
    ("What's the difference between a list and a tuple in Python?",
     "A **list** is mutable and a **tuple** is immutable. Use tuples for fixed records and lists when you need to append or modify items."),
    ("Explain what a decorator is.",
     "A decorator is a function that takes another function and returns a wrapped version of it, letting you add behaviour (logging, caching, auth) without changing the original body."),
    ("When should I use async in Python?",
     "Use `async`/`await` for I/O-bound work with many concurrent connections (HTTP, sockets, DB). For CPU-bound work prefer processes or native extensions."),
    ("What does HTTP 429 mean?",
     "429 Too Many Requests: the client is being rate-limited. Back off (ideally honouring `Retry-After`) and retry later."),
    ("Is Rust memory safe?",
     "Yes for safe Rust: the borrow checker enforces ownership and lifetimes at compile time. `unsafe` blocks opt out of those guarantees."),
    ("Thanks, that's all.", "You're welcome. Ping me when you want to make the next change."),
]


def _fence(call: dict[str, Any]) -> str:
    return "```json\n" + json.dumps(call, ensure_ascii=False) + "\n```"


def _record(turns: list[tuple[str, str]], system: str = AGENT_SYSTEM_PROMPT) -> dict[str, Any]:
    messages = [{"role": "system", "content": system}]
    messages += [{"role": role, "content": content} for role, content in turns]
    return {"messages": messages}


def _file_body(name: str) -> str:
    if name.endswith(".py"):
        return "def main():\n    print('hello from " + name + "')\n\n\nif __name__ == '__main__':\n    main()\n"
    if name.endswith(".md"):
        return "# Project\n\nQuick start:\n\n    pip install -e .\n"
    if name.endswith(".json"):
        return '{\n  "name": "app",\n  "version": "1.0.0"\n}\n'
    if name.endswith(".yaml"):
        return "debug: false\nport: 8080\n"
    if name.endswith(".sh"):
        return "#!/usr/bin/env bash\nset -euo pipefail\necho deploying\n"
    return "TODO: fill in\n"


def build_agent_dataset(n: int = 160, seed: int = 7) -> list[dict[str, Any]]:
    """Return ``n`` deterministic agent traces (chat records with tool calls)."""
    rng = random.Random(seed)
    out: list[dict[str, Any]] = []

    generators = [_gen_create, _gen_read, _gen_edit, _gen_delete, _gen_list, _gen_run, _gen_web, _gen_open, _gen_storage]
    while len(out) < n:
        roll = rng.random()
        if roll < 0.25:
            q, a = rng.choice(_QA)
            out.append(_record([("user", q), ("assistant", a)]))
        else:
            out.append(rng.choice(generators)(rng))
    return out[:n]


def _gen_create(rng: random.Random) -> dict[str, Any]:
    name = rng.choice(_FILES)
    body = _file_body(name)
    empty = rng.random() < 0.3
    ask = rng.choice([f"create {name}", f"make a new file called {name}", f"add {name} to the project"])
    if empty:
        ask += " (empty is fine)"
        body = ""
    call = {"name": "create_file", "path": name, "content": body}
    return _record([
        ("user", ask),
        ("assistant", _fence(call)),
        ("tool", f"create_file ok: wrote {len(body)} bytes to {name}"),
        ("assistant", f"Created `{name}`." + ("" if empty else " Let me know if you want different contents.")),
    ])


def _gen_read(rng: random.Random) -> dict[str, Any]:
    name = rng.choice(_FILES)
    body = _file_body(name)
    ask = rng.choice([f"what's in {name}?", f"show me {name}", f"read {name} and summarise it"])
    return _record([
        ("user", ask),
        ("assistant", _fence({"name": "read_file", "path": name})),
        ("tool", f"read_file ok ({len(body)} bytes):\n{body}"),
        ("assistant", f"`{name}` contains {len(body.splitlines())} lines:\n\n```\n{body.rstrip()}\n```"),
    ])


def _gen_open(rng: random.Random) -> dict[str, Any]:
    name = rng.choice(_FILES)
    return _record([
        ("user", rng.choice([f"open {name}", f"open up {name} for me"])),
        ("assistant", _fence({"name": "open_file", "path": name})),
        ("tool", f"open_file ok: {name}"),
        ("assistant", f"Opened `{name}`."),
    ])


def _gen_edit(rng: random.Random) -> dict[str, Any]:
    name = rng.choice([f for f in _FILES if f.endswith(".py")])
    old, new = "print('hello from " + name + "')", "print('hello, world')"
    return _record([
        ("user", rng.choice([f"in {name}, change the greeting to say hello, world", f"update the print in {name} to 'hello, world'"])),
        ("assistant", _fence({"name": "read_file", "path": name})),
        ("tool", f"read_file ok:\n{_file_body(name)}"),
        ("assistant", _fence({"name": "edit_file", "path": name, "old": old, "new": new})),
        ("tool", f"edit_file ok: 1 replacement in {name}"),
        ("assistant", f"Updated `{name}`: the greeting now prints `hello, world`."),
    ])


def _gen_delete(rng: random.Random) -> dict[str, Any]:
    name = rng.choice(["notes.txt", "scratch.tmp", "old_backup.py", "debug.log"])
    return _record([
        ("user", rng.choice([f"delete {name}", f"remove {name}, we don't need it"])),
        ("assistant", _fence({"name": "delete_file", "path": name})),
        ("tool", f"delete_file ok: removed {name}"),
        ("assistant", f"Deleted `{name}`."),
    ])


def _gen_list(rng: random.Random) -> dict[str, Any]:
    d = rng.choice(_DIRS)
    entries = rng.sample(_FILES, 3)
    listing = "\n".join(entries)
    where = "the workspace root" if d == "." else f"`{d}`"
    return _record([
        ("user", rng.choice([f"what files are in {d}?", f"list {d}", f"ls {d}"])),
        ("assistant", _fence({"name": "list_dir", "path": d})),
        ("tool", f"list_dir ok:\n{listing}"),
        ("assistant", f"{where} contains:\n\n" + "\n".join(f"- `{e}`" for e in entries)),
    ])


def _gen_run(rng: random.Random) -> dict[str, Any]:
    ask, cmd, output = rng.choice(_COMMANDS)
    return _record([
        ("user", rng.choice([ask, f"can you {ask}?", f"please {ask}"])),
        ("assistant", _fence({"name": "run_command", "command": cmd})),
        ("tool", f"run_command ok (exit 0):\n{output}"),
        ("assistant", f"Ran `{cmd}`:\n\n```\n{output}\n```"),
    ])


def _gen_web(rng: random.Random) -> dict[str, Any]:
    url = rng.choice(_URLS)
    return _record([
        ("user", rng.choice([f"fetch {url}", f"what does {url} return?"])),
        ("assistant", _fence({"name": "web_request", "url": url, "method": "GET"})),
        ("tool", 'web_request ok (200):\n{"status": "ok"}'),
        ("assistant", f"`{url}` responded with HTTP 200:\n\n```\n{{\"status\": \"ok\"}}\n```"),
    ])


def _gen_storage(rng: random.Random) -> dict[str, Any]:
    free = rng.choice([12.4, 87.1, 240.0])
    return _record([
        ("user", rng.choice(["how much disk space is left?", "check storage"])),
        ("assistant", _fence({"name": "check_storage"})),
        ("tool", f"check_storage ok: {free} GB free"),
        ("assistant", f"You have about **{free} GB** free."),
    ])


# --------------------------------------------------------------------------- preference (DPO)


def _prompt_text(task: str, system: str = AGENT_SYSTEM_PROMPT) -> str:
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{task}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def build_agent_preference_dataset(n: int = 80, seed: int = 11) -> list[dict[str, Any]]:
    """Return ``n`` DPO preference pairs for agent tool-calling.

    Each record is ``{"prompt", "chosen", "rejected"}`` where ``chosen`` is a correct tool call
    (or correct prose answer for no-tool questions) and ``rejected`` is a plausible failure:
    wrong tool, missing argument, prose-only answer when a tool was needed, or a phantom call
    when none was needed.
    """
    rng = random.Random(seed)
    pairs: list[dict[str, Any]] = []

    while len(pairs) < n:
        kind = rng.randrange(5)
        if kind == 0:
            name = rng.choice(_FILES)
            task = rng.choice([f"create {name}", f"make a file called {name}"])
            good = _fence({"name": "create_file", "path": name, "content": _file_body(name)})
            bads = [
                f"Sure! The file {name} has been created.",
                _fence({"name": "create_file", "path": rng.choice(_FILES)}),
                _fence({"name": "read_file", "path": name}),
            ]
        elif kind == 1:
            ask, cmd, _ = rng.choice(_COMMANDS)
            task = f"can you {ask}?"
            good = _fence({"name": "run_command", "command": cmd})
            bads = [
                f"To do that, run `{cmd}` in your terminal.",
                _fence({"name": "run_command"}),
                _fence({"name": "list_dir", "path": "."}),
            ]
        elif kind == 2:
            name = rng.choice(_FILES)
            task = rng.choice([f"what's in {name}?", f"read {name}"])
            good = _fence({"name": "read_file", "path": name})
            bads = [
                _fence({"name": "delete_file", "path": name}),
                _fence({"name": "read_file"}),
                f"Here's what {name} contains:\n\n{_file_body(name)}",
            ]
        elif kind == 3:
            name = rng.choice(["notes.txt", "scratch.tmp", "debug.log"])
            task = f"delete {name}"
            good = _fence({"name": "delete_file", "path": name})
            bads = [
                f"I've deleted {name} for you.",
                _fence({"name": "open_file", "path": name}),
            ]
        else:
            q, a = rng.choice(_QA[:5])
            task = q
            good = a
            bads = [
                _fence({"name": "web_request", "url": "https://google.com", "method": "GET"}),
                _fence({"name": "run_command", "command": "echo answer"}),
            ]
        pairs.append({"prompt": _prompt_text(task), "chosen": good, "rejected": rng.choice(bads)})
    return pairs[:n]


# --------------------------------------------------------------------------- validation


def validate_agent_records(records: list[Any]) -> tuple[list[dict[str, Any]], list[str]]:
    """Validate chat records for agent SFT.

    Returns ``(valid_records, issues)``. A record is valid when it has a ``messages`` list with
    at least one user and one assistant turn, and every ```json fence emitted by the assistant
    parses to a known tool call. Legacy ``{input, output}`` pairs are converted to chat records.
    """
    from inferforge.agent.tools import parse_tool_calls

    valid: list[dict[str, Any]] = []
    issues: list[str] = []
    for i, rec in enumerate(records):
        if not isinstance(rec, dict):
            issues.append(f"record {i}: not an object")
            continue
        if "messages" not in rec and "input" in rec and "output" in rec:
            rec = {"messages": [
                {"role": "user", "content": str(rec["input"])},
                {"role": "assistant", "content": str(rec["output"])},
            ]}
        msgs = rec.get("messages")
        if not isinstance(msgs, list) or not msgs:
            issues.append(f"record {i}: missing 'messages' list")
            continue
        roles = [str(m.get("role", "")).lower() for m in msgs if isinstance(m, dict)]
        if "user" not in roles or "assistant" not in roles:
            issues.append(f"record {i}: needs at least one user and one assistant turn")
            continue
        bad = False
        for j, m in enumerate(msgs):
            if str(m.get("role", "")).lower() != "assistant":
                continue
            content = str(m.get("content", ""))
            calls = parse_tool_calls(content)
            # Only flag ```json fences that look like tool calls (name/tool/action key present)
            # but failed to parse — display-only JSON blocks are fine.
            for fence in re.findall(r"```json\s*(\{.*?\})\s*```", content, re.DOTALL | re.IGNORECASE):
                try:
                    obj = json.loads(fence)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict) and any(k in obj for k in ("name", "tool", "action")):
                    calls.append(obj)
            if re.search(r"```json", content, re.IGNORECASE) and not calls:
                issues.append(f"record {i}, turn {j}: json fence is not a recognised tool call")
                bad = True
                break
            for call in calls:
                if call["name"] not in CANONICAL_TOOLS:
                    issues.append(f"record {i}, turn {j}: unknown tool '{call['name']}'")
                    bad = True
                    break
                if call["name"] in {"create_file", "edit_file", "delete_file", "read_file", "open_file", "list_dir"} and not call.get("path"):
                    issues.append(f"record {i}, turn {j}: {call['name']} needs a 'path'")
                    bad = True
                    break
                if call["name"] == "run_command" and not call.get("command"):
                    issues.append(f"record {i}, turn {j}: run_command needs a 'command'")
                    bad = True
                    break
            if bad:
                break
        if not bad:
            valid.append(rec)
    return valid, issues
