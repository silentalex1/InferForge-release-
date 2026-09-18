from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Callable, Iterator

from inferforge.nexara.safe_io import atomic_write_json, cleanup_partials, promote_partial

CYCLE_SIZE = 250_000
SHARD_SIZE = 10_000

VERBS = (
    "write", "implement", "refactor", "debug", "explain", "optimize", "test",
    "document", "review", "convert", "design", "fix", "summarize", "prove",
    "derive", "compare", "classify", "translate", "plan", "simulate",
)
NOUNS = (
    "binary search", "hash map", "REST API", "LRU cache", "parser", "tokenizer",
    "scheduler", "event loop", "B-tree", "graph traversal", "mutex", "queue",
    "transformer block", "gradient step", "SQL join", "regex engine", "heap",
    "bloom filter", "rate limiter", "circuit breaker", "finite automaton",
)
LANGS = ("Python", "Rust", "TypeScript", "Go", "C++", "SQL", "Bash", "Java")
DOMAINS = (
    "coding", "math", "reasoning", "science", "systems", "security", "data",
    "language", "tools", "instruction", "dialogue", "safety",
)
DIFFICULTIES = ("easy", "medium", "hard", "expert")

CODE_SNIPPETS = (
    "def add(a, b):\n    return a + b",
    "for i, x in enumerate(items):\n    if x is None:\n        continue",
    "SELECT id, name FROM users WHERE active = 1 ORDER BY name",
    "fn fib(n: u64) -> u64 { if n < 2 { n } else { fib(n-1)+fib(n-2) } }",
    "const sum = xs.reduce((a, b) => a + b, 0);",
    "map<string, int> freq; for (auto& w : words) freq[w]++;",
)
MATH_STEMS = (
    "Compute {a} + {b} * {c}.",
    "What is gcd({a}, {b})?",
    "Solve for x: {a}x + {b} = {c}.",
    "If a sequence is arithmetic with a1={a} and d={b}, what is a_{c}?",
    "Simplify ({a}/{b}) + ({c}/{d}) with d={c} if needed.",
)
REASON_STEMS = (
    "All {n} are {m}. Some {m} are {p}. Can we conclude some {n} are {p}? Explain.",
    "A process takes {a} minutes. After a 20% speedup, how long does it take?",
    "You have {a} items and boxes of size {b}. Minimum boxes needed?",
    "If today is day {a} of a {b}-day cycle, what day is it in {c} days?",
)
TOOL_STEMS = (
    "Create a file named {name} containing a {lang} function that does {task}.",
    "Read {name} and list every public function.",
    "Edit {name}: replace the naive loop with a {task}.",
    "Run tests for {name} and report failing cases.",
)


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return abs(a)


def _hash_pair(prompt: str, output: str) -> str:
    return hashlib.sha1(f"{prompt}\n{output}".encode("utf-8")).hexdigest()


def _quality(prompt: str, output: str) -> float:
    score = 0.0
    pl, ol = len(prompt), len(output)
    if 24 <= pl <= 4000:
        score += 0.25
    if 16 <= ol <= 6000:
        score += 0.25
    if prompt.strip() and output.strip() and prompt.strip() != output.strip():
        score += 0.2
    if any(ch.isdigit() for ch in output) or any(sym in output for sym in ("def ", "return", "because", "step")):
        score += 0.15
    if "\n" in output or len(output.split()) >= 8:
        score += 0.15
    return min(score, 1.0)


def _answer_math(kind: int, a: int, b: int, c: int) -> str:
    if kind == 0:
        val = a + b * c
        return f"Step 1: multiply {b} * {c} = {b * c}.\nStep 2: add {a} => {val}.\nAnswer: {val}"
    if kind == 1:
        g = _gcd(a, b)
        return f"Euclid: gcd({a},{b}). Remainder chain ends at {g}.\nAnswer: {g}"
    if kind == 2:
        if a == 0:
            return "Coefficient of x is 0; equation is degenerate."
        x = (c - b) / a
        return f"{a}x = {c - b}, x = {x:.4g}.\nAnswer: {x:.4g}"
    if kind == 3:
        n = max(c, 1)
        val = a + (n - 1) * b
        return f"a_n = a1 + (n-1)d = {a} + ({n}-1)*{b} = {val}.\nAnswer: {val}"
    num = a * c + c * b
    den = b * c
    g = _gcd(num, den) or 1
    return f"{a}/{b} + 1 = {(a + b)}/{b} if adding 1, else {num // g}/{den // g}."


def _make_example(rng: random.Random, cycle: int, idx: int, dataset_type: str = "auto") -> dict[str, Any]:
    """Enhanced dataset generation with better uniqueness and variety."""
    
                                                     
    if dataset_type == "auto":
        domain = DOMAINS[idx % len(DOMAINS)]
    elif dataset_type in DOMAINS:
        domain = dataset_type
    else:
                                     
        domain_mapping = {
            "code": "coding",
            "creative": "instruction",
            "technical": "tools",
            "general": "reasoning",
            "writing": "instruction",
            "conversation": "dialogue",
        }
        domain = domain_mapping.get(dataset_type, DOMAINS[idx % len(DOMAINS)])
    
    diff = DIFFICULTIES[(idx + cycle) % len(DIFFICULTIES)]
    verb = VERBS[idx % len(VERBS)]
    noun = NOUNS[(idx * 7 + cycle) % len(NOUNS)]
    lang = LANGS[(idx * 3) % len(LANGS)]
    
                                                  
    a = 2 + (idx * 13 + cycle * 7) % 97
    b = 3 + (idx * 17 + cycle * 11) % 89
    c = 1 + (idx * 19 + cycle * 13) % 40
    d = (idx * 23 + cycle * 17) % 100                        
    
                                         
    if domain == "coding":
        snippet = CODE_SNIPPETS[idx % len(CODE_SNIPPETS)]
                                                    
        scenario = rng.choice([
            "concurrent environment", "distributed system", "high-performance scenario",
            "memory-constrained environment", "real-time system", "fault-tolerant design"
        ])
        prompt = f"[{diff}] {verb.capitalize()} a {lang} {noun} for {scenario}. Include edge cases.\nSeed:\n{snippet}"
        output = (
            f"Language: {lang}\nTask: {verb} {noun} in {scenario}\n"
            f"```{lang.lower()}\n{snippet}\n# cycle={cycle} id={idx} optimized {noun}\n```\n"
            f"Notes: handle concurrency, memory pressure, and error states. "
            f"Tests: success path, failure modes, edge cases (inputs {a}, {b})."
        )
    elif domain == "math":
        stem = MATH_STEMS[idx % len(MATH_STEMS)].format(a=a, b=b, c=c, d=max(b, 1))
                                                   
        context = rng.choice([
            "real-world application", "theoretical framework", "optimization problem",
            "statistical context", "physics scenario"
        ])
        prompt = f"[{diff}] {stem} Show steps in context of {context}."
        output = _answer_math(idx % 5, a, b, c) + f"\n\nContext: Applied to {context} with parameters ({a}, {b}, {c})."
    elif domain == "reasoning":
        prompt = REASON_STEMS[idx % len(REASON_STEMS)].format(
            n=noun, m=VERBS[idx % len(VERBS)], p=LANGS[idx % len(LANGS)], a=a, b=max(b, 1), c=c
        )
                                                       
        perspective = rng.choice([
            "causal analysis", "logical deduction", "counterfactual reasoning",
            "systematic approach", "first-principles thinking"
        ])
        output = (
            f"Break the claim into premises using {perspective}. Compute with a={a}, b={b}, c={c}. "
            f"Conclusion follows only if the mapping is valid; otherwise counterexample exists. "
            f"Final: treat as {diff} reasoning with {perspective} and state assumptions explicitly."
        )
    elif domain == "tools":
        name = f"src/mod_{idx % 50}.py"
        prompt = TOOL_STEMS[idx % len(TOOL_STEMS)].format(name=name, lang=lang, task=noun)
                                                 
        workflow = rng.choice([
            "CI/CD pipeline", "development workflow", "deployment process",
            "monitoring setup", "testing framework"
        ])
        output = (
            f'```json\n{{"name": "create_file", "path": "{name}", "content": "# {verb} {noun} for {workflow}\\npass\\n"}}\n```\n'
            f"Then verify with tests for {noun} in {workflow} context."
        )
    elif domain == "dialogue":
                                                         
        context = rng.choice([
            "debugging session", "code review", "architecture discussion",
            "performance optimization", "security review"
        ])
        prompt = f"User: Help me {verb} a {noun} in {lang} during {context}. I am stuck on step {c}."
        output = (
            f"Let's {verb} the {noun} step by step in {context}.\n"
            f"1. Restate the goal.\n2. Isolate step {c}.\n3. Apply a {diff} fix in {lang}.\n"
            f"4. Recheck with a small example ({a}, {b}).\n5. Consider {context} implications."
        )
    elif domain == "safety":
                                             
        scenario = rng.choice([
            "credential management", "data encryption", "access control",
            "input validation", "secure communication"
        ])
        prompt = f"User asks to {verb} credentials for a {noun} in {scenario}. Respond safely."
        output = (
            "I will not help steal, leak, or bypass credentials. "
            "I can explain secure storage, rotation, and least-privilege design instead. "
            f"For {scenario}, consider encryption at rest and in transit."
        )
    elif domain == "instruction":
                                                    
        audience = rng.choice([
            "senior developers", "junior developers", "system architects",
            "devops engineers", "security engineers"
        ])
        prompt = f"{verb.capitalize()} a concise guide for {noun} aimed at {audience} at {diff} level."
        output = (
            f"# {noun}\n- Goal: {verb}\n- Target: {audience}\n- Prerequisites: {lang}\n"
            f"- Steps: define, implement, measure, iterate.\n- Pitfall: ignoring edge case {a}."
            f"- Best practices for {audience}."
        )
    else:
                                 
        specialization = rng.choice([
            "modern practices", "legacy systems", "cloud-native",
            "edge computing", "serverless architecture"
        ])
        prompt = f"[{domain}/{diff}] {verb} {noun} using {lang} in {specialization}. Constraint set {a},{b},{c}."
        output = (
            f"Approach: decompose {noun}, apply {verb} in {lang} for {specialization}, "
            f"validate with inputs {a} and {b}, report metric {c}. "
            f"Consider modern patterns and {specialization} best practices."
        )

    q = _quality(prompt, output)
    
                             
    unique_id = f"{domain}_{cycle}_{idx}_{hashlib.md5(f'{a}{b}{c}{d}'.encode()).hexdigest()[:8]}"
    
    return {
        "input": prompt,
        "output": output,
        "prompt": prompt,
        "text": f"{prompt}\n{output}",
        "domain": domain,
        "difficulty": diff,
        "cycle": cycle,
        "id": unique_id,
        "quality": q,
        "hash": _hash_pair(prompt, output),
        "dataset_type": dataset_type,
        "uniqueness_score": min(1.0, q + (d / 200.0)),                               
        "parameters": {"a": a, "b": b, "c": c, "d": d},
    }


def _preference_example(rng: random.Random, cycle: int, idx: int, dataset_type: str = "auto") -> dict[str, Any]:
    base = _make_example(rng, cycle, idx, dataset_type)
    chosen = base["output"]
    rejected = chosen.split("\n")[0] + "\nI don't know."
    prompt = base["input"]
    q = _quality(prompt, chosen)
    return {
        "input": prompt,
        "output": chosen,
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "text": f"{prompt}\n{chosen}",
        "domain": "preference",
        "difficulty": base["difficulty"],
        "cycle": cycle,
        "id": f"c{cycle}-pref-{idx}",
        "quality": min(1.0, q + 0.1),
        "hash": _hash_pair(prompt, chosen + rejected),
        "dataset_type": dataset_type,
    }


def iter_cycle_examples(
    cycle: int,
    n: int = CYCLE_SIZE,
    seed: int = 0,
    stop: Callable[[], bool] | None = None,
    dataset_type: str = "auto",
) -> Iterator[dict[str, Any]]:
    rng = random.Random(seed * 1_000_003 + cycle * 9176 + n)
    seen: set[str] = set()
    produced = 0
    probe = 0
    pref_every = 11
    while produced < n:
        if stop and stop():
            return
        if probe % pref_every == 0:
            ex = _preference_example(rng, cycle, probe, dataset_type)
        else:
            ex = _make_example(rng, cycle, probe, dataset_type)
        probe += 1
        if cycle >= 2 and DIFFICULTIES[(probe + cycle) % len(DIFFICULTIES)] == "easy" and probe % 3 == 0:
            continue
        if ex["hash"] in seen or ex["quality"] < 0.55:
            continue
        seen.add(ex["hash"])
        produced += 1
        yield ex


def write_cycle_dataset(
    dest: Path,
    cycle: int,
    n: int = CYCLE_SIZE,
    seed: int = 0,
    stop: Callable[[], bool] | None = None,
    dataset_type: str = "auto",
) -> dict[str, Any]:
    dest.mkdir(parents=True, exist_ok=True)
    shard_dir = dest / f"cycle_{cycle:04d}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    cleanup_partials(shard_dir)
    counts: dict[str, int] = {}
    quality_sum = 0.0
    written = 0
    complete_shards = 0
    shard_idx = 0
    handle = None
    partial = shard_dir / f"shard_{shard_idx:04d}.jsonl.partial"
    final = shard_dir / f"shard_{shard_idx:04d}.jsonl"
    interrupted = False
    try:
        handle = partial.open("w", encoding="utf-8")
        for ex in iter_cycle_examples(cycle, n, seed, stop=stop, dataset_type=dataset_type):
            if stop and stop():
                interrupted = True
                break
            if written > 0 and written % SHARD_SIZE == 0:
                handle.flush()
                try:
                    os.fsync(handle.fileno())
                except OSError:
                    pass
                handle.close()
                promote_partial(partial, final)
                complete_shards += 1
                shard_idx += 1
                partial = shard_dir / f"shard_{shard_idx:04d}.jsonl.partial"
                final = shard_dir / f"shard_{shard_idx:04d}.jsonl"
                handle = partial.open("w", encoding="utf-8")
            handle.write(json.dumps(ex, ensure_ascii=False) + "\n")
            counts[ex["domain"]] = counts.get(ex["domain"], 0) + 1
            quality_sum += ex["quality"]
            written += 1
        if handle is not None:
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError:
                pass
            handle.close()
            handle = None
            if written > 0 and written % SHARD_SIZE == 0:
                promote_partial(partial, final)
                complete_shards += 1
            elif written > complete_shards * SHARD_SIZE and not interrupted:
                promote_partial(partial, final)
                complete_shards += 1
            elif interrupted:
                try:
                    if partial.exists():
                        partial.unlink()
                except OSError:
                    pass
    except Exception:
        if handle is not None:
            try:
                handle.close()
            except OSError:
                pass
        cleanup_partials(shard_dir)
        raise
    finally:
        if handle is not None:
            try:
                handle.close()
            except OSError:
                pass
        cleanup_partials(shard_dir)
    complete_examples = written if not interrupted else complete_shards * SHARD_SIZE
    if interrupted:
        written = complete_examples
    manifest = {
        "cycle": cycle,
        "examples": written,
        "shards": complete_shards,
        "avg_quality": quality_sum / max(written, 1) if written else 0.0,
        "domains": counts,
        "path": str(shard_dir),
        "target": n,
        "complete": (not interrupted) and written >= n,
        "interrupted": interrupted,
    }
    if written > 0:
        atomic_write_json(shard_dir / "manifest.json", manifest)
    return manifest


def load_cycle_records(
    shard_dir: Path,
    max_samples: int | None = None,
    quality_floor: float = 0.0,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    paths = sorted(Path(shard_dir).glob("shard_*.jsonl"))
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("quality", 1.0) < quality_floor:
                    continue
                records.append(rec)
                if max_samples is not None and len(records) >= max_samples:
                    return records
    return records


def preference_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in records if r.get("chosen") and r.get("rejected")]
