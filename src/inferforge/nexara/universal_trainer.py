from __future__ import annotations

import json
import math
import os
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from inferforge.nexara.alignment import ALIGNMENT_METHODS, AlignmentConfig, AlignmentTrainer
from inferforge.nexara.architectures import (
    count_parameters,
    load_any_model,
)
from inferforge.nexara.scaling import (
    compute_optimal_lr,
    fit_recipe_to_hardware,
    get_recipe,
)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    F = None
    TORCH_AVAILABLE = False

ProgressCb = Callable[[str, float, dict[str, Any] | None], None]

CHAT_ROLES = {"system", "user", "human", "assistant", "gpt", "model", "tool"}


@dataclass
class UniversalTrainConfig:
    output_dir: str = "./nexara_runs"
    model: str | None = None
    from_scratch: bool = False
    scale: str = "tiny"
    params: int | None = None
    task: str = "causal_lm"
    stage: str = "auto"
    data: str | list | None = None
    eval_data: str | list | None = None
    text_field: str = "text"
    max_samples: int | None = None
    seq_len: int | None = None
    epochs: int = 1
    max_steps: int | None = None
    batch_size: int | None = None
    gradient_accumulation_steps: int | None = None
    learning_rate: float | None = None
    min_learning_rate: float | None = None
    weight_decay: float = 0.1
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    peft_method: str = "auto"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] | None = None
    bits: int | None = None
    packing: bool = True
    gradient_checkpointing: bool | None = None
    bf16: bool | None = None
    fp16: bool | None = None
    seed: int = 42
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3
    resume: str | None = None
    trust_remote_code: bool = True
    alignment_method: str = "dpo"
    alignment: dict[str, Any] = field(default_factory=dict)
    architecture: dict[str, Any] = field(default_factory=dict)
    num_workers: int = 0
    report_to: str = "none"
    time_budget_minutes: float | None = None
    train_on_completions_only: bool = True
    early_stopping_patience: int | None = None
    max_nan_skips: int = 50

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UniversalTrainConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in known})


def detect_record_schema(record: dict[str, Any]) -> str:
    keys = {k.lower() for k in record.keys()}
    if {"chosen", "rejected"} <= keys or {"chosen", "rejected", "prompt"} <= keys:
        return "preference"
    if "messages" in keys or "conversations" in keys:
        return "chat"
    if "instruction" in keys and "output" in keys:
        return "alpaca"
    if "input" in keys and "output" in keys:
        return "io"
    if "prompt" in keys and ("completion" in keys or "response" in keys):
        return "prompt_completion"
    if "text" in keys or "content" in keys:
        return "text"
    return "text"


def render_chat(messages: list[dict[str, Any]]) -> str:
    parts = []
    for msg in messages:
        role = str(msg.get("role") or msg.get("from") or "user").lower()
        if role in {"human", "user"}:
            role = "user"
        elif role in {"gpt", "assistant", "model"}:
            role = "assistant"
        content = msg.get("content") or msg.get("value") or msg.get("text") or ""
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    if parts and not parts[-1].startswith("<|im_start|>assistant"):
        parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


def record_to_text(record: Any) -> str:
    if isinstance(record, str):
        return record
    if not isinstance(record, dict):
        return str(record)
    schema = detect_record_schema(record)
    if schema == "chat":
        msgs = record.get("messages") or record.get("conversations") or []
        return render_chat(msgs)
    if schema == "alpaca":
        inst = record.get("instruction", "")
        inp = record.get("input") or ""
        out = record.get("output") or record.get("response") or ""
        if inp:
            return f"<|im_start|>user\n{inst}\n{inp}<|im_end|>\n<|im_start|>assistant\n{out}<|im_end|>"
        return f"<|im_start|>user\n{inst}<|im_end|>\n<|im_start|>assistant\n{out}<|im_end|>"
    if schema == "io":
        return f"<|im_start|>user\n{record.get('input','')}<|im_end|>\n<|im_start|>assistant\n{record.get('output','')}<|im_end|>"
    if schema == "prompt_completion":
        return f"{record.get('prompt','')}{record.get('completion') or record.get('response','')}"
    if schema == "preference":
        prompt = record.get("prompt") or record.get("input") or ""
        chosen = record.get("chosen") or record.get("response") or ""
        if isinstance(chosen, list):
            chosen = render_chat(chosen)
        return f"{prompt}{chosen}"
    return str(record.get("text") or record.get("content") or record.get("output") or "")


def record_to_prompt_completion(record: Any) -> tuple[str, str]:
    """Split a record into (prompt, completion) so SFT loss can be applied to the completion only.

    Records without a natural split (plain text) return ("", text) so every token is trained on.
    """
    if not isinstance(record, dict):
        return "", record_to_text(record)
    schema = detect_record_schema(record)
    if schema == "chat":
        msgs = list(record.get("messages") or record.get("conversations") or [])
        last_assistant = None
        for idx in range(len(msgs) - 1, -1, -1):
            role = str(msgs[idx].get("role") or msgs[idx].get("from") or "").lower()
            if role in {"assistant", "gpt", "model"}:
                last_assistant = idx
                break
        if last_assistant is None:
            return "", render_chat(msgs)
        prompt = render_chat(msgs[:last_assistant])
        final = msgs[last_assistant]
        content = final.get("content") or final.get("value") or final.get("text") or ""
        return prompt + "\n" if prompt else "", f"{content}<|im_end|>"
    if schema in {"alpaca", "io"}:
        text = record_to_text(record)
        marker = "<|im_start|>assistant\n"
        pos = text.rfind(marker)
        if pos < 0:
            return "", text
        cut = pos + len(marker)
        return text[:cut], text[cut:]
    if schema == "prompt_completion":
        return str(record.get("prompt", "")), str(record.get("completion") or record.get("response", ""))
    if schema == "preference":
        prompt = record.get("prompt") or record.get("input") or ""
        chosen = record.get("chosen") or record.get("response") or ""
        if isinstance(chosen, list):
            chosen = render_chat(chosen)
        return str(prompt), str(chosen)
    return "", record_to_text(record)


def record_to_segments(record: Any) -> list[tuple[str, bool]]:
    """Split a record into ``(text, trainable)`` segments.

    For chat records every assistant turn is trainable and everything else (system, user, tool
    results) is context only, so multi-turn agent traces train on *all* assistant actions.
    Other schemas fall back to a single (prompt, completion) split.
    """
    if isinstance(record, dict) and detect_record_schema(record) == "chat":
        msgs = record.get("messages") or record.get("conversations") or []
        segments: list[tuple[str, bool]] = []
        for msg in msgs:
            role = str(msg.get("role") or msg.get("from") or "user").lower()
            if role in {"human", "user"}:
                role = "user"
            elif role in {"gpt", "assistant", "model"}:
                role = "assistant"
            content = msg.get("content") or msg.get("value") or msg.get("text") or ""
            if role == "assistant":
                segments.append((f"<|im_start|>assistant\n", False))
                segments.append((f"{content}<|im_end|>\n", True))
            else:
                segments.append((f"<|im_start|>{role}\n{content}<|im_end|>\n", False))
        if segments and not any(train for _, train in segments):
            # No assistant turn: nothing to learn from as SFT, train on everything instead.
            return [("".join(t for t, _ in segments), True)]
        return segments
    prompt, completion = record_to_prompt_completion(record)
    segments = []
    if prompt:
        segments.append((prompt, False))
    segments.append((completion, True))
    return segments


def load_raw_records(source: str | list | None, max_samples: int | None = None) -> list[Any]:
    if source is None:
        return []
    if isinstance(source, list):
        recs = source
        return recs[:max_samples] if max_samples else recs
    path = Path(str(source))
    recs: list[Any] = []
    if path.exists():
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        recs.append(json.loads(line))
                    except json.JSONDecodeError:
                        recs.append({"text": line})
        elif suffix == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                if "data" in data:
                    recs = data["data"]
                elif "examples" in data:
                    recs = data["examples"]
                else:
                    recs = [data]
            else:
                recs = data
        elif suffix in {".txt", ".md"}:
            recs = [{"text": block.strip()} for block in path.read_text(encoding="utf-8").split("\n\n") if block.strip()]
        elif suffix == ".csv":
            try:
                import csv
                with path.open("r", encoding="utf-8", newline="") as f:
                    recs = list(csv.DictReader(f))
            except Exception:
                recs = [{"text": path.read_text(encoding="utf-8")}]
        elif suffix == ".parquet":
            try:
                import pandas as pd
                recs = pd.read_parquet(path).to_dict(orient="records")
            except Exception as exc:
                raise ValueError(f"Failed to read parquet: {exc}") from exc
        else:
            recs = [{"text": path.read_text(encoding="utf-8")}]
    else:
        try:
            from datasets import load_dataset
            ds = load_dataset(str(source), split="train")
            recs = [ds[i] for i in range(len(ds))]
        except Exception as exc:
            raise FileNotFoundError(f"Data source not found: {source}") from exc
    if max_samples is not None:
        recs = recs[:max_samples]
    return recs


def records_are_preference(records: list[Any]) -> bool:
    if not records:
        return False
    sample = records[0]
    return isinstance(sample, dict) and detect_record_schema(sample) == "preference"


class CharTokenizer:
    def __init__(self, texts: list[str] | None = None, vocab: dict[str, int] | None = None):
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.unk_token = "<unk>"
        if vocab:
            self.stoi = vocab
        else:
            chars = set()
            for t in texts or []:
                chars.update(t)
            specials = [self.pad_token, self.eos_token, self.unk_token]
            alphabet = specials + sorted(chars)
            self.stoi = {ch: i for i, ch in enumerate(alphabet)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.pad_token_id = self.stoi[self.pad_token]
        self.eos_token_id = self.stoi[self.eos_token]
        self.unk_token_id = self.stoi[self.unk_token]
        self.vocab_size = len(self.stoi)

    def encode(self, text: str, add_eos: bool = True) -> list[int]:
        ids = [self.stoi.get(ch, self.unk_token_id) for ch in text]
        if add_eos:
            ids.append(self.eos_token_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos.get(int(i), "") for i in ids if int(i) != self.pad_token_id)

    def __call__(self, text, truncation=True, max_length=1024, padding="max_length", return_tensors=None):
        if isinstance(text, list):
            encs = [self.encode(t) for t in text]
            if truncation:
                encs = [e[:max_length] for e in encs]
            if padding == "max_length":
                encs = [e + [self.pad_token_id] * (max_length - len(e)) for e in encs]
            if return_tensors == "pt" and TORCH_AVAILABLE:
                return {"input_ids": torch.tensor(encs, dtype=torch.long), "attention_mask": torch.tensor([[int(x != self.pad_token_id) for x in e] for e in encs])}
            return {"input_ids": encs}
        ids = self.encode(text)
        if truncation:
            ids = ids[:max_length]
        if padding == "max_length":
            ids = ids + [self.pad_token_id] * (max_length - len(ids))
        if return_tensors == "pt" and TORCH_AVAILABLE:
            return {"input_ids": torch.tensor([ids], dtype=torch.long)}
        return {"input_ids": ids}

    def save_pretrained(self, path: str) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        (p / "char_tokenizer.json").write_text(json.dumps({"stoi": self.stoi}), encoding="utf-8")

    @classmethod
    def from_pretrained(cls, path: str) -> "CharTokenizer":
        data = json.loads((Path(path) / "char_tokenizer.json").read_text(encoding="utf-8"))
        return cls(vocab=data["stoi"])


def load_tokenizer(model_name: str | None, texts: list[str] | None = None, from_scratch: bool = False):
    if not from_scratch and model_name:
        try:
            from transformers import AutoTokenizer
            try:
                tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=True, local_files_only=True)
            except Exception:
                tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=True)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token or tok.unk_token or "[PAD]"
            return tok
        except Exception:
            pass
    fallbacks = ["gpt2", "facebook/opt-125m", "hf-internal-testing/tiny-random-gpt2"]
    if not from_scratch:
        for name in fallbacks:
            try:
                from transformers import AutoTokenizer
                tok = AutoTokenizer.from_pretrained(name)
                if tok.pad_token is None:
                    tok.pad_token = tok.eos_token
                return tok
            except Exception:
                continue
    return CharTokenizer(texts or ["hello world"])


def _encode_ids(tokenizer, text: str) -> list[int]:
    if not text:
        return []
    if hasattr(tokenizer, "encode"):
        try:
            ids = tokenizer.encode(text, add_special_tokens=False)
        except TypeError:
            ids = tokenizer.encode(text)
    else:
        ids = tokenizer(text)["input_ids"]
        if ids and isinstance(ids[0], list):
            ids = ids[0]
    return list(ids)


def tokenize_records(
    records: list[Any],
    tokenizer,
    seq_len: int,
    packing: bool = True,
    completions_only: bool = False,
) -> dict[str, list[list[int]]]:
    """Tokenize records into fixed-length training sequences.

    - Attention masks and labels are derived from real sequence lengths, never from token
      equality with the pad id, so EOS is still learned when ``pad_token_id == eos_token_id``.
    - With ``completions_only`` the prompt part of each record gets label ``-100`` so the loss is
      only applied to the assistant/completion tokens (standard SFT practice).
    """
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is None:
        eos = 0
    pad = getattr(tokenizer, "pad_token_id", None)
    if pad is None:
        pad = eos
    all_ids: list[int] = []
    all_labels: list[int] = []
    sequences: list[tuple[list[int], list[int]]] = []
    for rec in records:
        if completions_only:
            ids, labels = [], []
            for text, trainable in record_to_segments(rec):
                seg = _encode_ids(tokenizer, text)
                ids.extend(seg)
                labels.extend(seg if trainable else [-100] * len(seg))
        else:
            ids = _encode_ids(tokenizer, record_to_text(rec))
            labels = list(ids)
        if not ids:
            continue
        if ids[-1] != eos:
            ids = ids + [eos]
            labels = labels + [eos]
        if packing:
            all_ids.extend(ids)
            all_labels.extend(labels)
        else:
            sequences.append((ids[:seq_len], labels[:seq_len]))
    input_ids: list[list[int]] = []
    masks: list[list[int]] = []
    out_labels: list[list[int]] = []

    def emit(ids: list[int], labels: list[int]) -> None:
        n = len(ids)
        if completions_only and all(lbl == -100 for lbl in labels):
            return
        input_ids.append(ids + [pad] * (seq_len - n))
        masks.append([1] * n + [0] * (seq_len - n))
        out_labels.append(labels + [-100] * (seq_len - n))

    if packing and all_ids:
        for i in range(0, len(all_ids), seq_len):
            chunk = all_ids[i : i + seq_len]
            if len(chunk) < 8 and input_ids:
                continue
            emit(chunk, all_labels[i : i + seq_len])
    else:
        for ids, labels in sequences:
            emit(ids, labels)
    if not input_ids:
        input_ids = [[pad] * seq_len]
        masks = [[1] + [0] * (seq_len - 1)]
        out_labels = [[-100] * seq_len]
    return {"input_ids": input_ids, "attention_mask": masks, "labels": out_labels}


def tokenize_preference(records: list[Any], tokenizer, seq_len: int) -> list[dict[str, Any]]:
    pad = getattr(tokenizer, "pad_token_id", 0) or 0
    out = []
    for rec in records:
        if not isinstance(rec, dict):
            continue
        prompt = rec.get("prompt") or rec.get("input") or ""
        chosen = rec.get("chosen") or ""
        rejected = rec.get("rejected") or ""
        if isinstance(chosen, list):
            chosen = render_chat(chosen)
        if isinstance(rejected, list):
            rejected = render_chat(rejected)
        def enc(text: str) -> list[int]:
            try:
                ids = tokenizer.encode(text, add_special_tokens=False)
            except TypeError:
                ids = tokenizer.encode(text)
            return list(ids)[:seq_len]
        p_ids = enc(str(prompt))
        c_ids = (p_ids + enc(str(chosen)))[:seq_len]
        r_ids = (p_ids + enc(str(rejected)))[:seq_len]
        def pad_ids(ids: list[int], p_len: int = len(p_ids)) -> tuple[list[int], list[int], list[int]]:
            attn = [1] * len(ids) + [0] * (seq_len - len(ids))
            labels = [-100] * min(p_len, len(ids)) + ids[min(p_len, len(ids)) :]
            ids = ids + [pad] * (seq_len - len(ids))
            labels = labels + [-100] * (seq_len - len(labels))
            return ids, attn, labels
        ci, ca, cl = pad_ids(c_ids)
        ri, ra, rl = pad_ids(r_ids)
        out.append({
            "chosen_input_ids": ci,
            "chosen_attention_mask": ca,
            "chosen_labels": cl,
            "rejected_input_ids": ri,
            "rejected_attention_mask": ra,
            "rejected_labels": rl,
            "prompt_len": len(p_ids),
        })
    return out


class TensorDataset:
    def __init__(self, data: dict[str, list]):
        self.data = data
        self.keys = list(data.keys())
        self.n = len(next(iter(data.values())))

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = {k: self.data[k][idx] for k in self.keys}
        if TORCH_AVAILABLE:
            item = {k: torch.tensor(v) if not isinstance(v, torch.Tensor) else v for k, v in item.items()}
        return item


def apply_peft(model, method: str, cfg: UniversalTrainConfig):
    method = (method or "lora").lower()
    if method in {"none", "full", "off"}:
        return model
    targets = cfg.lora_target_modules or [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
        "c_attn", "c_proj", "c_fc", "query_key_value", "dense", "fc1", "fc2",
        "w1", "w2", "w3",
    ]
    try:
        from peft import AdaLoraConfig, IA3Config, LoraConfig, TaskType, get_peft_model
        peft_task = TaskType.CAUSAL_LM
        if method in {"dora"}:
            try:
                lcfg = LoraConfig(
                    r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
                    target_modules=targets, task_type=peft_task, bias="none", use_dora=True,
                )
            except TypeError:
                lcfg = LoraConfig(
                    r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
                    target_modules=targets, task_type=peft_task, bias="none",
                )
            model = get_peft_model(model, lcfg)
        elif method in {"adalora"}:
            lcfg = AdaLoraConfig(
                r=cfg.lora_r, lora_alpha=cfg.lora_alpha, target_modules=targets, task_type=peft_task,
            )
            model = get_peft_model(model, lcfg)
        elif method in {"ia3"}:
            lcfg = IA3Config(target_modules=targets, task_type=peft_task)
            model = get_peft_model(model, lcfg)
        else:
            lcfg = LoraConfig(
                r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
                target_modules=targets, task_type=peft_task, bias="none",
            )
            model = get_peft_model(model, lcfg)
        if hasattr(model, "print_trainable_parameters"):
            model.print_trainable_parameters()
        return model
    except Exception:
        try:
            from inferforge.nexara.peft import LoRAConfig, LoRAModel
            wrapped = LoRAModel(model, LoRAConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout, target_modules=targets))
            wrapped.freeze_base_model()
            return wrapped.model
        except Exception:
            return model


def build_optimizer(model, cfg: UniversalTrainConfig, lr: float):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 1 or name.endswith("bias") or "norm" in name.lower():
            no_decay.append(param)
        else:
            decay.append(param)
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    opt_name = cfg.optimizer.lower()
    if opt_name == "sgd":
        return torch.optim.SGD(groups, lr=lr, momentum=0.9)
    if opt_name == "adam":
        return torch.optim.Adam(groups, lr=lr, betas=(0.9, 0.95), eps=1e-8)
    if opt_name == "lion":
        try:
            from lion_pytorch import Lion
            return Lion(groups, lr=lr, weight_decay=cfg.weight_decay)
        except Exception:
            pass
    if opt_name in {"adam8bit", "adamw8bit"}:
        try:
            import bitsandbytes as bnb
            return bnb.optim.AdamW8bit(groups, lr=lr, betas=(0.9, 0.95))
        except Exception:
            pass
    if opt_name == "adafactor":
        try:
            from transformers.optimization import Adafactor
            return Adafactor(groups, lr=lr, relative_step=False, scale_parameter=False)
        except Exception:
            pass
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95), eps=1e-8)


def build_scheduler(optimizer, steps: int, warmup: int, kind: str, min_lr_ratio: float = 0.1):
    """Warmup + decay schedule in *optimizer-step* units.

    Warmup starts at ``1/warmup`` (not ~0) so the first step is not wasted; ``min_lr_ratio``
    is the floor relative to the peak LR. Supports cosine (default), linear, constant and wsd
    (warmup-stable-decay: constant until the last 20% then linear to the floor).
    """
    min_lr_ratio = min(max(float(min_lr_ratio), 0.0), 1.0)
    kind = (kind or "cosine").lower()

    def lr_lambda(step: int) -> float:
        if steps <= 0:
            return 1.0
        if warmup > 0 and step < warmup:
            return (step + 1) / warmup
        progress = min((step - warmup) / max(steps - warmup, 1), 1.0)
        if kind == "constant":
            return 1.0
        if kind == "linear":
            return min_lr_ratio + (1.0 - min_lr_ratio) * (1.0 - progress)
        if kind == "wsd":
            decay_start = 0.8
            if progress < decay_start:
                return 1.0
            frac = (progress - decay_start) / (1.0 - decay_start)
            return min_lr_ratio + (1.0 - min_lr_ratio) * (1.0 - frac)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _min_lr_ratio(cfg: UniversalTrainConfig, peak_lr: float) -> float:
    if cfg.min_learning_rate is not None and peak_lr > 0:
        return float(cfg.min_learning_rate) / float(peak_lr)
    return 0.1


def _make_grad_scaler(enabled: bool):
    if not enabled:
        return None
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=True)


def _prune_checkpoints(out_dir: Path, limit: int | None) -> None:
    """Keep only the newest ``limit`` ``checkpoint-N`` directories under ``out_dir``."""
    if not limit or limit <= 0:
        return
    ckpts = []
    for p in out_dir.glob("checkpoint-*"):
        if p.is_dir():
            try:
                ckpts.append((int(p.name.split("-", 1)[1]), p))
            except ValueError:
                continue
    ckpts.sort()
    for _, path in ckpts[:-limit] if len(ckpts) > limit else []:
        shutil.rmtree(path, ignore_errors=True)


class UniversalTrainer:
    def __init__(self, config: UniversalTrainConfig | None = None, hardware: dict[str, Any] | None = None):
        self.config = config or UniversalTrainConfig()
        self.hardware = hardware or {}
        self.model = None
        self.tokenizer = None
        self.history: list[dict[str, Any]] = []
        self.best_loss = float("inf")
        self.monitor = None
        self._distributed = None

    def resolve(self) -> dict[str, Any]:
        cfg = self.config
        hw = self.hardware
        vram = float(hw.get("gpu_memory", 0) or 0)
        if vram > 256:
            vram = vram / 1024.0
        ram = float(hw.get("ram", 16) or 16)
        gpu_count = int(hw.get("gpu_count", 0) or 0)
        gpu_available = bool(hw.get("gpu_available", False))
        if TORCH_AVAILABLE and torch.cuda.is_available():
            gpu_available = True
            gpu_count = max(gpu_count, torch.cuda.device_count())
            if vram <= 0:
                try:
                    vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                except Exception:
                    vram = 8.0
        recipe = get_recipe(cfg.scale, cfg.params)
        fit = fit_recipe_to_hardware(recipe, vram, ram, gpu_count, gpu_available)
        stage = cfg.stage
        if stage in {None, "", "auto"}:
            if cfg.from_scratch or not cfg.model:
                stage = "pretrain"
            else:
                stage = "sft"
        peft = cfg.peft_method
        if peft in {None, "", "auto"}:
            peft = fit["peft_method"] if stage not in {"pretrain"} else "none"
            if stage in ALIGNMENT_METHODS and peft == "none" and fit["mode"] in {"lora", "qlora"}:
                peft = fit["peft_method"]
        seq_len = cfg.seq_len or fit["seq_len"]
        batch = cfg.batch_size or fit["micro_batch_size"]
        finetune = stage != "pretrain" and bool(cfg.model) and not cfg.from_scratch
        if cfg.gradient_accumulation_steps:
            accum = cfg.gradient_accumulation_steps
        elif finetune:
            # The recipe's global_batch_tokens targets scratch pretraining (hundreds of k tokens per
            # step). Fine-tuning wants an effective batch of ~16 sequences, otherwise a typical SFT
            # dataset yields only a handful of optimizer steps.
            accum = max(1, math.ceil(16 / max(int(batch) * max(gpu_count, 1), 1)))
        else:
            accum = fit["gradient_accumulation_steps"]
        lr = cfg.learning_rate or compute_optimal_lr(
            get_recipe(fit["fitted_scale"]).params,
            "pretrain" if stage == "pretrain" else ("align" if stage in ALIGNMENT_METHODS else "sft"),
        )
        bits = cfg.bits if cfg.bits is not None else (fit["bits"] if peft in {"qlora"} or fit["mode"] == "qlora" else None)
        if peft == "qlora":
            bits = bits or 4
        gc = cfg.gradient_checkpointing if cfg.gradient_checkpointing is not None else fit["gradient_checkpointing"]
        bf16 = cfg.bf16 if cfg.bf16 is not None else fit["bf16"]
        fp16 = cfg.fp16 if cfg.fp16 is not None else fit["fp16"]
        return {
            "stage": stage,
            "peft": peft,
            "seq_len": int(seq_len),
            "batch_size": int(batch),
            "gradient_accumulation_steps": int(accum),
            "learning_rate": float(lr),
            "bits": bits,
            "gradient_checkpointing": bool(gc),
            "bf16": bool(bf16),
            "fp16": bool(fp16) and not bf16,
            "fit": fit,
            "recipe": recipe.to_dict(),
            "device": "cuda" if gpu_available and TORCH_AVAILABLE and torch.cuda.is_available() else "cpu",
        }

    def _device(self, resolved: dict[str, Any]):
        if not TORCH_AVAILABLE:
            return None
        if resolved["device"] == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _maybe_compile(self, model):
        if not TORCH_AVAILABLE:
            return model
        if os.environ.get("NEXARA_COMPILE", "0") == "1":
            try:
                return torch.compile(model)
            except Exception:
                return model
        return model

    def train(self, progress: ProgressCb | None = None) -> dict[str, Any]:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for UniversalTrainer")
        cfg = self.config
        resolved = self.resolve()
        out_dir = Path(cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "train_config.json").write_text(json.dumps({**cfg.to_dict(), "resolved": {k: v for k, v in resolved.items() if k != "fit"}}, indent=2, default=str), encoding="utf-8")
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)
        records = load_raw_records(cfg.data, cfg.max_samples)
        if not records:
            records = [{"text": f"nexara scratch sample {i} " * 8} for i in range(64)]
        eval_records = load_raw_records(cfg.eval_data, None) if cfg.eval_data else records[: max(1, len(records) // 10)]
        texts = [record_to_text(r) for r in records[:20000]]
        tokenizer = load_tokenizer(None if cfg.from_scratch else cfg.model, texts, from_scratch=cfg.from_scratch)
        self.tokenizer = tokenizer
        vocab = getattr(tokenizer, "vocab_size", None) or (len(tokenizer) if hasattr(tokenizer, "__len__") else None)
        arch_over = dict(cfg.architecture)
        if vocab:
            arch_over.setdefault("vocab_size", int(vocab))
        arch_over.setdefault("seq_len", resolved["seq_len"])
        quant_bits = resolved["bits"] if resolved["peft"] == "qlora" or resolved["bits"] == 4 else None
        dtype = "bfloat16" if resolved["bf16"] else ("float16" if resolved["fp16"] else "float32")
        model = load_any_model(
            cfg.model,
            from_scratch=cfg.from_scratch or not cfg.model,
            scale=resolved["fit"]["fitted_scale"],
            params=cfg.params,
            task=cfg.task,
            torch_dtype=dtype if resolved["device"] == "cuda" else "float32",
            quantization_bits=quant_bits if resolved["device"] == "cuda" else None,
            device_map="auto" if resolved["device"] == "cuda" and quant_bits else None,
            trust_remote_code=cfg.trust_remote_code,
            overrides=arch_over,
        )
        if resolved["peft"] not in {"none", "full", "off"} and resolved["stage"] != "pretrain":
            model = apply_peft(model, resolved["peft"], cfg)
        if resolved["gradient_checkpointing"] and hasattr(model, "gradient_checkpointing_enable"):
            try:
                model.gradient_checkpointing_enable()
            except Exception:
                pass
        device = self._device(resolved)
        if getattr(model, "device", None) is None or str(getattr(model, "device", "cpu")) == "cpu":
            try:
                model = model.to(device)
            except Exception:
                pass
        gpu_count = int(resolved.get("fit", {}).get("gpu_count", 0) or 0)
        if TORCH_AVAILABLE and torch.cuda.is_available():
            gpu_count = max(gpu_count, torch.cuda.device_count())
        world_size = int(os.environ.get("WORLD_SIZE", "1") or 1)
        if (
            gpu_count > 1
            and quant_bits is None
            and world_size <= 1
            and os.environ.get("NEXARA_DISABLE_DATAPARALLEL", "0") != "1"
        ):
            try:
                model = torch.nn.DataParallel(model)
            except Exception:
                pass
        elif world_size > 1 and TORCH_AVAILABLE:
            try:
                from inferforge.nexara.distributed_training import (
                    DistributedConfig,
                    DistributedTrainer,
                )

                dist_cfg = DistributedConfig(
                    world_size=world_size,
                    use_ddp=True,
                    mixed_precision=bool(resolved["fp16"] or resolved["bf16"]),
                )
                dist_trainer = DistributedTrainer(dist_cfg)
                rank = int(os.environ.get("RANK", "0") or 0)
                if not dist_trainer.initialized:
                    dist_trainer.setup(rank, world_size)
                model = dist_trainer.wrap_model(model)
                self._distributed = dist_trainer
            except Exception:
                self._distributed = None
        model = self._maybe_compile(model)
        self.model = model
        if cfg.time_budget_minutes and cfg.time_budget_minutes > 0:
            deadline = time.time() + float(cfg.time_budget_minutes) * 60.0
            self.stop_fn = lambda: time.time() >= deadline
        stage = resolved["stage"]
        if stage in ALIGNMENT_METHODS or records_are_preference(records):
            result = self._train_alignment(model, tokenizer, records, eval_records, resolved, progress)
        else:
            result = self._train_lm(model, tokenizer, records, eval_records, resolved, progress)
        self._save(out_dir / "final", model, tokenizer, merge_adapters=True)
        result["output_dir"] = str(out_dir / "final")
        result["resolved"] = {k: v for k, v in resolved.items() if k != "recipe"}
        result["parameters"] = count_parameters(model) if hasattr(model, "parameters") else {}
        (out_dir / "metrics.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        return result

    def _train_lm(self, model, tokenizer, records, eval_records, resolved, progress) -> dict[str, Any]:
        cfg = self.config
        seq_len = resolved["seq_len"]
        completions_only = bool(cfg.train_on_completions_only) and resolved["stage"] != "pretrain"
        packed = tokenize_records(records, tokenizer, seq_len, packing=cfg.packing, completions_only=completions_only)
        dataset = TensorDataset(packed)
        eval_ds = TensorDataset(
            tokenize_records(eval_records or records[:8], tokenizer, seq_len, packing=cfg.packing, completions_only=completions_only)
        )
        generator = torch.Generator()
        generator.manual_seed(cfg.seed)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=resolved["batch_size"], shuffle=True, num_workers=cfg.num_workers,
            generator=generator, drop_last=False,
        )
        eval_loader = torch.utils.data.DataLoader(eval_ds, batch_size=max(1, resolved["batch_size"]), shuffle=False)
        # All step accounting below is in *optimizer* steps; `accum` micro-batches make one step.
        accum = max(int(resolved["gradient_accumulation_steps"]), 1)
        micro_per_epoch = max(1, len(loader))
        steps_per_epoch = max(1, math.ceil(micro_per_epoch / accum))
        max_steps = int(cfg.max_steps or steps_per_epoch * cfg.epochs)
        warmup = int(max_steps * cfg.warmup_ratio)
        optimizer = build_optimizer(model, cfg, resolved["learning_rate"])
        scheduler = build_scheduler(
            optimizer, max_steps, warmup, cfg.scheduler, _min_lr_ratio(cfg, resolved["learning_rate"])
        )
        use_fp16 = resolved["fp16"] and resolved["device"] == "cuda"
        scaler = _make_grad_scaler(use_fp16)
        autocast_dtype = torch.bfloat16 if resolved["bf16"] else (torch.float16 if use_fp16 else None)
        from inferforge.nexara.training_monitor import TrainingMonitor

        self.monitor = TrainingMonitor(cfg.output_dir)
        device = self._device(resolved)
        out_dir = Path(cfg.output_dir)

        global_step = 0
        if cfg.resume:
            restored = self._load_checkpoint(Path(cfg.resume), model, optimizer, scheduler)
            if restored:
                global_step = int(restored)
        start_step = global_step
        epoch = global_step // steps_per_epoch
        skip_micro = (global_step % steps_per_epoch) * accum

        model.train()
        t0 = time.time()
        running_loss = 0.0
        running_count = 0
        tokens_seen = 0
        samples_seen = 0
        micro_steps = 0
        consecutive_nan = 0
        evals_without_improvement = 0
        status = "completed"
        stop_fn = getattr(self, "stop_fn", None)

        def optimizer_step() -> bool:
            """Clip + step + schedule. Returns False if the step was skipped for non-finite grads."""
            if scaler is not None:
                scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            if not torch.isfinite(grad_norm):
                if self.monitor:
                    self.monitor.record_nan_skip()
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.update()
                return False
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            return True

        def log_and_checkpoint() -> None:
            nonlocal running_loss, running_count, evals_without_improvement, status
            if global_step % cfg.logging_steps == 0 or global_step == max_steps:
                avg = running_loss / max(running_count, 1)
                running_loss, running_count = 0.0, 0
                elapsed = max(time.time() - t0, 1e-6)
                row = {
                    "step": global_step,
                    "loss": avg,
                    "ppl": math.exp(min(avg, 20)),
                    "lr": scheduler.get_last_lr()[0],
                    "epoch": global_step / steps_per_epoch,
                    "tokens_per_sec": tokens_seen / elapsed,
                    "eta_seconds": (max_steps - global_step) * elapsed / max(global_step - start_step, 1),
                }
                self.history.append(row)
                if self.monitor:
                    self.monitor.log(
                        global_step, loss=avg, learning_rate=row["lr"], epoch=row["epoch"],
                        ppl=row["ppl"], tokens_per_sec=row["tokens_per_sec"],
                    )
                if progress:
                    progress("train", global_step / max_steps, row)
            if cfg.eval_steps and global_step % cfg.eval_steps == 0 and global_step < max_steps:
                ev = self._evaluate(model, eval_loader, resolved)
                if ev["eval_loss"] < self.best_loss:
                    self.best_loss = ev["eval_loss"]
                    evals_without_improvement = 0
                    self._save(out_dir / "best", model, tokenizer)
                else:
                    evals_without_improvement += 1
                    if cfg.early_stopping_patience and evals_without_improvement >= cfg.early_stopping_patience:
                        status = "early_stopped"
            if cfg.save_steps and global_step % cfg.save_steps == 0 and global_step < max_steps:
                self._save_checkpoint(out_dir / f"checkpoint-{global_step}", model, optimizer, scheduler, global_step)
                _prune_checkpoints(out_dir, cfg.save_total_limit)

        while global_step < max_steps and status == "completed":
            generator.manual_seed(cfg.seed + epoch)
            accum_count = 0
            for idx, batch in enumerate(loader):
                if skip_micro > 0:
                    skip_micro -= 1
                    continue
                if callable(stop_fn) and stop_fn():
                    status = "interrupted"
                    break
                batch = {k: v.to(device, non_blocking=True) if hasattr(v, "to") else v for k, v in batch.items()}
                ctx = torch.autocast(device_type="cuda", dtype=autocast_dtype) if autocast_dtype and resolved["device"] == "cuda" else _nullctx()
                with ctx:
                    loss = self._forward_loss(model, batch)
                if not torch.isfinite(loss):
                    consecutive_nan += 1
                    if self.monitor:
                        self.monitor.record_nan_skip()
                    optimizer.zero_grad(set_to_none=True)
                    accum_count = 0
                    if consecutive_nan >= max(int(cfg.max_nan_skips), 1):
                        status = "diverged"
                        break
                    continue
                scaled = loss / accum
                if scaler is not None:
                    scaler.scale(scaled).backward()
                else:
                    scaled.backward()
                micro_steps += 1
                accum_count += 1
                running_loss += float(loss.item())
                running_count += 1
                mask = batch.get("attention_mask")
                tokens_seen += int(mask.sum().item()) if mask is not None else int(batch["input_ids"].numel())
                samples_seen += int(batch["input_ids"].shape[0])
                # Step on a full accumulation window, or flush the partial window at epoch end.
                if accum_count < accum and idx != micro_per_epoch - 1:
                    continue
                accum_count = 0
                if not optimizer_step():
                    consecutive_nan += 1
                    if consecutive_nan >= max(int(cfg.max_nan_skips), 1):
                        status = "diverged"
                        break
                    continue
                consecutive_nan = 0
                global_step += 1
                log_and_checkpoint()
                if global_step >= max_steps or status != "completed":
                    break
            epoch += 1

        if status in {"interrupted", "diverged", "early_stopped"}:
            self._save_checkpoint(out_dir / f"checkpoint-{global_step}", model, optimizer, scheduler, global_step)
            _prune_checkpoints(out_dir, cfg.save_total_limit)
        eval_metrics = self._evaluate(model, eval_loader, resolved)
        if eval_metrics["eval_loss"] < self.best_loss:
            self.best_loss = eval_metrics["eval_loss"]
            self._save(out_dir / "best", model, tokenizer)
        elapsed = time.time() - t0
        result = {
            "status": status,
            "interrupted": status == "interrupted",
            "stage": resolved["stage"],
            "steps": global_step,
            "micro_steps": micro_steps,
            "samples_seen": samples_seen,
            "tokens_seen": tokens_seen,
            "tokens_per_sec": tokens_seen / max(elapsed, 1e-6),
            "epochs_completed": global_step / steps_per_epoch,
            "max_steps": max_steps,
            "completions_only": completions_only,
            "train_history": self.history[-50:],
            "eval": eval_metrics,
            "best_loss": self.best_loss if self.best_loss < float("inf") else eval_metrics.get("eval_loss"),
            "seconds": elapsed,
        }
        if self.monitor:
            result["monitor"] = self.monitor.summary()
        return result

    def _forward_loss(self, model, batch) -> torch.Tensor:
        try:
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
                labels=batch.get("labels", batch["input_ids"]),
            )
        except TypeError:
            out = model(input_ids=batch["input_ids"], labels=batch.get("labels", batch["input_ids"]))
        if isinstance(out, dict):
            loss = out.get("loss")
            if loss is None:
                logits = out["logits"]
                loss = F.cross_entropy(
                    logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
                    batch["input_ids"][:, 1:].contiguous().view(-1),
                    ignore_index=-100,
                )
            return loss
        if hasattr(out, "loss") and out.loss is not None:
            return out.loss
        logits = out.logits if hasattr(out, "logits") else out
        return F.cross_entropy(
            logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
            batch["input_ids"][:, 1:].contiguous().view(-1),
            ignore_index=-100,
        )

    def _evaluate(self, model, loader, resolved) -> dict[str, Any]:
        model.eval()
        total = 0.0
        n = 0
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self._device(resolved)) if hasattr(v, "to") else v for k, v in batch.items()}
                loss = self._forward_loss(model, batch)
                total += float(loss.item())
                n += 1
                if n >= 32:
                    break
        model.train()
        avg = total / max(n, 1)
        return {"eval_loss": avg, "eval_perplexity": math.exp(min(avg, 20)), "eval_batches": n}

    def _train_alignment(self, model, tokenizer, records, eval_records, resolved, progress) -> dict[str, Any]:
        cfg = self.config
        method = cfg.alignment_method if resolved["stage"] not in ALIGNMENT_METHODS else resolved["stage"]
        if not records_are_preference(records):
            converted = []
            for rec in records:
                text = record_to_text(rec)
                converted.append({"prompt": text[: max(1, len(text) // 2)], "chosen": text, "rejected": text[: max(1, len(text) // 3)]})
            records = converted
        pairs = tokenize_preference(records, tokenizer, resolved["seq_len"])
        if not pairs:
            return {"status": "error", "error": "no preference pairs"}
        keys = [k for k in pairs[0].keys() if k != "prompt_len"]
        data = {k: [p[k] for p in pairs] for k in keys}
        data["prompt_len"] = [p["prompt_len"] for p in pairs]
        dataset = TensorDataset(data)
        loader = torch.utils.data.DataLoader(dataset, batch_size=max(1, resolved["batch_size"]), shuffle=True)
        align_cfg = AlignmentConfig.from_dict({
            "method": method,
            "learning_rate": resolved["learning_rate"],
            **cfg.alignment,
        })
        trainer = AlignmentTrainer(align_cfg)
        ref_model = None
        if method in {"dpo", "ipo", "kto"} and not align_cfg.reference_free:
            try:
                ref_model = load_any_model(
                    cfg.model,
                    from_scratch=False,
                    scale=resolved["fit"]["fitted_scale"],
                    task=cfg.task,
                    torch_dtype="float16" if resolved["device"] == "cuda" else "float32",
                )
                ref_model.eval()
                for p in ref_model.parameters():
                    p.requires_grad = False
                try:
                    ref_model.to(self._device(resolved))
                except Exception:
                    pass
            except Exception:
                ref_model = None
        accum = max(int(resolved["gradient_accumulation_steps"]), 1)
        micro_per_epoch = max(1, len(loader))
        steps_per_epoch = max(1, math.ceil(micro_per_epoch / accum))
        max_steps = int(cfg.max_steps or steps_per_epoch * cfg.epochs)
        optimizer = build_optimizer(model, cfg, resolved["learning_rate"])
        scheduler = build_scheduler(
            optimizer, max_steps, int(max_steps * cfg.warmup_ratio), cfg.scheduler,
            _min_lr_ratio(cfg, resolved["learning_rate"]),
        )
        if self.monitor is None:
            from inferforge.nexara.training_monitor import TrainingMonitor
            self.monitor = TrainingMonitor(cfg.output_dir)
        device = self._device(resolved)
        out_dir = Path(cfg.output_dir)
        model.train()
        t0 = time.time()
        step = 0
        micro_steps = 0
        samples_seen = 0
        consecutive_nan = 0
        running = 0.0
        running_count = 0
        last_metrics: dict[str, Any] = {}
        status = "completed"
        stop_fn = getattr(self, "stop_fn", None)
        while step < max_steps and status == "completed":
            accum_count = 0
            for idx, batch in enumerate(loader):
                if callable(stop_fn) and stop_fn():
                    status = "interrupted"
                    break
                batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
                loss, metrics = trainer.step(model, batch, ref_model)
                if not torch.isfinite(loss):
                    consecutive_nan += 1
                    if self.monitor:
                        self.monitor.record_nan_skip()
                    optimizer.zero_grad(set_to_none=True)
                    accum_count = 0
                    if consecutive_nan >= max(int(cfg.max_nan_skips), 1):
                        status = "diverged"
                        break
                    continue
                (loss / accum).backward()
                micro_steps += 1
                accum_count += 1
                samples_seen += int(batch["chosen_input_ids"].shape[0])
                running += float(loss.item())
                running_count += 1
                last_metrics = metrics
                if accum_count < accum and idx != micro_per_epoch - 1:
                    continue
                accum_count = 0
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                if not torch.isfinite(grad_norm):
                    consecutive_nan += 1
                    if self.monitor:
                        self.monitor.record_nan_skip()
                    optimizer.zero_grad(set_to_none=True)
                    if consecutive_nan >= max(int(cfg.max_nan_skips), 1):
                        status = "diverged"
                        break
                    continue
                consecutive_nan = 0
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if step % cfg.logging_steps == 0 or step == max_steps:
                    row = {"step": step, "loss": running / max(running_count, 1), "lr": scheduler.get_last_lr()[0], **metrics}
                    running, running_count = 0.0, 0
                    self.history.append(row)
                    if self.monitor:
                        self.monitor.log(step, loss=row["loss"], **{k: v for k, v in metrics.items() if isinstance(v, (int, float))})
                    if progress:
                        progress("align", step / max_steps, row)
                if cfg.save_steps and step % cfg.save_steps == 0 and step < max_steps:
                    self._save_checkpoint(out_dir / f"checkpoint-{step}", model, optimizer, scheduler, step)
                    _prune_checkpoints(out_dir, cfg.save_total_limit)
                if step >= max_steps:
                    break
        if status in {"interrupted", "diverged"}:
            self._save_checkpoint(out_dir / f"checkpoint-{step}", model, optimizer, scheduler, step)
            _prune_checkpoints(out_dir, cfg.save_total_limit)
        return {
            "status": status,
            "interrupted": status == "interrupted",
            "stage": method,
            "steps": step,
            "micro_steps": micro_steps,
            "samples_seen": samples_seen,
            "max_steps": max_steps,
            "train_history": self.history[-50:],
            "last_metrics": last_metrics,
            "seconds": time.time() - t0,
        }

    def _unwrap(self, model):
        return model.module if hasattr(model, "module") else model

    def _save(self, path: Path, model, tokenizer, merge_adapters: bool = False) -> None:
        """Save model + tokenizer. With ``merge_adapters`` a PEFT model is also exported as a
        standalone merged model under ``path`` (the raw adapter goes to ``path/adapter``) so the
        result is directly loadable by ``forge run`` / transformers without peft."""
        path.mkdir(parents=True, exist_ok=True)
        model = self._unwrap(model)
        is_peft = hasattr(model, "peft_config") and hasattr(model, "merge_and_unload")
        try:
            if is_peft and merge_adapters:
                model.save_pretrained(str(path / "adapter"))
                try:
                    import copy
                    merged = copy.deepcopy(model).merge_and_unload()
                    merged.save_pretrained(str(path), safe_serialization=True)
                except Exception:
                    # Fall back to adapter-only if there isn't enough memory to duplicate the model.
                    model.save_pretrained(str(path))
            elif hasattr(model, "save_pretrained"):
                model.save_pretrained(str(path))
            else:
                torch.save(model.state_dict(), path / "pytorch_model.bin")
        except Exception:
            torch.save(model.state_dict(), path / "pytorch_model.bin")
        try:
            if hasattr(tokenizer, "save_pretrained"):
                tokenizer.save_pretrained(str(path))
        except Exception:
            pass

    def _save_checkpoint(self, path: Path, model, optimizer, scheduler, step: int) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self._save(path, model, self.tokenizer)
        torch.save({
            "step": step,
            "model": self._unwrap(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_loss": self.best_loss,
            "history": self.history,
        }, path / "trainer_state.pt")

    def _load_checkpoint(self, path: Path, model, optimizer, scheduler) -> int:
        state_path = path / "trainer_state.pt"
        if not state_path.exists():
            return 0
        try:
            state = torch.load(state_path, map_location="cpu", weights_only=False)
        except TypeError:
            state = torch.load(state_path, map_location="cpu")
        try:
            target = self._unwrap(model)
            if "model" in state:
                target.load_state_dict(state["model"])
        except Exception:
            pass
        try:
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
        except Exception:
            pass
        self.best_loss = state.get("best_loss", float("inf"))
        self.history = state.get("history", [])
        return int(state.get("step", 0) or 0)


class _nullctx:
    def __enter__(self):
        return None

    def __exit__(self, *args):
        return False


def train_any(
    *,
    model: str | None = None,
    data: str | list | None = None,
    output_dir: str = "./nexara_runs",
    from_scratch: bool = False,
    scale: str = "tiny",
    stage: str = "auto",
    hardware: dict[str, Any] | None = None,
    **kwargs,
) -> dict[str, Any]:
    cfg = UniversalTrainConfig(
        output_dir=output_dir,
        model=model,
        data=data,
        from_scratch=from_scratch,
        scale=scale,
        stage=stage,
        **{k: v for k, v in kwargs.items() if k in UniversalTrainConfig.__dataclass_fields__},
    )
    trainer = UniversalTrainer(cfg, hardware=hardware)
    return trainer.train()
