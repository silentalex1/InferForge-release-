from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from inferforge.nexara.scaling import ScaleRecipe, get_recipe

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


HF_ARCH_MAP = {
    "gpt": "gpt2",
    "gpt2": "gpt2",
    "llama": "llama",
    "mistral": "mistral",
    "qwen": "qwen2",
    "qwen2": "qwen2",
    "gemma": "gemma",
    "phi": "phi",
    "phi3": "phi3",
    "falcon": "falcon",
    "mpt": "mpt",
    "neox": "gpt_neox",
    "olmo": "olmo",
    "moe": "mixtral",
    "mixtral": "mixtral",
    "mamba": "mamba",
    "bert": "bert",
    "t5": "t5",
    "bart": "bart",
}

TASK_MODEL_CLASS = {
    "causal_lm": "AutoModelForCausalLM",
    "text-generation": "AutoModelForCausalLM",
    "code-completion": "AutoModelForCausalLM",
    "chat": "AutoModelForCausalLM",
    "seq2seq": "AutoModelForSeq2SeqLM",
    "t5": "AutoModelForSeq2SeqLM",
    "classification": "AutoModelForSequenceClassification",
    "token_classification": "AutoModelForTokenClassification",
    "embedding": "AutoModel",
    "reward_model": "AutoModelForSequenceClassification",
    "masked_lm": "AutoModelForMaskedLM",
}


@dataclass
class ArchConfig:
    n_layer: int = 12
    n_embd: int = 768
    n_head: int = 12
    n_kv_head: int = 12
    n_inner: int = 3072
    vocab_size: int = 32000
    seq_len: int = 2048
    rope_theta: float = 10000.0
    dropout: float = 0.0
    rms_norm: bool = True
    rms_eps: float = 1e-6
    swiglu: bool = True
    rope: bool = True
    tied_embeddings: bool = True
    use_moe: bool = False
    n_experts: int = 1
    n_active_experts: int = 1
    moe_aux_loss: float = 0.01
    bias: bool = False
    family: str = "llama"
    architecture: str = "llama"
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    initializer_range: float = 0.02
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_recipe(cls, recipe: ScaleRecipe) -> "ArchConfig":
        return cls(
            n_layer=recipe.n_layer,
            n_embd=recipe.n_embd,
            n_head=recipe.n_head,
            n_kv_head=recipe.n_kv_head,
            n_inner=recipe.n_inner,
            vocab_size=recipe.vocab_size,
            seq_len=recipe.seq_len,
            rope_theta=recipe.rope_theta,
            dropout=recipe.dropout,
            rms_norm=recipe.rms_norm,
            swiglu=recipe.swiglu,
            rope=recipe.rope,
            tied_embeddings=recipe.tied_embeddings,
            use_moe=recipe.use_moe,
            n_experts=recipe.n_experts,
            n_active_experts=recipe.n_active_experts,
            moe_aux_loss=recipe.moe_aux_loss,
            family=recipe.family,
            architecture=recipe.architecture,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ArchConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        kwargs = {k: v for k, v in data.items() if k in known}
        extra = {k: v for k, v in data.items() if k not in known}
        cfg = cls(**kwargs)
        cfg.extra.update(extra)
        return cfg

    @property
    def head_dim(self) -> int:
        return self.n_embd // max(self.n_head, 1)


if TORCH_AVAILABLE:

    class RMSNorm(nn.Module):
        def __init__(self, dim: int, eps: float = 1e-6):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(dim))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            orig = x.dtype
            x = x.float()
            norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
            return (self.weight * x * norm).to(orig)

    class RotaryEmbedding(nn.Module):
        def __init__(self, dim: int, max_seq: int = 131072, base: float = 10000.0):
            super().__init__()
            inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)
            self.max_seq = max_seq
            self._cache_len = 0
            self.register_buffer("cos_cached", torch.empty(0), persistent=False)
            self.register_buffer("sin_cached", torch.empty(0), persistent=False)

        def _update(self, seq_len: int, device, dtype):
            if seq_len <= self._cache_len and self.cos_cached.device == device:
                return
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            emb = torch.cat((freqs, freqs), dim=-1)
            self.cos_cached = emb.cos().to(dtype)
            self.sin_cached = emb.sin().to(dtype)
            self._cache_len = seq_len

        def forward(self, x: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
            self._update(seq_len, x.device, x.dtype)
            return self.cos_cached[:seq_len], self.sin_cached[:seq_len]

    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        q = (q * cos) + (_rotate_half(q) * sin)
        k = (k * cos) + (_rotate_half(k) * sin)
        return q, k

    class Attention(nn.Module):
        def __init__(self, cfg: ArchConfig):
            super().__init__()
            self.n_head = cfg.n_head
            self.n_kv_head = cfg.n_kv_head
            self.head_dim = cfg.head_dim
            self.n_rep = cfg.n_head // max(cfg.n_kv_head, 1)
            self.scale = self.head_dim ** -0.5
            self.q_proj = nn.Linear(cfg.n_embd, cfg.n_head * self.head_dim, bias=cfg.bias)
            self.k_proj = nn.Linear(cfg.n_embd, cfg.n_kv_head * self.head_dim, bias=cfg.bias)
            self.v_proj = nn.Linear(cfg.n_embd, cfg.n_kv_head * self.head_dim, bias=cfg.bias)
            self.o_proj = nn.Linear(cfg.n_head * self.head_dim, cfg.n_embd, bias=cfg.bias)
            self.drop = nn.Dropout(cfg.dropout)
            self.use_rope = cfg.rope
            if cfg.rope:
                self.rope = RotaryEmbedding(self.head_dim, cfg.seq_len * 8, cfg.rope_theta)

        def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
            if self.n_rep == 1:
                return x
            b, h, t, d = x.shape
            x = x[:, :, None, :, :].expand(b, h, self.n_rep, t, d)
            return x.reshape(b, h * self.n_rep, t, d)

        def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
            b, t, _ = x.shape
            q = self.q_proj(x).view(b, t, self.n_head, self.head_dim).transpose(1, 2)
            k = self.k_proj(x).view(b, t, self.n_kv_head, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).view(b, t, self.n_kv_head, self.head_dim).transpose(1, 2)
            if self.use_rope:
                cos, sin = self.rope(q, t)
                q, k = apply_rope(q, k, cos, sin)
            k = self._repeat_kv(k)
            v = self._repeat_kv(v)
            if hasattr(F, "scaled_dot_product_attention"):
                attn_mask = None
                if mask is not None:
                    attn_mask = mask[:, None, None, :].to(dtype=q.dtype)
                    attn_mask = (1.0 - attn_mask) * torch.finfo(q.dtype).min
                y = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=attn_mask, dropout_p=self.drop.p if self.training else 0.0, is_causal=True
                )
            else:
                att = (q @ k.transpose(-2, -1)) * self.scale
                causal = torch.triu(torch.ones(t, t, device=x.device, dtype=torch.bool), diagonal=1)
                att = att.masked_fill(causal, torch.finfo(att.dtype).min)
                if mask is not None:
                    att = att.masked_fill(mask[:, None, None, :] == 0, torch.finfo(att.dtype).min)
                att = self.drop(F.softmax(att, dim=-1))
                y = att @ v
            y = y.transpose(1, 2).contiguous().view(b, t, -1)
            return self.o_proj(y)

    class SwiGLU(nn.Module):
        def __init__(self, dim: int, hidden: int, bias: bool, dropout: float):
            super().__init__()
            self.w1 = nn.Linear(dim, hidden, bias=bias)
            self.w2 = nn.Linear(hidden, dim, bias=bias)
            self.w3 = nn.Linear(dim, hidden, bias=bias)
            self.drop = nn.Dropout(dropout)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.drop(self.w2(F.silu(self.w1(x)) * self.w3(x)))

    class MLP(nn.Module):
        def __init__(self, dim: int, hidden: int, bias: bool, dropout: float):
            super().__init__()
            self.fc = nn.Linear(dim, hidden, bias=bias)
            self.proj = nn.Linear(hidden, dim, bias=bias)
            self.drop = nn.Dropout(dropout)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.drop(self.proj(F.gelu(self.fc(x))))

    class MoE(nn.Module):
        def __init__(self, cfg: ArchConfig):
            super().__init__()
            self.n_experts = cfg.n_experts
            self.n_active = cfg.n_active_experts
            self.aux_coef = cfg.moe_aux_loss
            make = lambda: SwiGLU(cfg.n_embd, cfg.n_inner, cfg.bias, cfg.dropout) if cfg.swiglu else MLP(cfg.n_embd, cfg.n_inner, cfg.bias, cfg.dropout)
            self.experts = nn.ModuleList([make() for _ in range(cfg.n_experts)])
            self.gate = nn.Linear(cfg.n_embd, cfg.n_experts, bias=False)
            self.last_aux = torch.tensor(0.0)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            b, t, c = x.shape
            flat = x.reshape(-1, c)
            logits = self.gate(flat)
            weights, idx = torch.topk(logits, self.n_active, dim=-1)
            weights = F.softmax(weights, dim=-1)
            out = torch.zeros_like(flat)
            for k in range(self.n_active):
                expert_ids = idx[:, k]
                coef = weights[:, k].unsqueeze(-1)
                for e in range(self.n_experts):
                    mask = expert_ids == e
                    if not mask.any():
                        continue
                    out[mask] = out[mask] + coef[mask] * self.experts[e](flat[mask])
            probs = F.softmax(logits, dim=-1)
            mean_prob = probs.mean(dim=0)
            mean_sel = F.one_hot(idx, self.n_experts).float().sum(dim=1).mean(dim=0) / self.n_active
            self.last_aux = self.n_experts * (mean_prob * mean_sel).sum()
            return out.view(b, t, c)

    class Block(nn.Module):
        def __init__(self, cfg: ArchConfig):
            super().__init__()
            Norm = RMSNorm if cfg.rms_norm else nn.LayerNorm
            self.n1 = Norm(cfg.n_embd, eps=cfg.rms_eps) if cfg.rms_norm else nn.LayerNorm(cfg.n_embd)
            self.n2 = Norm(cfg.n_embd, eps=cfg.rms_eps) if cfg.rms_norm else nn.LayerNorm(cfg.n_embd)
            self.attn = Attention(cfg)
            if cfg.use_moe and cfg.n_experts > 1:
                self.mlp = MoE(cfg)
            elif cfg.swiglu:
                self.mlp = SwiGLU(cfg.n_embd, cfg.n_inner, cfg.bias, cfg.dropout)
            else:
                self.mlp = MLP(cfg.n_embd, cfg.n_inner, cfg.bias, cfg.dropout)

        def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
            x = x + self.attn(self.n1(x), mask)
            x = x + self.mlp(self.n2(x))
            return x

    class NexaraTransformer(nn.Module):
        def __init__(self, cfg: ArchConfig):
            super().__init__()
            self.cfg = cfg
            self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
            self.drop = nn.Dropout(cfg.dropout)
            self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
            if cfg.rms_norm:
                self.norm = RMSNorm(cfg.n_embd, cfg.rms_eps)
            else:
                self.norm = nn.LayerNorm(cfg.n_embd)
            self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
            if cfg.tied_embeddings:
                self.lm_head.weight = self.tok_emb.weight
            if not cfg.rope:
                self.pos_emb = nn.Embedding(cfg.seq_len, cfg.n_embd)
            else:
                self.pos_emb = None
            self.apply(self._init)

        def _init(self, module):
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=self.cfg.initializer_range)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=self.cfg.initializer_range)

        def gradient_checkpointing_enable(self):
            self._grad_ckpt = True

        def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
            labels: torch.Tensor | None = None,
            **kwargs,
        ) -> dict[str, torch.Tensor]:
            b, t = input_ids.shape
            x = self.tok_emb(input_ids)
            if self.pos_emb is not None:
                pos = torch.arange(t, device=input_ids.device)
                x = x + self.pos_emb(pos)[None, :, :]
            x = self.drop(x)
            use_ckpt = getattr(self, "_grad_ckpt", False) and self.training
            for block in self.blocks:
                if use_ckpt:
                    x = torch.utils.checkpoint.checkpoint(block, x, attention_mask, use_reentrant=False)
                else:
                    x = block(x, attention_mask)
            x = self.norm(x)
            logits = self.lm_head(x)
            loss = None
            aux = 0.0
            for block in self.blocks:
                mlp = block.mlp
                if isinstance(mlp, MoE):
                    aux = aux + mlp.last_aux
            if labels is not None:
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )
                if isinstance(aux, torch.Tensor):
                    loss = loss + self.cfg.moe_aux_loss * aux
            return {"loss": loss, "logits": logits}

        def generate(
            self,
            input_ids: torch.Tensor,
            max_new_tokens: int = 32,
            temperature: float = 1.0,
            top_k: int = 0,
            eos_token_id: int | None = None,
        ) -> torch.Tensor:
            self.eval()
            ids = input_ids
            for _ in range(max_new_tokens):
                cropped = ids[:, -self.cfg.seq_len :]
                logits = self.forward(cropped)["logits"][:, -1, :]
                if temperature <= 0:
                    next_id = logits.argmax(dim=-1, keepdim=True)
                else:
                    logits = logits / max(temperature, 1e-6)
                    if top_k > 0:
                        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                        logits[logits < v[:, [-1]]] = -float("inf")
                    probs = F.softmax(logits, dim=-1)
                    next_id = torch.multinomial(probs, num_samples=1)
                ids = torch.cat([ids, next_id], dim=1)
                if eos_token_id is not None and (next_id == eos_token_id).all():
                    break
            return ids

        def num_parameters(self, trainable_only: bool = False) -> int:
            params = self.parameters() if not trainable_only else (p for p in self.parameters() if p.requires_grad)
            return sum(p.numel() for p in params)

        def save_pretrained(self, path: str) -> None:
            import json
            from pathlib import Path
            p = Path(path)
            p.mkdir(parents=True, exist_ok=True)
            torch.save(self.state_dict(), p / "pytorch_model.bin")
            (p / "config.json").write_text(json.dumps(self.cfg.to_dict(), indent=2), encoding="utf-8")

        @classmethod
        def from_pretrained(cls, path: str) -> "NexaraTransformer":
            import json
            from pathlib import Path
            p = Path(path)
            cfg = ArchConfig.from_dict(json.loads((p / "config.json").read_text(encoding="utf-8")))
            model = cls(cfg)
            state = torch.load(p / "pytorch_model.bin", map_location="cpu")
            model.load_state_dict(state)
            return model

else:
    NexaraTransformer = None


def build_arch_config(
    scale: str | None = None,
    params: int | None = None,
    overrides: dict[str, Any] | None = None,
) -> ArchConfig:
    recipe = get_recipe(scale, params)
    cfg = ArchConfig.from_recipe(recipe)
    if overrides:
        for k, v in overrides.items():
            if hasattr(cfg, k) and v is not None:
                setattr(cfg, k, v)
    return cfg


def build_scratch_model(
    scale: str | None = None,
    params: int | None = None,
    config: ArchConfig | None = None,
    overrides: dict[str, Any] | None = None,
):
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required to build a scratch model")
    cfg = config or build_arch_config(scale, params, overrides)
    return NexaraTransformer(cfg)


def build_hf_scratch_model(cfg: ArchConfig, task: str = "causal_lm"):
    try:
        from transformers import AutoConfig, AutoModelForCausalLM
    except ImportError as exc:
        raise ImportError("transformers is required for HuggingFace scratch models") from exc
    arch = HF_ARCH_MAP.get(cfg.architecture, HF_ARCH_MAP.get(cfg.family, "llama"))
    common = {
        "vocab_size": cfg.vocab_size,
        "hidden_size": cfg.n_embd,
        "num_hidden_layers": cfg.n_layer,
        "num_attention_heads": cfg.n_head,
        "intermediate_size": cfg.n_inner,
        "max_position_embeddings": cfg.seq_len,
        "rms_norm_eps": cfg.rms_eps,
        "tie_word_embeddings": cfg.tied_embeddings,
        "hidden_act": "silu" if cfg.swiglu else "gelu",
        "use_cache": False,
    }
    if arch in {"llama", "mistral", "qwen2", "gemma", "phi3", "olmo"}:
        common["num_key_value_heads"] = cfg.n_kv_head
        common["rope_theta"] = cfg.rope_theta
    if arch == "gpt2":
        hf_cfg = AutoConfig.for_model(
            "gpt2",
            vocab_size=cfg.vocab_size,
            n_embd=cfg.n_embd,
            n_layer=cfg.n_layer,
            n_head=cfg.n_head,
            n_inner=cfg.n_inner,
            n_positions=cfg.seq_len,
            resid_pdrop=cfg.dropout,
            embd_pdrop=cfg.dropout,
            attn_pdrop=cfg.dropout,
        )
    elif arch == "mixtral":
        hf_cfg = AutoConfig.for_model(
            "mixtral",
            **common,
            num_local_experts=cfg.n_experts,
            num_experts_per_tok=cfg.n_active_experts,
        )
    else:
        try:
            hf_cfg = AutoConfig.for_model(arch, **common)
        except Exception:
            hf_cfg = AutoConfig.for_model("llama", **common)
    model = AutoModelForCausalLM.from_config(hf_cfg)
    return model


def load_any_model(
    source: str | None,
    *,
    from_scratch: bool = False,
    scale: str | None = None,
    params: int | None = None,
    task: str = "causal_lm",
    torch_dtype: str | None = None,
    quantization_bits: int | None = None,
    device_map: str | None = "auto",
    trust_remote_code: bool = True,
    overrides: dict[str, Any] | None = None,
):
    if from_scratch or source in {None, "", "scratch", "random"}:
        try:
            return build_hf_scratch_model(build_arch_config(scale, params, overrides), task)
        except Exception:
            return build_scratch_model(scale, params, overrides=overrides)
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required to load models")
    from pathlib import Path
    path = Path(source) if source else None
    if path and (path / "config.json").exists() and not (path / "tokenizer.json").exists() and not (path / "tokenizer_config.json").exists():
        try:
            return NexaraTransformer.from_pretrained(str(path))
        except Exception:
            pass
    try:
        from transformers import (
            AutoModel,
            AutoModelForCausalLM,
            AutoModelForSeq2SeqLM,
            AutoModelForSequenceClassification,
            BitsAndBytesConfig,
        )
    except ImportError as exc:
        if path and (path / "pytorch_model.bin").exists():
            return NexaraTransformer.from_pretrained(str(path))
        raise ImportError("transformers is required to load pretrained models") from exc
    dtype = None
    if torch_dtype == "float16" or torch_dtype == "fp16":
        dtype = torch.float16
    elif torch_dtype == "bfloat16" or torch_dtype == "bf16":
        dtype = torch.bfloat16
    elif torch_dtype == "float32" or torch_dtype == "fp32":
        dtype = torch.float32
    quant = None
    if quantization_bits in {4, 8}:
        try:
            kwargs = {"load_in_4bit": quantization_bits == 4, "load_in_8bit": quantization_bits == 8}
            if quantization_bits == 4:
                kwargs.update({
                    "bnb_4bit_compute_dtype": dtype or torch.float16,
                    "bnb_4bit_quant_type": "nf4",
                    "bnb_4bit_use_double_quant": True,
                })
            quant = BitsAndBytesConfig(**kwargs)
        except Exception:
            quant = None
    cls_name = TASK_MODEL_CLASS.get(task, "AutoModelForCausalLM")
    cls = {
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoModelForSeq2SeqLM": AutoModelForSeq2SeqLM,
        "AutoModelForSequenceClassification": AutoModelForSequenceClassification,
        "AutoModel": AutoModel,
    }.get(cls_name, AutoModelForCausalLM)
    load_kw: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
    }
    if dtype is not None:
        load_kw["torch_dtype"] = dtype
    if quant is not None:
        load_kw["quantization_config"] = quant
        load_kw["device_map"] = device_map
    elif device_map and torch.cuda.is_available():
        load_kw["device_map"] = device_map
    return cls.from_pretrained(source, **load_kw)


def count_parameters(model) -> dict[str, int]:
    total = 0
    trainable = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    return {"total": total, "trainable": trainable, "frozen": total - trainable}
