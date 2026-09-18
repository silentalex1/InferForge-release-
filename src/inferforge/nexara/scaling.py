from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

SCALE_ORDER = (
    "nano",
    "micro",
    "tiny",
    "small",
    "base",
    "medium",
    "large",
    "xl",
    "xxl",
    "huge",
    "ultra",
    "frontier",
    "gpt4",
    "gpt4o",
)

CAPABILITY_ALIASES = {
    "scratch": "nano",
    "toy": "nano",
    "gpt2": "tiny",
    "gpt-2": "tiny",
    "gpt2-medium": "small",
    "gpt2-large": "base",
    "gpt2-xl": "medium",
    "gpt3-small": "base",
    "gpt3": "ultra",
    "gpt-3": "ultra",
    "gpt3.5": "frontier",
    "gpt-3.5": "frontier",
    "llama-7b": "large",
    "llama2-7b": "large",
    "llama2-13b": "xl",
    "llama2-70b": "huge",
    "llama3-8b": "large",
    "llama3-70b": "huge",
    "llama3-405b": "frontier",
    "mistral-7b": "large",
    "mixtral-8x7b": "xl",
    "qwen-7b": "large",
    "qwen-72b": "huge",
    "phi-2": "small",
    "phi-3": "medium",
    "gemma-2b": "base",
    "gemma-7b": "large",
    "gpt-4": "gpt4",
    "gpt-4o": "gpt4o",
    "gpt4-turbo": "gpt4o",
    "o1": "gpt4o",
    "frontier-moe": "gpt4",
}


@dataclass
class ScaleRecipe:
    name: str
    params: int
    n_layer: int
    n_embd: int
    n_head: int
    n_kv_head: int
    n_inner: int
    vocab_size: int
    seq_len: int
    rope_theta: float
    architecture: str
    tokens: int
    global_batch_tokens: int
    learning_rate: float
    min_learning_rate: float
    weight_decay: float
    warmup_ratio: float
    grad_clip: float
    dropout: float
    use_moe: bool = False
    n_experts: int = 1
    n_active_experts: int = 1
    moe_aux_loss: float = 0.01
    stages: list[str] = field(default_factory=list)
    context_final: int = 2048
    tied_embeddings: bool = True
    rms_norm: bool = True
    swiglu: bool = True
    rope: bool = True
    family: str = "llama"
    flops: float = 0.0
    gpu_hours_h100: float = 0.0
    min_vram_gb: float = 0.0
    recommended_gpus: int = 1
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def active_params(self) -> int:
        if not self.use_moe or self.n_experts <= 1:
            return self.params
        dense_frac = 0.5
        return int(self.params * (dense_frac + (1.0 - dense_frac) * self.n_active_experts / self.n_experts))


def estimate_params(
    n_layer: int,
    n_embd: int,
    n_head: int,
    n_kv_head: int,
    n_inner: int,
    vocab_size: int,
    use_moe: bool = False,
    n_experts: int = 1,
    tied: bool = True,
) -> int:
    head_dim = max(n_embd // max(n_head, 1), 1)
    attn = n_embd * (n_head * head_dim) + n_embd * (n_kv_head * head_dim) * 2 + (n_head * head_dim) * n_embd
    ffn_one = n_embd * n_inner * 3 + n_inner * n_embd
    if use_moe and n_experts > 1:
        ffn = ffn_one * n_experts + n_embd * n_experts
    else:
        ffn = ffn_one
    norms = n_embd * 2
    block = attn + ffn + norms
    embed = vocab_size * n_embd
    head = 0 if tied else vocab_size * n_embd
    final_norm = n_embd
    return int(n_layer * block + embed + head + final_norm)


def chinchilla_tokens(params: int, multiplier: float = 20.0) -> int:
    return int(max(params * multiplier, 1_000_000))


def kaplan_tokens(params: int) -> int:
    return int(5.4e3 * (params ** 0.74))


def compute_optimal_lr(params: int, stage: str = "pretrain") -> float:
    n_b = max(params / 1e9, 0.001)
    if stage in {"sft", "finetune"}:
        return min(2e-4, max(5e-6, 3e-4 * (n_b ** -0.3)))
    if stage in {"dpo", "orpo", "kto", "align"}:
        return min(5e-5, max(1e-6, 5e-5 * (n_b ** -0.25)))
    return min(6e-4, max(1e-5, 3.2e-3 * (n_b ** -0.23)))


def global_batch_tokens_for(params: int) -> int:
    n_b = params / 1e9
    if n_b < 0.02:
        return 16_384
    if n_b < 0.15:
        return 65_536
    if n_b < 0.5:
        return 262_144
    if n_b < 2:
        return 524_288
    if n_b < 8:
        return 2_097_152
    if n_b < 20:
        return 4_194_304
    if n_b < 80:
        return 8_388_608
    return 16_777_216


def estimate_flops(params: int, tokens: int, moe_active_frac: float = 1.0) -> float:
    return 6.0 * params * tokens * moe_active_frac


def estimate_h100_hours(flops: float, mfu: float = 0.4) -> float:
    peak = 990e12
    seconds = flops / (peak * mfu)
    return seconds / 3600.0


def estimate_min_vram_gb(params: int, seq_len: int, peft: bool = False, bits: int = 16) -> float:
    bytes_per = bits / 8
    model = params * bytes_per
    if peft:
        trainable = params * 0.02 * 4 * 2
        activations = seq_len * 4096 * 2 * 4
        return (model + trainable + activations) / (1024 ** 3) * 1.3
    adam = params * 8 * 2
    grads = params * bytes_per
    activations = seq_len * min(params / 50, 2e8) * 2
    return (model + adam + grads + activations) / (1024 ** 3) * 1.4


def _recipe(
    name: str,
    n_layer: int,
    n_embd: int,
    n_head: int,
    n_kv_head: int,
    n_inner: int,
    vocab_size: int,
    seq_len: int,
    architecture: str,
    family: str,
    token_mult: float,
    use_moe: bool = False,
    n_experts: int = 1,
    n_active_experts: int = 1,
    stages: list[str] | None = None,
    context_final: int | None = None,
    rope_theta: float = 10000.0,
    notes: str = "",
) -> ScaleRecipe:
    params = estimate_params(
        n_layer, n_embd, n_head, n_kv_head, n_inner, vocab_size,
        use_moe=use_moe, n_experts=n_experts, tied=True,
    )
    tokens = chinchilla_tokens(params, token_mult)
    active = params
    if use_moe and n_experts > 1:
        active = estimate_params(
            n_layer, n_embd, n_head, n_kv_head, n_inner, vocab_size,
            use_moe=True, n_experts=n_active_experts, tied=True,
        )
    moe_frac = active / max(params, 1)
    flops = estimate_flops(params, tokens, moe_frac)
    lr = compute_optimal_lr(params, "pretrain")
    batch_tok = global_batch_tokens_for(params)
    vram = estimate_min_vram_gb(params, seq_len)
    gpus = 1
    if vram > 80:
        gpus = max(1, int(math.ceil(vram / 80)))
    if vram > 640:
        gpus = max(gpus, 64)
    if vram > 5000:
        gpus = max(gpus, 2048)
    return ScaleRecipe(
        name=name,
        params=params,
        n_layer=n_layer,
        n_embd=n_embd,
        n_head=n_head,
        n_kv_head=n_kv_head,
        n_inner=n_inner,
        vocab_size=vocab_size,
        seq_len=seq_len,
        rope_theta=rope_theta,
        architecture=architecture,
        tokens=tokens,
        global_batch_tokens=batch_tok,
        learning_rate=lr,
        min_learning_rate=lr * 0.1,
        weight_decay=0.1,
        warmup_ratio=0.05 if params > 1e9 else 0.1,
        grad_clip=1.0,
        dropout=0.0 if params > 1e8 else 0.1,
        use_moe=use_moe,
        n_experts=n_experts,
        n_active_experts=n_active_experts,
        stages=stages or ["pretrain", "sft"],
        context_final=context_final or seq_len,
        family=family,
        flops=flops,
        gpu_hours_h100=estimate_h100_hours(flops),
        min_vram_gb=round(vram, 2),
        recommended_gpus=gpus,
        notes=notes,
    )


def _build_catalog() -> dict[str, ScaleRecipe]:
    return {
        "nano": _recipe(
            "nano", 4, 128, 4, 4, 512, 8000, 256, "gpt", "gpt", 20.0,
            stages=["pretrain", "sft"],
            notes="CPU-friendly scratch model for tests and toys",
        ),
        "micro": _recipe(
            "micro", 6, 384, 6, 6, 1536, 16000, 512, "gpt", "gpt", 20.0,
            stages=["pretrain", "sft"],
            notes="Laptop CPU/GPU scratch model",
        ),
        "tiny": _recipe(
            "tiny", 12, 768, 12, 12, 3072, 50257, 1024, "gpt2", "gpt2", 20.0,
            stages=["pretrain", "sft"],
            notes="GPT-2 124M class",
        ),
        "small": _recipe(
            "small", 24, 1024, 16, 16, 4096, 50257, 1024, "gpt2", "gpt2", 20.0,
            stages=["pretrain", "sft"],
            notes="GPT-2 medium/large class ~350M",
        ),
        "base": _recipe(
            "base", 24, 2048, 16, 16, 8192, 32000, 2048, "llama", "llama", 20.0,
            stages=["pretrain", "sft", "dpo"],
            notes="~1.3B LLaMA-style",
        ),
        "medium": _recipe(
            "medium", 32, 2560, 20, 4, 10240, 32000, 4096, "llama", "llama", 20.0,
            rope_theta=500000.0,
            stages=["pretrain", "sft", "dpo"],
            notes="~3B GQA model",
        ),
        "large": _recipe(
            "large", 32, 4096, 32, 8, 14336, 128256, 8192, "llama", "llama", 20.0,
            rope_theta=500000.0,
            stages=["pretrain", "sft", "dpo"],
            context_final=131072,
            notes="7-8B class (Llama 3 8B / Mistral 7B)",
        ),
        "xl": _recipe(
            "xl", 40, 5120, 40, 8, 13824, 128256, 8192, "llama", "llama", 20.0,
            rope_theta=500000.0,
            stages=["pretrain", "midtrain", "sft", "dpo"],
            context_final=131072,
            notes="13B class",
        ),
        "xxl": _recipe(
            "xxl", 48, 6656, 52, 8, 17920, 128256, 8192, "llama", "llama", 20.0,
            rope_theta=500000.0,
            stages=["pretrain", "midtrain", "sft", "dpo"],
            context_final=131072,
            notes="30-34B class",
        ),
        "huge": _recipe(
            "huge", 80, 8192, 64, 8, 28672, 128256, 8192, "llama", "llama", 20.0,
            rope_theta=500000.0,
            stages=["pretrain", "midtrain", "sft", "dpo"],
            context_final=131072,
            notes="70B class",
        ),
        "ultra": _recipe(
            "ultra", 96, 12288, 96, 16, 49152, 100277, 8192, "gpt", "gpt", 20.0,
            stages=["pretrain", "sft", "rlhf"],
            context_final=8192,
            notes="GPT-3 175B class",
        ),
        "frontier": _recipe(
            "frontier", 126, 16384, 128, 8, 53248, 128256, 8192, "llama", "llama", 20.0,
            rope_theta=500000.0,
            stages=["pretrain", "midtrain", "sft", "dpo"],
            context_final=131072,
            notes="Llama 3 405B class dense",
        ),
        "gpt4": _recipe(
            "gpt4", 120, 18432, 144, 16, 73728, 100277, 8192, "moe", "gpt", 25.0,
            use_moe=True, n_experts=16, n_active_experts=2,
            stages=["pretrain", "midtrain", "sft", "dpo", "ppo"],
            context_final=32768,
            rope_theta=500000.0,
            notes="GPT-4 class MoE ~1.8T total / ~220B active (public estimates)",
        ),
        "gpt4o": _recipe(
            "gpt4o", 128, 16384, 128, 16, 65536, 200000, 8192, "moe", "gpt", 30.0,
            use_moe=True, n_experts=32, n_active_experts=4,
            stages=["pretrain", "midtrain", "sft", "dpo", "orpo", "ppo"],
            context_final=131072,
            rope_theta=1000000.0,
            notes="GPT-4o class: multimodal MoE, 128k context, full alignment stack",
        ),
    }


RECIPE_CATALOG = _build_catalog()


def resolve_scale(name: str | None, params: int | None = None) -> str:
    if name:
        key = name.strip().lower().replace(" ", "")
        key = CAPABILITY_ALIASES.get(key, key)
        if key in RECIPE_CATALOG:
            return key
        if key in CAPABILITY_ALIASES:
            return CAPABILITY_ALIASES[key]
    if params is not None and params > 0:
        best = "nano"
        best_delta = float("inf")
        for scale, recipe in RECIPE_CATALOG.items():
            delta = abs(math.log10(max(recipe.params, 1)) - math.log10(params))
            if delta < best_delta:
                best_delta = delta
                best = scale
        return best
    return "tiny"


def get_recipe(name: str | None = None, params: int | None = None) -> ScaleRecipe:
    return RECIPE_CATALOG[resolve_scale(name, params)]


def list_recipes() -> list[dict[str, Any]]:
    rows = []
    for name in SCALE_ORDER:
        r = RECIPE_CATALOG[name]
        rows.append({
            "name": r.name,
            "params": r.params,
            "active_params": r.active_params,
            "layers": r.n_layer,
            "hidden": r.n_embd,
            "heads": r.n_head,
            "seq": r.seq_len,
            "tokens": r.tokens,
            "lr": r.learning_rate,
            "moe": r.use_moe,
            "stages": r.stages,
            "min_vram_gb": r.min_vram_gb,
            "h100_hours": round(r.gpu_hours_h100, 1),
            "gpus": r.recommended_gpus,
            "notes": r.notes,
        })
    return rows


def fit_recipe_to_hardware(
    recipe: ScaleRecipe,
    vram_gb: float,
    ram_gb: float,
    gpu_count: int,
    gpu_available: bool,
) -> dict[str, Any]:
    usable_vram = max(vram_gb * gpu_count * 0.85, 0.0) if gpu_available else 0.0
    usable_ram = ram_gb * 0.6
    full_vram = estimate_min_vram_gb(recipe.params, recipe.seq_len, peft=False, bits=16)
    qlora_vram = estimate_min_vram_gb(recipe.params, min(recipe.seq_len, 4096), peft=True, bits=4)
    lora_vram = estimate_min_vram_gb(recipe.params, min(recipe.seq_len, 4096), peft=True, bits=16)

    mode = "scratch_full"
    bits = 16
    peft_method = "none"
    fallback_scale = recipe.name
    seq_len = recipe.seq_len
    reasons: list[str] = []

    if not gpu_available:
        if recipe.params > 50_000_000:
            fallback_scale = "micro" if ram_gb < 16 else "tiny"
            mode = "scratch_cpu"
            seq_len = 256 if ram_gb < 16 else 512
            reasons.append("No GPU: downscaling scratch pretrain to CPU-feasible size")
        else:
            mode = "scratch_cpu"
            seq_len = min(seq_len, 512)
            reasons.append("CPU training enabled for small scratch model")
    elif usable_vram >= full_vram:
        mode = "scratch_full" if recipe.name in {"nano", "micro", "tiny", "small", "base", "medium"} else "full_finetune"
        if recipe.params >= 7_000_000_000 and gpu_count < recipe.recommended_gpus:
            mode = "fsdp_full"
            reasons.append("Using FSDP/ZeRO to shard a large dense/MoE model")
        else:
            reasons.append("Hardware can hold full weights + Adam")
    elif usable_vram >= lora_vram:
        mode = "lora"
        peft_method = "dora"
        reasons.append("Full Adam does not fit; using DoRA/LoRA fine-tune")
    elif usable_vram >= qlora_vram:
        mode = "qlora"
        peft_method = "qlora"
        bits = 4
        reasons.append("Using 4-bit QLoRA to fit the model")
    else:
        for scale in reversed(SCALE_ORDER):
            cand = RECIPE_CATALOG[scale]
            need = estimate_min_vram_gb(cand.params, min(cand.seq_len, 2048), peft=True, bits=4)
            if need <= max(usable_vram, 1.0) or (not gpu_available and cand.params < 20_000_000):
                fallback_scale = scale
                break
        mode = "qlora" if gpu_available else "scratch_cpu"
        peft_method = "qlora" if gpu_available else "none"
        bits = 4 if gpu_available else 32
        seq_len = 512 if not gpu_available else 2048
        reasons.append(f"Target {recipe.name} does not fit; falling back to {fallback_scale}")

    fitted = RECIPE_CATALOG[fallback_scale]
    micro_batch = 1
    if gpu_available:
        per_gpu = max(usable_vram / max(gpu_count, 1), 1.0)
        micro_batch = max(1, min(32, int(per_gpu / max(fitted.min_vram_gb / 8, 0.5))))
    else:
        micro_batch = 1 if ram_gb < 16 else 2

    target_batch_tokens = fitted.global_batch_tokens
    tokens_per_step = max(micro_batch * gpu_count * seq_len, seq_len)
    accum = max(1, int(math.ceil(target_batch_tokens / tokens_per_step)))

    return {
        "requested_scale": recipe.name,
        "fitted_scale": fallback_scale,
        "mode": mode,
        "peft_method": peft_method,
        "bits": bits,
        "seq_len": seq_len,
        "micro_batch_size": micro_batch,
        "gradient_accumulation_steps": accum,
        "effective_batch_tokens": tokens_per_step * accum,
        "learning_rate": compute_optimal_lr(
            fitted.params,
            "pretrain" if mode.startswith("scratch") else "sft",
        ),
        "gradient_checkpointing": mode in {"qlora", "lora", "fsdp_full"} or fitted.params > 1e9,
        "use_fsdp": mode == "fsdp_full" or (gpu_count > 1 and fitted.params > 3e9),
        "use_deepspeed": gpu_count >= 8 and fitted.params > 7e9,
        "flash_attention": gpu_available,
        "bf16": gpu_available and vram_gb >= 16,
        "fp16": gpu_available and vram_gb < 16,
        "cpu_offload": mode == "qlora" and usable_vram < qlora_vram * 1.2,
        "reasons": reasons,
        "recipe": fitted.to_dict(),
        "hardware": {
            "vram_gb": vram_gb,
            "ram_gb": ram_gb,
            "gpu_count": gpu_count,
            "gpu_available": gpu_available,
            "usable_vram_gb": usable_vram,
        },
    }


def plan_training(
    target: str = "tiny",
    vram_gb: float = 0.0,
    ram_gb: float = 16.0,
    gpu_count: int = 0,
    gpu_available: bool = False,
    data_tokens: int | None = None,
) -> dict[str, Any]:
    recipe = get_recipe(target)
    fit = fit_recipe_to_hardware(recipe, vram_gb, ram_gb, gpu_count, gpu_available)
    fitted = get_recipe(fit["fitted_scale"])
    tokens = data_tokens or fitted.tokens
    if fit["mode"] in {"lora", "qlora", "full_finetune"}:
        tokens = min(tokens, max(int(fitted.params * 0.1), 1_000_000))
    steps = max(1, tokens // max(fit["effective_batch_tokens"], 1))
    stage_plan = []
    remaining = tokens
    for i, stage in enumerate(fitted.stages):
        frac = 0.7 if stage == "pretrain" else 0.15 if stage == "midtrain" else 0.1 if stage == "sft" else 0.05
        if i == len(fitted.stages) - 1:
            stage_tokens = remaining
        else:
            stage_tokens = int(tokens * frac)
            remaining -= stage_tokens
        stage_plan.append({
            "stage": stage,
            "tokens": max(stage_tokens, 1),
            "seq_len": fitted.context_final if stage in {"midtrain", "sft", "dpo", "orpo"} else fit["seq_len"],
            "lr": compute_optimal_lr(fitted.params, stage),
            "method": "qlora" if fit["mode"] == "qlora" and stage != "pretrain" else (
                "dora" if fit["peft_method"] == "dora" and stage != "pretrain" else "full"
            ),
        })
    return {
        "target": recipe.name,
        "target_params": recipe.params,
        "fitted": fit,
        "tokens": tokens,
        "steps": steps,
        "stages": stage_plan,
        "flops": estimate_flops(fitted.params, tokens),
        "h100_hours": estimate_h100_hours(estimate_flops(fitted.params, tokens)),
        "honest_gap": {
            "can_match_target": fit["fitted_scale"] == recipe.name and fit["mode"] in {"scratch_full", "fsdp_full"},
            "message": (
                "This run matches the requested scale."
                if fit["fitted_scale"] == recipe.name
                else f"Requested {recipe.name} ({recipe.params:,} params). Hardware will run {fit['fitted_scale']} "
                     f"({fitted.params:,} params) via {fit['mode']}."
            ),
        },
    }
