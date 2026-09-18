from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    F = None
    TORCH_AVAILABLE = False


ALIGNMENT_METHODS = ("dpo", "orpo", "kto", "ipo", "simpo", "slic", "cpo")


@dataclass
class AlignmentConfig:
    method: str = "dpo"
    beta: float = 0.1
    label_smoothing: float = 0.0
    loss_type: str = "sigmoid"
    reference_free: bool = False
    gamma: float = 0.5
    desirable_weight: float = 1.0
    undesirable_weight: float = 1.0
    simpo_gamma: float = 0.5
    orpo_lambda: float = 0.1
    max_length: int = 2048
    max_prompt_length: int = 1024
    learning_rate: float = 5e-6
    epochs: int = 1
    batch_size: int = 2
    gradient_accumulation_steps: int = 8

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AlignmentConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in known})


def _logps_from_logits(logits, labels, prompt_len: int | None = None):
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    log_probs = F.log_softmax(shift_logits, dim=-1)
    gather = torch.gather(log_probs, dim=-1, index=shift_labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    mask = shift_labels.ne(-100)
    if prompt_len is not None:
        idx = torch.arange(mask.size(1), device=mask.device).unsqueeze(0)
        if torch.is_tensor(prompt_len):
            cutoff = (prompt_len.to(mask.device) - 1).clamp(min=0).unsqueeze(1)
        else:
            cutoff = max(prompt_len - 1, 0)
        mask = mask & (idx >= cutoff)
    gather = gather * mask
    seq_logps = gather.sum(dim=-1)
    lengths = mask.sum(dim=-1).clamp(min=1)
    return seq_logps, seq_logps / lengths


def dpo_loss(
    policy_chosen_logps,
    policy_rejected_logps,
    ref_chosen_logps,
    ref_rejected_logps,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
    loss_type: str = "sigmoid",
):
    pi = policy_chosen_logps - policy_rejected_logps
    ref = ref_chosen_logps - ref_rejected_logps
    logits = pi - ref
    if loss_type == "ipo":
        losses = (logits - 1.0 / (2 * beta)) ** 2
    elif loss_type == "hinge":
        losses = F.relu(1.0 - beta * logits)
    else:
        losses = -F.logsigmoid(beta * logits) * (1.0 - label_smoothing) - F.logsigmoid(-beta * logits) * label_smoothing
    chosen_rewards = (policy_chosen_logps - ref_chosen_logps).detach()
    rejected_rewards = (policy_rejected_logps - ref_rejected_logps).detach()
    reward_acc = (chosen_rewards > rejected_rewards).float().mean()
    return losses.mean(), {
        "rewards/chosen": chosen_rewards.mean().item(),
        "rewards/rejected": rejected_rewards.mean().item(),
        "rewards/margin": (chosen_rewards - rejected_rewards).mean().item(),
        "rewards/accuracy": reward_acc.item(),
        "logits": logits.mean().item(),
    }


def orpo_loss(
    policy_chosen_logps,
    policy_rejected_logps,
    chosen_nll,
    lambda_orpo: float = 0.1,
    beta: float = 0.1,
):
    log_odds = (policy_chosen_logps - policy_rejected_logps) - (
        torch.log1p(-torch.exp(policy_chosen_logps).clamp(max=0.999))
        - torch.log1p(-torch.exp(policy_rejected_logps).clamp(max=0.999))
    )
    sig = F.logsigmoid(beta * log_odds)
    loss = chosen_nll - lambda_orpo * sig.mean()
    return loss, {
        "orpo/nll": chosen_nll.item() if hasattr(chosen_nll, "item") else float(chosen_nll),
        "orpo/log_odds": log_odds.mean().item(),
    }


def kto_loss(
    policy_logps,
    ref_logps,
    desirable,
    beta: float = 0.1,
    desirable_weight: float = 1.0,
    undesirable_weight: float = 1.0,
    kl_mean: float | None = None,
):
    kl = (policy_logps - ref_logps).mean() if kl_mean is None else kl_mean
    logits = policy_logps - ref_logps - kl
    des = desirable.bool()
    losses = torch.zeros_like(policy_logps)
    if des.any():
        losses[des] = desirable_weight * (1.0 - F.sigmoid(beta * logits[des]))
    if (~des).any():
        losses[~des] = undesirable_weight * F.sigmoid(beta * logits[~des])
    return losses.mean(), {
        "kto/kl": float(kl.detach().item()) if hasattr(kl, "item") else float(kl),
        "kto/desirable_frac": des.float().mean().item(),
    }


def simpo_loss(
    policy_chosen_avg,
    policy_rejected_avg,
    beta: float = 2.0,
    gamma: float = 0.5,
):
    logits = beta * (policy_chosen_avg - policy_rejected_avg) - gamma
    losses = -F.logsigmoid(logits)
    return losses.mean(), {
        "simpo/margin": (policy_chosen_avg - policy_rejected_avg).mean().item(),
        "simpo/logits": logits.mean().item(),
    }


def cpo_loss(
    policy_chosen_logps,
    policy_rejected_logps,
    chosen_nll,
    beta: float = 0.1,
):
    logits = policy_chosen_logps - policy_rejected_logps
    pref = -F.logsigmoid(beta * logits).mean()
    loss = pref + chosen_nll
    return loss, {"cpo/pref": pref.item(), "cpo/nll": float(chosen_nll)}


def slic_loss(policy_chosen_logps, policy_rejected_logps, delta: float = 1.0):
    losses = F.relu(delta - (policy_chosen_logps - policy_rejected_logps))
    return losses.mean(), {"slic/margin": (policy_chosen_logps - policy_rejected_logps).mean().item()}


def compute_alignment_loss(
    method: str,
    policy_chosen_logps,
    policy_rejected_logps,
    ref_chosen_logps=None,
    ref_rejected_logps=None,
    chosen_nll=None,
    policy_chosen_avg=None,
    policy_rejected_avg=None,
    desirable=None,
    config: AlignmentConfig | None = None,
):
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for alignment losses")
    cfg = config or AlignmentConfig(method=method)
    method = (method or cfg.method).lower()
    if method == "dpo":
        if ref_chosen_logps is None:
            ref_chosen_logps = torch.zeros_like(policy_chosen_logps)
            ref_rejected_logps = torch.zeros_like(policy_rejected_logps)
        return dpo_loss(
            policy_chosen_logps, policy_rejected_logps,
            ref_chosen_logps, ref_rejected_logps,
            beta=cfg.beta, label_smoothing=cfg.label_smoothing, loss_type=cfg.loss_type,
        )
    if method == "ipo":
        if ref_chosen_logps is None:
            ref_chosen_logps = torch.zeros_like(policy_chosen_logps)
            ref_rejected_logps = torch.zeros_like(policy_rejected_logps)
        return dpo_loss(
            policy_chosen_logps, policy_rejected_logps,
            ref_chosen_logps, ref_rejected_logps,
            beta=cfg.beta, loss_type="ipo",
        )
    if method == "orpo":
        if chosen_nll is None:
            chosen_nll = -policy_chosen_logps.mean()
        return orpo_loss(policy_chosen_logps, policy_rejected_logps, chosen_nll, cfg.orpo_lambda, cfg.beta)
    if method == "kto":
        if desirable is None:
            desirable = torch.ones_like(policy_chosen_logps)
        logps = policy_chosen_logps
        ref = ref_chosen_logps if ref_chosen_logps is not None else torch.zeros_like(logps)
        return kto_loss(logps, ref, desirable, cfg.beta, cfg.desirable_weight, cfg.undesirable_weight)
    if method == "simpo":
        cavg = policy_chosen_avg if policy_chosen_avg is not None else policy_chosen_logps
        ravg = policy_rejected_avg if policy_rejected_avg is not None else policy_rejected_logps
        return simpo_loss(cavg, ravg, beta=cfg.beta if cfg.beta != 0.1 else 2.0, gamma=cfg.simpo_gamma)
    if method == "cpo":
        if chosen_nll is None:
            chosen_nll = -policy_chosen_logps.mean()
        return cpo_loss(policy_chosen_logps, policy_rejected_logps, chosen_nll, cfg.beta)
    if method == "slic":
        return slic_loss(policy_chosen_logps, policy_rejected_logps)
    raise ValueError(f"Unknown alignment method: {method}. Choose from {ALIGNMENT_METHODS}")


class AlignmentTrainer:
    def __init__(self, config: AlignmentConfig | None = None):
        self.config = config or AlignmentConfig()

    def pair_logps(self, model, input_ids, attention_mask, labels, prompt_len=None):
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out["logits"] if isinstance(out, dict) else out.logits
        seq, avg = _logps_from_logits(logits, labels, prompt_len)
        nll = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            nll = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return seq, avg, nll

    def step(self, model, batch, ref_model=None):
        method = self.config.method
        c_seq, c_avg, c_nll = self.pair_logps(
            model, batch["chosen_input_ids"], batch.get("chosen_attention_mask"),
            batch["chosen_labels"], batch.get("prompt_len"),
        )
        r_seq, r_avg, _ = self.pair_logps(
            model, batch["rejected_input_ids"], batch.get("rejected_attention_mask"),
            batch["rejected_labels"], batch.get("prompt_len"),
        )
        ref_c = ref_r = None
        if ref_model is not None and method in {"dpo", "ipo", "kto"}:
            with torch.no_grad():
                ref_c, _, _ = self.pair_logps(
                    ref_model, batch["chosen_input_ids"], batch.get("chosen_attention_mask"),
                    batch["chosen_labels"], batch.get("prompt_len"),
                )
                ref_r, _, _ = self.pair_logps(
                    ref_model, batch["rejected_input_ids"], batch.get("rejected_attention_mask"),
                    batch["rejected_labels"], batch.get("prompt_len"),
                )
        return compute_alignment_loss(
            method,
            c_seq, r_seq,
            ref_c, ref_r,
            chosen_nll=c_nll,
            policy_chosen_avg=c_avg,
            policy_rejected_avg=r_avg,
            desirable=batch.get("desirable"),
            config=self.config,
        )
