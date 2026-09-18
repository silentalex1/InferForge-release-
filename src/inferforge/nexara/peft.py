"""Parameter-Efficient Fine-Tuning (PEFT) with LoRA and QLoRA support."""

from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class LoRAConfig:
    r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: list[str] = None
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


class LoRALayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, r: int, alpha: int, dropout: float = 0.0):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        
        self.lora_A = nn.Parameter(torch.randn(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")
        
        result = F.linear(x, self.lora_B @ self.lora_A) * self.scaling
        return self.dropout(result)


class LoRALinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, config: LoRAConfig):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.config = config
        
        self.linear = nn.Linear(in_features, out_features, bias=config.bias != "none")
        
        if config.r > 0:
            self.lora = LoRALayer(in_features, out_features, config.r, config.lora_alpha, config.lora_dropout)
        else:
            self.lora = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.linear(x)
        
        if self.lora is not None:
            result = result + self.lora(x)
        
        return result
    
    def merge_weights(self) -> None:
        if self.lora is None:
            return
        
        with torch.no_grad():
            delta_weight = self.lora.lora_B @ self.lora.lora_A * self.lora.scaling
            self.linear.weight.data += delta_weight
            self.lora = None


class QLoRALayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, r: int, alpha: int, 
                 bits: int = 4, dropout: float = 0.0):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.bits = bits
        
        self.lora_A = nn.Parameter(torch.randn(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        
        self.quant_scale = nn.Parameter(torch.ones(out_features))
        self.quant_zero_point = nn.Parameter(torch.zeros(out_features))
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.quant_scale.view(-1, 1)
        zero_point = self.quant_zero_point.view(-1, 1)
        
        qmin = 0
        qmax = (1 << self.bits) - 1
        
        x_quant = torch.clamp(torch.round(x / scale) + zero_point, qmin, qmax)
        x_dequant = (x_quant - zero_point) * scale
        
        return x_dequant
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")
        
        result = F.linear(x, self.lora_B @ self.lora_A) * self.scaling
        result = self.quantize(result)
        return self.dropout(result)


class QLoRALinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, config: LoRAConfig, bits: int = 4):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.config = config
        self.bits = bits
        
        self.linear = nn.Linear(in_features, out_features, bias=config.bias != "none")
        
        if config.r > 0:
            self.lora = QLoRALayer(in_features, out_features, config.r, config.lora_alpha, bits, config.lora_dropout)
        else:
            self.lora = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.linear(x)
        
        if self.lora is not None:
            result = result + self.lora(x)
        
        return result


class AdapterLayer(nn.Module):
    def __init__(self, embed_dim: int, bottleneck_dim: int = 64, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.bottleneck_dim = bottleneck_dim
        
        self.down_proj = nn.Linear(embed_dim, bottleneck_dim)
        self.activation = nn.GELU()
        self.up_proj = nn.Linear(bottleneck_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.down_proj(x)
        x = self.activation(x)
        x = self.up_proj(x)
        x = self.dropout(x)
        return residual + x


class PrefixTuning(nn.Module):
    def __init__(self, embed_dim: int, num_prefix_tokens: int = 10):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_prefix_tokens = num_prefix_tokens
        
        self.prefix_embeddings = nn.Parameter(torch.randn(num_prefix_tokens, embed_dim))
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size = hidden_states.size(0)
        
        prefix = self.prefix_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        
        hidden_states = torch.cat([prefix, hidden_states], dim=1)
        
        return hidden_states


class PromptTuning(nn.Module):
    def __init__(self, vocab_size: int, prompt_length: int = 10):
        super().__init__()
        self.vocab_size = vocab_size
        self.prompt_length = prompt_length
        
        self.prompt_embeddings = nn.Parameter(torch.randn(prompt_length, vocab_size))
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size = input_ids.size(0)
        
        prompt_ids = torch.argmax(self.prompt_embeddings, dim=-1)
        prompt_ids = prompt_ids.unsqueeze(0).expand(batch_size, -1)
        
        input_ids = torch.cat([prompt_ids, input_ids], dim=1)
        
        return input_ids


class LoRAModel:
    def __init__(self, model: nn.Module, config: LoRAConfig):
        self.model = model
        self.config = config
        self.lora_layers = []
        
        self._apply_lora()
    
    def _apply_lora(self) -> None:
        target_modules = self.config.target_modules or ["q_proj", "k_proj", "v_proj", "o_proj"]
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                for target in target_modules:
                    if target in name:
                        self._replace_with_lora(name, module)
                        break
    
    def _replace_with_lora(self, name: str, module: nn.Linear) -> None:
        lora_linear = LoRALinear(
            module.in_features,
            module.out_features,
            self.config
        )
        
        lora_linear.linear.weight.data = module.weight.data.clone()
        if module.bias is not None:
            lora_linear.linear.bias.data = module.bias.data.clone()
        
        self._set_module_by_name(name, lora_linear)
        self.lora_layers.append(name)
    
    def _set_module_by_name(self, name: str, new_module: nn.Module) -> None:
        parts = name.split('.')
        parent = self.model
        
        for part in parts[:-1]:
            parent = getattr(parent, part)
        
        setattr(parent, parts[-1], new_module)
    
    def get_trainable_parameters(self) -> list[nn.Parameter]:
        trainable = []
        
        for name, param in self.model.named_parameters():
            if any(lora_name in name for lora_name in self.lora_layers):
                if "lora" in name:
                    trainable.append(param)
        
        return trainable
    
    def freeze_base_model(self) -> None:
        for name, param in self.model.named_parameters():
            if "lora" not in name:
                param.requires_grad = False
    
    def unfreeze_all(self) -> None:
        for param in self.model.parameters():
            param.requires_grad = True
    
    def merge_lora_weights(self) -> None:
        for name, module in self.model.named_modules():
            if isinstance(module, LoRALinear):
                module.merge_weights()
    
    def save_lora_weights(self, path: str) -> None:
        lora_state_dict = {}
        
        for name, param in self.model.named_parameters():
            if "lora" in name:
                lora_state_dict[name] = param.data
        
        torch.save(lora_state_dict, path)
    
    def load_lora_weights(self, path: str) -> None:
        lora_state_dict = torch.load(path)
        
        for name, param in self.model.named_parameters():
            if name in lora_state_dict:
                param.data = lora_state_dict[name]


class QLoRAModel:
    def __init__(self, model: nn.Module, config: LoRAConfig, bits: int = 4):
        self.model = model
        self.config = config
        self.bits = bits
        self.lora_layers = []
        
        self._apply_qlora()
    
    def _apply_qlora(self) -> None:
        target_modules = self.config.target_modules or ["q_proj", "k_proj", "v_proj", "o_proj"]
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                for target in target_modules:
                    if target in name:
                        self._replace_with_qlora(name, module)
                        break
    
    def _replace_with_qlora(self, name: str, module: nn.Linear) -> None:
        qlora_linear = QLoRALinear(
            module.in_features,
            module.out_features,
            self.config,
            self.bits
        )
        
        qlora_linear.linear.weight.data = module.weight.data.clone()
        if module.bias is not None:
            qlora_linear.linear.bias.data = module.bias.data.clone()
        
        self._set_module_by_name(name, qlora_linear)
        self.lora_layers.append(name)
    
    def _set_module_by_name(self, name: str, new_module: nn.Module) -> None:
        parts = name.split('.')
        parent = self.model
        
        for part in parts[:-1]:
            parent = getattr(parent, part)
        
        setattr(parent, parts[-1], new_module)
    
    def get_trainable_parameters(self) -> list[nn.Parameter]:
        trainable = []
        
        for name, param in self.model.named_parameters():
            if any(lora_name in name for lora_name in self.lora_layers):
                if "lora" in name:
                    trainable.append(param)
        
        return trainable
    
    def freeze_base_model(self) -> None:
        for name, param in self.model.named_parameters():
            if "lora" not in name:
                param.requires_grad = False
    
    def quantize_base_model(self) -> None:
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and "lora" not in name:
                self._quantize_linear(module)
    
    def _quantize_linear(self, module: nn.Linear) -> None:
        weight = module.weight.data
        scale = weight.abs().max() / ((1 << self.bits) - 1)
        quantized_weight = (weight / scale).round() * scale
        module.weight.data = quantized_weight


import math
