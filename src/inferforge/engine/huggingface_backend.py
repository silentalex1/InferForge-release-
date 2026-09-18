"""HuggingFace transformers backend for InferForge."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Iterator

from inferforge.core.registry import ModelRecord
from inferforge.engine.base import ChatEngine, ChatMessage, GenerationConfig


class HuggingFaceEngine(ChatEngine):
    """Engine for running models using HuggingFace transformers."""
    
    def __init__(
        self,
        model: ModelRecord,
        device: str = "auto",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        use_flash_attention: bool = False,
        trust_remote_code: bool = False,
    ):
        """Initialize HuggingFace engine.
        
        Args:
            model: Model record with HF model identifier
            device: Device to load model on ('auto', 'cuda', 'cpu')
            load_in_8bit: Enable 8-bit quantization
            load_in_4bit: Enable 4-bit quantization
            use_flash_attention: Enable Flash Attention 2
            trust_remote_code: Allow custom model code execution
        """
        self.model_record = model
        self.device = device
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self.use_flash_attention = use_flash_attention
        self.trust_remote_code = trust_remote_code
        
        self.model = None
        self.tokenizer = None
        self._initialized = False
    
    def initialize(self) -> None:
        """Load model and tokenizer."""
        if self._initialized:
            return
        
        try:
            import torch
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
            )
        except ImportError as e:
            raise RuntimeError(
                f"HuggingFace backend requires transformers and torch: {e}\n"
                "Install with: pip install transformers torch accelerate bitsandbytes"
            ) from e
        
        local_path = self.model_record.path
        if local_path and Path(local_path).is_dir() and (Path(local_path) / "config.json").exists():
            model_id = local_path  # locally fine-tuned model (e.g. from `forge train`)
        else:
            model_id = self.model_record.meta.get("hf_model_id") or self.model_record.name
        
        # Configure quantization
        quantization_config = None
        if self.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif self.load_in_8bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=self.trust_remote_code,
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        cuda_ok = torch.cuda.is_available()
        device = "cuda" if (self.device == "auto" and cuda_ok) else ("cpu" if self.device == "auto" else self.device)
        model_kwargs = {"trust_remote_code": self.trust_remote_code}
        if device != "cpu":
            # device_map="auto" requires accelerate; only use it when there is a GPU to spread to
            model_kwargs["device_map"] = device

        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
        else:
            model_kwargs["dtype"] = torch.float16 if device != "cpu" else torch.float32
        
        if self.use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"
        
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        except TypeError:
            # transformers <5.x doesn't accept `dtype`
            if "dtype" in model_kwargs:
                model_kwargs["torch_dtype"] = model_kwargs.pop("dtype")
            self.model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)

        if model_kwargs.get("device_map") is None:
            self.model = self.model.to(device)

        self.model.eval()
        self._initialized = True
    
    def generate(
        self,
        messages: list[ChatMessage],
        config: GenerationConfig | None = None,
    ) -> str:
        """Generate a complete response."""
        if not self._initialized:
            self.initialize()
        
        config = config or GenerationConfig()

        # Format messages using chat template
        prompt = self._format_chat(messages)

        # Tokenize — leave room for the response inside the model's context window
        gen_tokens = config.max_tokens or 512
        ctx = getattr(getattr(self.model, "config", None), "max_position_embeddings", 2048) or 2048
        max_input = max(256, min(ctx, 32768) - gen_tokens - 8)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input,
        )

        if self.model.device.type != "cpu":
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Generate
        with __import__("torch").no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=config.max_tokens or 512,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                do_sample=config.temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        # <|im_start|>/<|im_end|> are plain text for templateless models — stop at turn end
        for marker in ("<|im_start|>", "<|im_end|>"):
            if marker in response:
                response = response.split(marker)[0]

        return response.strip()
    
    def stream(
        self,
        messages: list[ChatMessage],
        config: GenerationConfig | None = None,
    ) -> Iterator[str]:
        """Stream response tokens."""
        if not self._initialized:
            self.initialize()
        
        config = config or GenerationConfig()

        # Format messages
        prompt = self._format_chat(messages)

        # Tokenize — same context budgeting as generate()
        gen_tokens = config.max_tokens or 512
        ctx = getattr(getattr(self.model, "config", None), "max_position_embeddings", 2048) or 2048
        max_input = max(256, min(ctx, 32768) - gen_tokens - 8)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input,
        )

        if self.model.device.type != "cpu":
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Streaming generation
        from threading import Thread

        from transformers import TextIteratorStreamer
        
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        
        generation_kwargs = {
            **inputs,
            "max_new_tokens": config.max_tokens or 512,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": config.top_k,
            "do_sample": config.temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "streamer": streamer,
        }
        
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        
        for text in streamer:
            yield text
    
    def _format_chat(self, messages: list[ChatMessage]) -> str:
        """Format messages using the tokenizer's chat template, falling back to
        the ``<|im_start|>`` format our fine-tuned models are trained on."""
        # Agent-trained models were SFT'd on im_start markup — use it directly so the
        # tool-call format survives even when a chat template exists.
        agentic = bool(getattr(self.model_record, "meta", {}) and self.model_record.meta.get("agentic"))
        if hasattr(self.tokenizer, "apply_chat_template") and not agentic:
            formatted_messages = [
                {"role": msg.role, "content": msg.content}
                for msg in messages
            ]
            try:
                return self.tokenizer.apply_chat_template(
                    formatted_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass  # no chat template (e.g. gpt2) — use im_start fallback

        formatted = ""
        for msg in messages:
            role = msg.role if msg.role in {"system", "user", "assistant", "tool"} else "user"
            formatted += f"<|im_start|>{role}\n{msg.content}<|im_end|>\n"
        formatted += "<|im_start|>assistant\n"
        return formatted
    
    def chat(
        self,
        messages: list[ChatMessage],
        system: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> str:
        """ChatEngine API: map the legacy ``options`` dict onto GenerationConfig."""
        opts = options or {}
        msgs = list(messages)
        if not any(m.role == "system" for m in msgs):
            if system is None and getattr(self.model_record, "meta", {}) and self.model_record.meta.get("agentic"):
                # Agent-fine-tuned models expect the training system prompt
                from inferforge.training.agent_dataset import AGENT_SYSTEM_PROMPT
                system = AGENT_SYSTEM_PROMPT
            if system:
                msgs.insert(0, ChatMessage(role="system", content=system))
        cfg = GenerationConfig(
            max_tokens=int(opts.get("max_tokens") or opts.get("num_predict") or 512),
            temperature=float(opts.get("temperature", 0.7)),
            top_p=float(opts.get("top_p", 0.95)),
            top_k=int(opts.get("top_k", 40)),
            stop=opts.get("stop") or GenerationConfig().stop,
        )
        text = self.generate(msgs, cfg)
        for marker in cfg.stop or []:
            if marker and marker in text:
                text = text.split(marker)[0]
        return text.strip()

    def stream_chat(
        self,
        messages: list[ChatMessage],
        system: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> Iterator[str]:
        opts = options or {}
        msgs = list(messages)
        if system and not any(m.role == "system" for m in msgs):
            msgs.insert(0, ChatMessage(role="system", content=system))
        cfg = GenerationConfig(
            max_tokens=int(opts.get("max_tokens") or opts.get("num_predict") or 512),
            temperature=float(opts.get("temperature", 0.7)),
            top_p=float(opts.get("top_p", 0.95)),
            top_k=int(opts.get("top_k", 40)),
            stop=opts.get("stop") or GenerationConfig().stop,
        )
        buf = ""
        for piece in self.stream(msgs, cfg):
            buf += piece
            if any(s and s in buf for s in cfg.stop or []) or "<|im_start|>" in buf or "<|im_end|>" in buf:
                # trim the piece at the stop marker and end the stream
                cut = buf
                for marker in (list(cfg.stop or []) + ["<|im_start|>", "<|im_end|>"]):
                    if marker and marker in cut:
                        cut = cut.split(marker)[0]
                tail = cut[len(buf) - len(piece):]
                if tail:
                    yield tail
                return
            yield piece

    def close(self) -> None:
        """Clean up model resources."""
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        # Force garbage collection
        gc.collect()
        
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        
        self._initialized = False
    
    def get_info(self) -> dict[str, Any]:
        """Get engine information."""
        info = {
            "backend": "huggingface",
            "model": self.model_record.name,
            "device": self.device,
            "quantization": None,
            "flash_attention": self.use_flash_attention,
        }
        
        if self.load_in_4bit:
            info["quantization"] = "4bit"
        elif self.load_in_8bit:
            info["quantization"] = "8bit"
        
        if self._initialized and self.model is not None:
            info["loaded"] = True
            try:
                import torch
                info["device_type"] = self.model.device.type
                if torch.cuda.is_available():
                    info["gpu_memory_allocated_gb"] = torch.cuda.memory_allocated() / 1024**3
                    info["gpu_memory_reserved_gb"] = torch.cuda.memory_reserved() / 1024**3
            except Exception:
                pass
        else:
            info["loaded"] = False
        
        return info
