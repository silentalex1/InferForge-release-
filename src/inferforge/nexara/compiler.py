"""Nexara compiler with hardware optimization and code generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from inferforge.nexara.parser import NexaraModel, NexaraParser


class NexaraCompiler:
    """Compiler for Nexara AI-native code to training configurations."""
    
    def __init__(self):
        self.parser = NexaraParser()
    
    def compile(self, code: str, output_dir: Path) -> dict[str, Any]:
        """Compile Nexara code to training configurations."""
        models = self.parser.parse(code)
        
        if not models:
            raise ValueError("No models found in Nexara code")
        
        compiled_models = {}
        training_configs = {}
        
        for model in models:
            model_config = self._compile_model(model)
            compiled_models[model.name] = model_config
            training_configs[model.name] = model_config["training"]
        
        # Write config to file
        config_path = output_dir / "nexara_config.json"
        with config_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "models": compiled_models,
                    "training_configs": training_configs,
                    "version": "1.0",
                },
                f,
                indent=2,
            )
        
        return {
            "models": compiled_models,
            "training_configs": training_configs,
            "output_path": str(config_path),
        }
    
    def _compile_model(self, model: NexaraModel) -> dict[str, Any]:
        """Compile a single model definition."""
        config = {
            "name": model.name,
            "base_model": model.base_model,
            "task": model.task,
            "training": self._compile_training_config(model),
            "hardware": self._compile_hardware_config(model),
            "dataset": self._compile_dataset_config(model),
            "architecture": self._compile_architecture_config(model),
            "optimization": self._compile_optimization_config(model),
        }
        
        return config
    
    def _compile_training_config(self, model: NexaraModel) -> dict[str, Any]:
        """Compile training configuration."""
        config = model.training_config.copy() if model.training_config else {}
        
        # Set defaults
        config.setdefault("epochs", 3)
        config.setdefault("batch_size", 4)
        config.setdefault("learning_rate", 2e-5)
        config.setdefault("optimizer", "adamw")
        config.setdefault("warmup_steps", 100)
        config.setdefault("weight_decay", 0.01)
        config.setdefault("gradient_accumulation_steps", 1)
        config.setdefault("max_grad_norm", 1.0)
        config.setdefault("lr_scheduler", "cosine")
        config.setdefault("save_steps", 500)
        config.setdefault("eval_steps", 500)
        config.setdefault("logging_steps", 10)
        
        return config
    
    def _compile_hardware_config(self, model: NexaraModel) -> dict[str, Any]:
        """Compile hardware configuration."""
        config = model.hardware_config.copy() if model.hardware_config else {}
        
        # Set defaults
        config.setdefault("prefer_gpu", True)
        config.setdefault("min_ram", 8)
        config.setdefault("min_vram", 4)
        config.setdefault("mixed_precision", True)
        config.setdefault("fp16", True)
        config.setdefault("bf16", False)
        config.setdefault("gradient_checkpointing", False)
        config.setdefault("max_memory_mb", None)
        
        return config
    
    def _compile_dataset_config(self, model: NexaraModel) -> dict[str, Any]:
        """Compile dataset configuration."""
        config = model.dataset_config.copy() if model.dataset_config else {}
        
        # Set defaults
        config.setdefault("type", "custom")
        config.setdefault("examples", 1000)
        config.setdefault("validation_split", 0.1)
        config.setdefault("test_split", 0.0)
        config.setdefault("max_length", 2048)
        config.setdefault("truncation", True)
        config.setdefault("padding", "max_length")
        
        return config
    
    def _compile_architecture_config(self, model: NexaraModel) -> dict[str, Any]:
        """Compile architecture configuration."""
        config = model.architecture_config.copy() if model.architecture_config else {}
        
        # Set defaults based on task
        if model.task == "code-completion":
            config.setdefault("attention_heads", 32)
            config.setdefault("hidden_size", 4096)
            config.setdefault("num_layers", 32)
        elif model.task == "chat":
            config.setdefault("attention_heads", 28)
            config.setdefault("hidden_size", 3584)
            config.setdefault("num_layers", 28)
        
        return config
    
    def _compile_optimization_config(self, model: NexaraModel) -> dict[str, Any]:
        """Compile optimization configuration."""
        config = model.optimization_config.copy() if model.optimization_config else {}
        
        # Set defaults
        config.setdefault("use_lora", False)
        config.setdefault("lora_r", 8)
        config.setdefault("lora_alpha", 16)
        config.setdefault("lora_dropout", 0.05)
        config.setdefault("lora_target_modules", ["q_proj", "v_proj", "k_proj", "o_proj"])
        config.setdefault("quantization", None)
        config.setdefault("pruning", None)
        
        return config
    
    def optimize_for_hardware(
        self, training_config: dict[str, Any], hardware: dict[str, Any]
    ) -> dict[str, Any]:
        """Optimize training configuration for specific hardware."""
        optimized = training_config.copy()
        
        ram_gb = hardware.get("ram", 16)
        gpu_available = hardware.get("gpu_available", False)
        gpu_memory_mb = hardware.get("gpu_memory", 0)
        cpu_cores = hardware.get("cpu_cores", 4)
        
        # Adjust batch size based on available memory
        if gpu_available and gpu_memory_mb > 0:
            # GPU available - optimize for GPU memory
            if gpu_memory_mb < 8000:  # < 8GB
                optimized["batch_size"] = min(optimized.get("batch_size", 4), 2)
                optimized["gradient_accumulation_steps"] = 4
                optimized["gradient_checkpointing"] = True
            elif gpu_memory_mb < 16000:  # < 16GB
                optimized["batch_size"] = min(optimized.get("batch_size", 4), 4)
                optimized["gradient_accumulation_steps"] = 2
            else:  # >= 16GB
                optimized["batch_size"] = optimized.get("batch_size", 8)
                optimized["gradient_accumulation_steps"] = 1
            
            # Enable mixed precision on GPU
            optimized["fp16"] = True
            optimized["use_cuda"] = True
        else:
            # CPU only - optimize for RAM
            if ram_gb < 16:
                optimized["batch_size"] = 1
                optimized["gradient_accumulation_steps"] = 8
            elif ram_gb < 32:
                optimized["batch_size"] = 2
                optimized["gradient_accumulation_steps"] = 4
            else:
                optimized["batch_size"] = 4
                optimized["gradient_accumulation_steps"] = 2
            
            optimized["fp16"] = False
            optimized["use_cuda"] = False
        
        # Set number of dataloader workers based on CPU cores
        optimized["dataloader_num_workers"] = min(cpu_cores // 2, 8)
        
        # Adjust learning rate based on effective batch size
        effective_batch_size = (
            optimized["batch_size"] * optimized.get("gradient_accumulation_steps", 1)
        )
        if effective_batch_size != training_config.get("batch_size", 4):
            # Scale learning rate with batch size (linear scaling rule)
            base_lr = training_config.get("learning_rate", 2e-5)
            scale_factor = effective_batch_size / training_config.get("batch_size", 4)
            optimized["learning_rate"] = base_lr * scale_factor
        
        # Memory optimization flags
        if ram_gb < 32 or (gpu_available and gpu_memory_mb < 12000):
            optimized["max_memory_mb"] = int(ram_gb * 1024 * 0.7)  # Use 70% of RAM
            optimized["gradient_checkpointing"] = True
        
        return optimized
    
    def generate_python_code(
        self, compiled: dict[str, Any], output_path: Path
    ) -> Path:
        """Generate Python training script from compiled configuration."""
        script_content = '''"""
Auto-generated training script from Nexara compilation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
    )
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, TaskType
except ImportError as e:
    print(f"Error: Missing required package: {e}")
    print("Install with: pip install torch transformers datasets peft accelerate")
    sys.exit(1)


def load_config(config_path: Path) -> dict:
    """Load Nexara configuration."""
    with config_path.open("r") as f:
        return json.load(f)


def setup_model_and_tokenizer(base_model: str, optimization_config: dict):
    """Initialize model and tokenizer."""
    print(f"Loading base model: {base_model}")
    
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16 if optimization_config.get("use_cuda") else torch.float32,
        device_map="auto" if optimization_config.get("use_cuda") else None,
    )
    
    # Apply LoRA if configured
    if optimization_config.get("use_lora"):
        print("Applying LoRA configuration...")
        lora_config = LoraConfig(
            r=optimization_config.get("lora_r", 8),
            lora_alpha=optimization_config.get("lora_alpha", 16),
            lora_dropout=optimization_config.get("lora_dropout", 0.05),
            target_modules=optimization_config.get("lora_target_modules", ["q_proj", "v_proj"]),
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    return model, tokenizer


def _dataset_source(dataset_config: dict) -> str | None:
    for key in ("path", "source", "file", "name", "dataset"):
        value = dataset_config.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _record_to_text(record: dict) -> str:
    """Flatten common record schemas (alpaca / chat / prompt-completion / text) to one string."""
    if "text" in record and record["text"]:
        return str(record["text"])
    messages = record.get("messages") or record.get("conversations")
    if isinstance(messages, list):
        parts = []
        for msg in messages:
            role = str(msg.get("role") or msg.get("from") or "user")
            content = msg.get("content") or msg.get("value") or ""
            parts.append(f"<|im_start|>{role}\\n{content}<|im_end|>")
        return "\\n".join(parts)
    if "instruction" in record:
        prompt = str(record.get("instruction", ""))
        if record.get("input"):
            prompt += "\\n" + str(record["input"])
        return prompt + "\\n" + str(record.get("output", ""))
    prompt = record.get("prompt") or record.get("input") or ""
    answer = record.get("completion") or record.get("response") or record.get("output") or ""
    return f"{prompt}\\n{answer}".strip()


def load_raw_dataset(dataset_config: dict):
    """Load the dataset referenced by the Nexara config (local file, directory, or HF hub id)."""
    from datasets import Dataset, load_dataset

    source = _dataset_source(dataset_config)
    split = dataset_config.get("split", "train")
    if source is None:
        n = int(dataset_config.get("examples", 100) or 100)
        print("WARNING: no dataset path configured; using synthetic placeholder text.")
        return Dataset.from_dict({"text": ["Example training text " + str(i) for i in range(n)]})

    path = Path(source)
    if path.exists():
        suffix = path.suffix.lower()
        if path.is_dir():
            files = [str(p) for p in path.rglob("*") if p.suffix.lower() in {".jsonl", ".json", ".txt", ".csv", ".parquet"}]
            if not files:
                raise FileNotFoundError(f"No data files found under {path}")
            suffix = Path(files[0]).suffix.lower()
            data_files = files
        else:
            data_files = str(path)
        fmt = {".jsonl": "json", ".json": "json", ".txt": "text", ".csv": "csv", ".parquet": "parquet"}.get(suffix)
        if fmt is None:
            raise ValueError(f"Unsupported dataset file type: {suffix}")
        return load_dataset(fmt, data_files=data_files, split="train")

    print(f"Loading dataset from Hugging Face Hub: {source}")
    return load_dataset(source, dataset_config.get("config"), split=split)


def prepare_dataset(dataset_config: dict, tokenizer):
    """Prepare and tokenize the training dataset."""
    print("Preparing dataset...")
    max_length = dataset_config.get("max_length", 2048)
    text_field = dataset_config.get("text_field", "text")

    dataset = load_raw_dataset(dataset_config)
    max_examples = dataset_config.get("max_examples")
    if max_examples:
        dataset = dataset.select(range(min(int(max_examples), len(dataset))))

    if text_field not in dataset.column_names:
        dataset = dataset.map(lambda rec: {"text": _record_to_text(rec)}, remove_columns=dataset.column_names)
        text_field = "text"

    def tokenize_function(examples):
        return tokenizer(
            examples[text_field],
            truncation=dataset_config.get("truncation", True),
            max_length=max_length,
            padding=dataset_config.get("padding", "max_length"),
        )

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
    print(f"Dataset ready: {len(tokenized_dataset)} examples")
    return tokenized_dataset


def train_with_universal_trainer(config: dict, model_name: str) -> bool:
    """Train via InferForge's UniversalTrainer when it is installed. Returns False to fall back."""
    try:
        from inferforge.nexara.universal_trainer import UniversalTrainConfig, UniversalTrainer
    except ImportError:
        return False

    model_config = config["models"][model_name]
    training_config = model_config["training"]
    optimization_config = model_config["optimization"]
    dataset_config = model_config["dataset"]
    source = _dataset_source(dataset_config)
    if source is None:
        return False

    print(f"\\n=== Training Model (UniversalTrainer): {model_name} ===")
    cfg = UniversalTrainConfig(
        output_dir=f"./output/{model_name}",
        model=model_config.get("base_model"),
        data=source,
        text_field=dataset_config.get("text_field", "text"),
        max_samples=dataset_config.get("max_examples"),
        seq_len=dataset_config.get("max_length"),
        epochs=int(training_config.get("epochs", 3)),
        batch_size=training_config.get("batch_size"),
        gradient_accumulation_steps=training_config.get("gradient_accumulation_steps"),
        learning_rate=training_config.get("learning_rate"),
        weight_decay=training_config.get("weight_decay", 0.01),
        max_grad_norm=training_config.get("max_grad_norm", 1.0),
        scheduler=training_config.get("lr_scheduler", "cosine"),
        peft_method="lora" if optimization_config.get("use_lora") else "auto",
        lora_r=optimization_config.get("lora_r", 16),
        lora_alpha=optimization_config.get("lora_alpha", 32),
        lora_dropout=optimization_config.get("lora_dropout", 0.05),
        lora_target_modules=optimization_config.get("lora_target_modules"),
        gradient_checkpointing=training_config.get("gradient_checkpointing"),
        fp16=training_config.get("fp16"),
        bf16=training_config.get("bf16"),
        logging_steps=training_config.get("logging_steps", 10),
        save_steps=training_config.get("save_steps", 500),
        eval_steps=training_config.get("eval_steps", 500),
    )
    result = UniversalTrainer(cfg).train()
    print(f"\\nTraining complete for {model_name}: {json.dumps(result, default=str)[:500]}")
    return True


def train_model(config: dict, model_name: str):
    """Train a model using Nexara configuration."""
    model_config = config["models"][model_name]
    training_config = model_config["training"]
    hardware_config = model_config["hardware"]
    dataset_config = model_config["dataset"]
    optimization_config = model_config["optimization"]
    
    print(f"\\n=== Training Model: {model_name} ===")
    print(f"Base model: {model_config['base_model']}")
    print(f"Task: {model_config['task']}")
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(
        model_config["base_model"],
        {**training_config, **optimization_config}
    )
    
    # Prepare dataset
    train_dataset = prepare_dataset(dataset_config, tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=f"./output/{model_name}",
        num_train_epochs=training_config.get("epochs", 3),
        per_device_train_batch_size=training_config.get("batch_size", 4),
        gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 1),
        learning_rate=training_config.get("learning_rate", 2e-5),
        lr_scheduler_type=training_config.get("lr_scheduler", "cosine"),
        warmup_steps=training_config.get("warmup_steps", 100),
        weight_decay=training_config.get("weight_decay", 0.01),
        max_grad_norm=training_config.get("max_grad_norm", 1.0),
        logging_steps=training_config.get("logging_steps", 10),
        save_steps=training_config.get("save_steps", 500),
        eval_steps=training_config.get("eval_steps", 500),
        fp16=training_config.get("fp16", False),
        bf16=training_config.get("bf16", False),
        gradient_checkpointing=training_config.get("gradient_checkpointing", False),
        dataloader_num_workers=training_config.get("dataloader_num_workers", 4),
        remove_unused_columns=False,
        report_to="none",
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    # Train
    print("\\nStarting training...")
    trainer.train()
    
    # Save model
    output_dir = Path(f"./output/{model_name}/final")
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    
    print(f"\\nModel saved to: {output_dir}")
    print(f"Training complete for {model_name}!")


def main():
    """Main training entry point."""
    config_path = Path("nexara_config.json")
    
    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)
    
    config = load_config(config_path)
    
    print("=== Nexara Training System ===")
    print(f"Configuration version: {config.get('version', 'unknown')}")
    print(f"Models to train: {len(config['models'])}")
    
    # Train each model (prefer InferForge's UniversalTrainer, fall back to HF Trainer)
    use_universal = "--hf-trainer" not in sys.argv
    for model_name in config["models"]:
        try:
            if use_universal and train_with_universal_trainer(config, model_name):
                continue
            train_model(config, model_name)
        except Exception as e:
            print(f"\\nError training {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\\n=== All training complete ===")


if __name__ == "__main__":
    main()
'''
        
        output_path.write_text(script_content, encoding="utf-8")
        return output_path
