from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from inferforge.nexara.scaling import get_recipe, plan_training
from inferforge.nexara.universal_trainer import UniversalTrainConfig, UniversalTrainer

ProgressCb = Callable[[str, float, dict[str, Any] | None], None]


@dataclass
class PipelineStage:
    name: str
    stage: str
    enabled: bool = True
    from_scratch: bool = False
    peft_method: str = "auto"
    epochs: int = 1
    max_steps: int | None = None
    data: str | list | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PipelineConfig:
    name: str = "nexara-pipeline"
    target: str = "tiny"
    output_dir: str = "./nexara_pipeline"
    base_model: str | None = None
    data: str | list | None = None
    sft_data: str | list | None = None
    preference_data: str | list | None = None
    stages: list[PipelineStage] = field(default_factory=list)
    merge_adapters: bool = True
    export_gguf: bool = False
    quantize: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


def default_stages_for(target: str, from_scratch: bool, has_pref: bool) -> list[PipelineStage]:
    recipe = get_recipe(target)
    stages: list[PipelineStage] = []
    for name in recipe.stages:
        if name == "pretrain" and not from_scratch:
            continue
        if name in {"dpo", "orpo", "kto", "ppo", "rlhf"} and not has_pref and name != "sft":
            if not has_pref:
                continue
        stages.append(PipelineStage(
            name=name,
            stage=name,
            from_scratch=from_scratch and name == "pretrain",
            peft_method="none" if name == "pretrain" else "auto",
            epochs=1 if name != "pretrain" else 1,
        ))
    if not stages:
        stages.append(PipelineStage(name="sft", stage="sft", from_scratch=from_scratch, peft_method="auto"))
    return stages


class TrainingPipeline:
    def __init__(self, config: PipelineConfig, hardware: dict[str, Any] | None = None):
        self.config = config
        self.hardware = hardware or {}
        self.results: list[dict[str, Any]] = []

    def build_plan(self) -> dict[str, Any]:
        cfg = self.config
        plan = plan_training(
            target=cfg.target,
            vram_gb=float(self.hardware.get("gpu_memory", 0) or 0) / (1024 if float(self.hardware.get("gpu_memory", 0) or 0) > 256 else 1),
            ram_gb=float(self.hardware.get("ram", 16) or 16),
            gpu_count=int(self.hardware.get("gpu_count", 0) or 0),
            gpu_available=bool(self.hardware.get("gpu_available", False)),
        )
        stages = cfg.stages or default_stages_for(
            plan["fitted"]["fitted_scale"],
            from_scratch=cfg.base_model in {None, "", "scratch"},
            has_pref=cfg.preference_data is not None,
        )
        return {"plan": plan, "stages": [s.to_dict() for s in stages]}

    def run(self, progress: ProgressCb | None = None) -> dict[str, Any]:
        cfg = self.config
        out = Path(cfg.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        blueprint = self.build_plan()
        (out / "pipeline.json").write_text(json.dumps(blueprint, indent=2, default=str), encoding="utf-8")
        stages = cfg.stages or default_stages_for(
            blueprint["plan"]["fitted"]["fitted_scale"],
            from_scratch=cfg.base_model in {None, "", "scratch"},
            has_pref=cfg.preference_data is not None,
        )
        current_model = cfg.base_model
        from_scratch_next = cfg.base_model in {None, "", "scratch"}
        n = max(len(stages), 1)
        for i, stage in enumerate(stages):
            if not stage.enabled:
                continue
            stage_dir = out / f"{i:02d}_{stage.name}"
            data = stage.data
            if data is None:
                if stage.stage in {"dpo", "orpo", "kto", "ipo", "simpo", "cpo"}:
                    data = cfg.preference_data or cfg.data
                elif stage.stage == "sft":
                    data = cfg.sft_data or cfg.data
                else:
                    data = cfg.data
            train_cfg = UniversalTrainConfig(
                output_dir=str(stage_dir),
                model=None if from_scratch_next else current_model,
                from_scratch=from_scratch_next or stage.from_scratch,
                scale=blueprint["plan"]["fitted"]["fitted_scale"],
                stage=stage.stage,
                data=data,
                epochs=stage.epochs,
                max_steps=stage.max_steps,
                peft_method=stage.peft_method,
                alignment_method=stage.stage if stage.stage in {"dpo", "orpo", "kto", "ipo", "simpo"} else "dpo",
                **stage.extra,
            )
            trainer = UniversalTrainer(train_cfg, hardware=self.hardware)
            if progress:
                progress(f"stage:{stage.name}", i / n, {"stage": stage.name})
            result = trainer.train(progress=progress)
            result["stage_name"] = stage.name
            self.results.append(result)
            current_model = result.get("output_dir") or str(stage_dir / "final")
            from_scratch_next = False
            if cfg.merge_adapters and stage.peft_method not in {"none", "full", "off", "auto"}:
                merged = self._merge_adapter(current_model, str(stage_dir / "merged"))
                if merged:
                    current_model = merged
        final_dir = out / "final"
        if current_model and Path(current_model).exists() and Path(current_model).resolve() != final_dir.resolve():
            self._copy_model(current_model, str(final_dir))
        if cfg.quantize:
            self._quantize(str(final_dir), cfg.quantize)
        if cfg.export_gguf:
            self._export_gguf(str(final_dir), str(out / "model.gguf"))
        summary = {
            "status": "completed",
            "name": cfg.name,
            "stages": self.results,
            "final_model": str(final_dir),
            "plan": blueprint["plan"],
        }
        (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        return summary

    def _merge_adapter(self, model_path: str, dest: str) -> str | None:
        try:
            from peft import AutoPeftModelForCausalLM
            from transformers import AutoTokenizer
            model = AutoPeftModelForCausalLM.from_pretrained(model_path)
            merged = model.merge_and_unload()
            Path(dest).mkdir(parents=True, exist_ok=True)
            merged.save_pretrained(dest)
            try:
                AutoTokenizer.from_pretrained(model_path).save_pretrained(dest)
            except Exception:
                pass
            return dest
        except Exception:
            return None

    def _copy_model(self, src: str, dest: str) -> None:
        import shutil
        s = Path(src)
        d = Path(dest)
        d.mkdir(parents=True, exist_ok=True)
        if s.is_dir():
            for item in s.iterdir():
                target = d / item.name
                if item.is_file():
                    shutil.copy2(item, target)
                elif item.is_dir():
                    if target.exists():
                        shutil.rmtree(target)
                    shutil.copytree(item, target)

    def _quantize(self, model_path: str, method: str) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(model_path)
            if method in {"int8", "8bit"}:
                model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
            dest = Path(model_path) / f"quantized_{method}"
            dest.mkdir(parents=True, exist_ok=True)
            if hasattr(model, "save_pretrained"):
                model.save_pretrained(str(dest))
            else:
                torch.save(model.state_dict(), dest / "pytorch_model.bin")
        except Exception:
            pass

    def _export_gguf(self, model_path: str, dest: str) -> None:
        import shutil
        import subprocess
        import sys
        converters = [
            "llama_cpp/convert_hf_to_gguf.py",
            "convert_hf_to_gguf.py",
        ]
        for c in converters:
            if shutil.which(c) or Path(c).exists():
                try:
                    subprocess.run([sys.executable, c, model_path, "--outfile", dest], check=False, timeout=600)
                    return
                except Exception:
                    continue


def run_pipeline(
    target: str = "tiny",
    data: str | list | None = None,
    output_dir: str = "./nexara_pipeline",
    base_model: str | None = None,
    preference_data: str | list | None = None,
    hardware: dict[str, Any] | None = None,
    **kwargs,
) -> dict[str, Any]:
    cfg = PipelineConfig(
        target=target,
        data=data,
        output_dir=output_dir,
        base_model=base_model,
        preference_data=preference_data,
        **{k: v for k, v in kwargs.items() if k in PipelineConfig.__dataclass_fields__},
    )
    return TrainingPipeline(cfg, hardware=hardware).run()
