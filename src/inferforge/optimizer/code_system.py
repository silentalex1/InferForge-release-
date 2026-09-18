from __future__ import annotations

import json
from typing import Any

from inferforge.core.config import trained_models_dir

VALID_QUANTIZATIONS = {"Q2_K", "Q3_K", "Q4_0", "Q4_K", "Q5_0", "Q5_K", "Q6_K", "Q8_0", "F16", "BF16", "F32"}


class GenerationProfile:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.model_path = trained_models_dir() / model_name.replace(":", "-")
        self.config_path = self.model_path / "generation_profile.json"
        self._config: dict[str, Any] = {}
        self._load_config()

    def _load_config(self) -> None:
        if self.config_path.exists():
            with self.config_path.open("r", encoding="utf-8") as f:
                self._config = json.load(f)
        else:
            self._config = self._get_default_config()

    def _get_default_config(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "sampling": {
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40,
                "repeat_penalty": 1.1,
            },
        }

    def save_config(self) -> None:
        self.model_path.mkdir(parents=True, exist_ok=True)
        with self.config_path.open("w", encoding="utf-8") as f:
            json.dump(self._config, f, indent=2)

    def get_sampling_options(self) -> dict[str, Any]:
        return dict(self._config.get("sampling", self._get_default_config()["sampling"]))

    def apply_task_preset(self, task_type: str) -> None:
        presets = {
            "coding": {"temperature": 0.2, "top_p": 0.95, "top_k": 40, "repeat_penalty": 1.2},
            "chat": {"temperature": 0.7, "top_p": 0.9, "top_k": 40, "repeat_penalty": 1.1},
            "analysis": {"temperature": 0.3, "top_p": 0.8, "top_k": 40, "repeat_penalty": 1.15},
            "creative": {"temperature": 0.9, "top_p": 0.95, "top_k": 40, "repeat_penalty": 1.0},
        }
        preset = presets.get(task_type, presets["chat"])
        self._config.setdefault("sampling", {}).update(preset)
        self.save_config()

    def get_config(self) -> dict[str, Any]:
        return self._config.copy()

    def update_config(self, updates: dict[str, Any]) -> None:
        self._config.update(updates)
        self.save_config()


_profiles: dict[str, GenerationProfile] = {}


def get_generation_profile(model_name: str) -> GenerationProfile:
    if model_name not in _profiles:
        _profiles[model_name] = GenerationProfile(model_name)
    return _profiles[model_name]
