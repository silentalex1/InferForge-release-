"""Premium tier system with advanced feature flags and resource allocation."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class Tier(Enum):
    """User tier levels."""
    COMMUNITY = "community"
    STARTER = "starter"
    PREMIUM = "premium"
    PREMIUM_PLUS = "premium_plus"
    ENTERPRISE = "enterprise"


@dataclass
class TierLimits:
    """Resource limits per tier."""
    max_concurrent_trainings: int = 1
    max_gpu_nodes: int = 1
    max_training_hours_per_month: float = 10.0
    max_model_size_gb: float = 7.0
    max_dataset_size_examples: int = 1000
    priority_queue_multiplier: float = 1.0
    cloud_training_enabled: bool = False
    distributed_training_enabled: bool = False
    advanced_monitoring_enabled: bool = False
    collaborative_training_enabled: bool = False
    api_access_enabled: bool = False
    priority_support: bool = False
    custom_models_enabled: bool = False
    advanced_nexara_features: bool = False


@dataclass
class TierFeatures:
    """Feature availability per tier."""
    # Training Features
    basic_training: bool = True
    curriculum_learning: bool = True
    nexara_compilation: bool = False
    adaptive_training: bool = False
    multi_stage_training: bool = False
    
    # Performance Features
    gpu_optimization: bool = True
    mixed_precision: bool = False
    gradient_checkpointing: bool = False
    distributed_training: bool = False
    automatic_scaling: bool = False
    
    # Data Features
    advanced_data_formats: bool = False
    data_augmentation: bool = False
    smart_sampling: bool = False
    data_quality_scoring: bool = False
    
    # Monitoring Features
    basic_monitoring: bool = True
    real_time_metrics: bool = False
    tensorboard_integration: bool = False
    advanced_analytics: bool = False
    ml_insights: bool = False
    
    # Cloud Features
    local_training_only: bool = True
    cloud_training: bool = False
    hybrid_training: bool = False
    model_sharing: bool = False
    
    # Collaboration Features
    private_training: bool = True
    team_collaboration: bool = False
    shared_datasets: bool = False
    collaborative_training: bool = False
    
    # Support Features
    community_support: bool = True
    email_support: bool = False
    priority_support: bool = False
    dedicated_support: bool = False


TIER_LIMITS = {
    Tier.COMMUNITY: TierLimits(
        max_concurrent_trainings=1,
        max_gpu_nodes=1,
        max_training_hours_per_month=10.0,
        max_model_size_gb=7.0,
        max_dataset_size_examples=1000,
        priority_queue_multiplier=1.0,
    ),
    Tier.STARTER: TierLimits(
        max_concurrent_trainings=2,
        max_gpu_nodes=1,
        max_training_hours_per_month=50.0,
        max_model_size_gb=13.0,
        max_dataset_size_examples=10000,
        priority_queue_multiplier=2.0,
        advanced_monitoring_enabled=True,
    ),
    Tier.PREMIUM: TierLimits(
        max_concurrent_trainings=4,
        max_gpu_nodes=2,
        max_training_hours_per_month=200.0,
        max_model_size_gb=70.0,
        max_dataset_size_examples=100000,
        priority_queue_multiplier=5.0,
        cloud_training_enabled=True,
        distributed_training_enabled=True,
        advanced_monitoring_enabled=True,
        collaborative_training_enabled=True,
        api_access_enabled=True,
        custom_models_enabled=True,
        advanced_nexara_features=True,
    ),
    Tier.PREMIUM_PLUS: TierLimits(
        max_concurrent_trainings=8,
        max_gpu_nodes=4,
        max_training_hours_per_month=500.0,
        max_model_size_gb=405.0,
        max_dataset_size_examples=1000000,
        priority_queue_multiplier=10.0,
        cloud_training_enabled=True,
        distributed_training_enabled=True,
        advanced_monitoring_enabled=True,
        collaborative_training_enabled=True,
        api_access_enabled=True,
        priority_support=True,
        custom_models_enabled=True,
        advanced_nexara_features=True,
    ),
    Tier.ENTERPRISE: TierLimits(
        max_concurrent_trainings=16,
        max_gpu_nodes=8,
        max_training_hours_per_month=float('inf'),
        max_model_size_gb=1800.0,
        max_dataset_size_examples=float('inf'),
        priority_queue_multiplier=20.0,
        cloud_training_enabled=True,
        distributed_training_enabled=True,
        advanced_monitoring_enabled=True,
        collaborative_training_enabled=True,
        api_access_enabled=True,
        priority_support=True,
        custom_models_enabled=True,
        advanced_nexara_features=True,
    ),
}

TIER_FEATURES = {
    Tier.COMMUNITY: TierFeatures(),
    Tier.STARTER: TierFeatures(
        nexara_compilation=True,
        adaptive_training=True,
        mixed_precision=True,
        gradient_checkpointing=True,
        advanced_data_formats=True,
        real_time_metrics=True,
        email_support=True,
    ),
    Tier.PREMIUM: TierFeatures(
        nexara_compilation=True,
        adaptive_training=True,
        multi_stage_training=True,
        mixed_precision=True,
        gradient_checkpointing=True,
        distributed_training=True,
        automatic_scaling=True,
        advanced_data_formats=True,
        data_augmentation=True,
        smart_sampling=True,
        data_quality_scoring=True,
        real_time_metrics=True,
        tensorboard_integration=True,
        advanced_analytics=True,
        ml_insights=True,
        cloud_training=True,
        hybrid_training=True,
        model_sharing=True,
        team_collaboration=True,
        shared_datasets=True,
        collaborative_training=True,
        email_support=True,
        priority_support=True,
    ),
    Tier.PREMIUM_PLUS: TierFeatures(
        nexara_compilation=True,
        adaptive_training=True,
        multi_stage_training=True,
        mixed_precision=True,
        gradient_checkpointing=True,
        distributed_training=True,
        automatic_scaling=True,
        advanced_data_formats=True,
        data_augmentation=True,
        smart_sampling=True,
        data_quality_scoring=True,
        real_time_metrics=True,
        tensorboard_integration=True,
        advanced_analytics=True,
        ml_insights=True,
        cloud_training=True,
        hybrid_training=True,
        model_sharing=True,
        team_collaboration=True,
        shared_datasets=True,
        collaborative_training=True,
        email_support=True,
        priority_support=True,
        dedicated_support=True,
    ),
    Tier.ENTERPRISE: TierFeatures(
        nexara_compilation=True,
        adaptive_training=True,
        multi_stage_training=True,
        mixed_precision=True,
        gradient_checkpointing=True,
        distributed_training=True,
        automatic_scaling=True,
        advanced_data_formats=True,
        data_augmentation=True,
        smart_sampling=True,
        data_quality_scoring=True,
        real_time_metrics=True,
        tensorboard_integration=True,
        advanced_analytics=True,
        ml_insights=True,
        cloud_training=True,
        hybrid_training=True,
        model_sharing=True,
        team_collaboration=True,
        shared_datasets=True,
        collaborative_training=True,
        email_support=True,
        priority_support=True,
        dedicated_support=True,
    ),
}


class PremiumManager:
    """Manage premium features and tier access."""
    
    def __init__(self):
        self._current_tier = Tier.COMMUNITY
        self._usage_data = self._load_usage()
    
    def _license_path(self) -> Path:
        from inferforge.core.config import config_dir
        return config_dir() / "license.json"
    
    def _usage_path(self) -> Path:
        from inferforge.core.config import data_dir
        return data_dir() / "usage.json"
    
    def _load_usage(self) -> dict[str, Any]:
        path = self._usage_path()
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {
            "training_hours_this_month": 0.0,
            "month_start": time.strftime("%Y-%m"),
            "concurrent_trainings": 0,
            "last_reset": time.time(),
        }
    
    def _save_usage(self) -> None:
        path = self._usage_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._usage_data, indent=2), encoding="utf-8")
    
    def _detect_tier(self) -> Tier:
        """Detect current tier from license."""
        license_data = None
        license_path = self._license_path()
        
        if license_path.exists():
            try:
                license_data = json.loads(license_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        
        if not license_data:
            return Tier.COMMUNITY
        
        edition = license_data.get("edition", "").lower()
        edition_map = {
            "community": Tier.COMMUNITY,
            "starter": Tier.STARTER,
            "premium": Tier.PREMIUM,
            "premium plus": Tier.PREMIUM_PLUS,
            "premium_plus": Tier.PREMIUM_PLUS,
            "enterprise": Tier.ENTERPRISE,
        }
        
        return edition_map.get(edition, Tier.COMMUNITY)
    
    def get_current_tier(self) -> Tier:
        """Get current user tier."""
        self._current_tier = self._detect_tier()
        return self._current_tier
    
    def get_limits(self) -> TierLimits:
        """Get resource limits for current tier."""
        tier = self.get_current_tier()
        return TIER_LIMITS.get(tier, TIER_LIMITS[Tier.COMMUNITY])
    
    def get_features(self) -> TierFeatures:
        """Get feature availability for current tier."""
        tier = self.get_current_tier()
        return TIER_FEATURES.get(tier, TIER_FEATURES[Tier.COMMUNITY])
    
    def has_feature(self, feature_name: str) -> bool:
        """Check if current tier has a specific feature."""
        features = self.get_features()
        return getattr(features, feature_name, False)
    
    def check_training_limit(self, estimated_hours: float = 1.0) -> tuple[bool, str]:
        """Check if user can start training based on limits."""
        limits = self.get_limits()
        usage = self._usage_data
        
        # Reset monthly counter if new month
        current_month = time.strftime("%Y-%m")
        if usage.get("month_start") != current_month:
            usage["training_hours_this_month"] = 0.0
            usage["month_start"] = current_month
            self._save_usage()
        
        # Check concurrent training limit
        if usage.get("concurrent_trainings", 0) >= limits.max_concurrent_trainings:
            return False, f"Concurrent training limit reached ({limits.max_concurrent_trainings}). Please wait for current training to complete."
        
        # Check monthly hours limit
        remaining_hours = limits.max_training_hours_per_month - usage.get("training_hours_this_month", 0)
        if remaining_hours < estimated_hours:
            return False, f"Monthly training hours limit reached. {remaining_hours:.1f} hours remaining this month."
        
        return True, "Training limits check passed"
    
    def start_training(self, estimated_hours: float = 1.0) -> bool:
        """Record training start."""
        can_train, message = self.check_training_limit(estimated_hours)
        if not can_train:
            return False
        
        self._usage_data["concurrent_trainings"] = self._usage_data.get("concurrent_trainings", 0) + 1
        self._save_usage()
        return True
    
    def end_training(self, actual_hours: float) -> None:
        """Record training completion."""
        self._usage_data["concurrent_trainings"] = max(0, self._usage_data.get("concurrent_trainings", 0) - 1)
        self._usage_data["training_hours_this_month"] = self._usage_data.get("training_hours_this_month", 0) + actual_hours
        self._save_usage()
    
    def get_priority_score(self) -> float:
        """Get priority score for queue placement."""
        limits = self.get_limits()
        return limits.priority_queue_multiplier
    
    def get_tier_info(self) -> dict[str, Any]:
        """Get comprehensive tier information."""
        tier = self.get_current_tier()
        limits = self.get_limits()
        features = self.get_features()
        usage = self._usage_data
        
        return {
            "tier": tier.value,
            "limits": {
                "max_concurrent_trainings": limits.max_concurrent_trainings,
                "max_gpu_nodes": limits.max_gpu_nodes,
                "max_training_hours_per_month": limits.max_training_hours_per_month,
                "max_model_size_gb": limits.max_model_size_gb,
                "max_dataset_size_examples": limits.max_dataset_size_examples,
                "priority_queue_multiplier": limits.priority_queue_multiplier,
            },
            "usage": {
                "concurrent_trainings": usage.get("concurrent_trainings", 0),
                "training_hours_this_month": usage.get("training_hours_this_month", 0),
                "month": usage.get("month_start", ""),
            },
            "features": {
                k: v for k, v in features.__dict__.items()
            },
        }


# Global premium manager instance
_premium_manager = None

def get_premium_manager() -> PremiumManager:
    """Get global premium manager instance."""
    global _premium_manager
    if _premium_manager is None:
        _premium_manager = PremiumManager()
    return _premium_manager
