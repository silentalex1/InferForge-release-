"""
InferForge Model Merger - Advanced AI model merging capabilities.

This module provides sophisticated model merging functionality including:
- Tokenizer alignment and vocabulary unification
- Multiple merging strategies (SLERP, TIES, MoE)
- Hardware-optimized memory management
- Architecture-agnostic merging (Procrustes, SVD)
- Activation-guided importance masking (Fisher Information)
- Real-time merge evaluation and auto-tuning
- Support for merging any compatible models
"""

from inferforge.merger.alignment import DynamicSVD, ProcrustesAligner
from inferforge.merger.core.loader import (
    detect_format,
    detect_model_architecture,
    load_model_weights,
)
from inferforge.merger.core.tensor_utils import TensorUtils
from inferforge.merger.core.tokenizer_aligner import TokenizerAligner
from inferforge.merger.core.weight_blender import MergeConfig, MergeStrategy, WeightBlender
from inferforge.merger.execution import FisherImportanceMask, FisherResult
from inferforge.merger.pipeline import MergePipeline, merge_models
from inferforge.merger.validation import (
    EvaluationResult,
    MergeCheckpoint,
    MergeEvaluator,
    MergeQuality,
)

__all__ = [
    "TokenizerAligner",
    "WeightBlender",
    "MergeConfig",
    "MergeStrategy",
    "TensorUtils",
    "detect_format",
    "detect_model_architecture",
    "load_model_weights",
    "ProcrustesAligner",
    "DynamicSVD",
    "FisherImportanceMask",
    "FisherResult",
    "MergeEvaluator",
    "MergeQuality",
    "MergeCheckpoint",
    "EvaluationResult",
    "MergePipeline",
    "merge_models",
]
