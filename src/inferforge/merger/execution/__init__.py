"""
Execution module for memory-efficient and distributed model merging.
"""

from inferforge.merger.execution.fisher_mask import FisherImportanceMask, FisherResult

__all__ = [
    "FisherImportanceMask",
    "FisherResult",
]
