"""
TPUsight - A comprehensive TPU profiler inspired by NVIDIA Nsight
"""

from tpusight.core.profiler import TPUsight
from tpusight.core.data_collector import ProfileData, OperationRecord

__version__ = "0.1.0"
__all__ = ["TPUsight", "ProfileData", "OperationRecord"]

