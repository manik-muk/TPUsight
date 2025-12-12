"""Data collection structures for TPUsight profiling."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import time


class OperationType(Enum):
    """Types of JAX/XLA operations."""
    MATMUL = "matmul"
    CONV = "convolution"
    REDUCE = "reduce"
    ELEMENTWISE = "elementwise"
    TRANSPOSE = "transpose"
    RESHAPE = "reshape"
    GATHER = "gather"
    SCATTER = "scatter"
    COLLECTIVE = "collective"
    CUSTOM_CALL = "custom_call"
    OTHER = "other"


@dataclass
class OperationRecord:
    """Record of a single operation execution."""
    
    # Basic info
    name: str
    op_type: OperationType
    timestamp: float
    duration_ns: float
    
    # Shape info
    input_shapes: List[Tuple[int, ...]]
    output_shapes: List[Tuple[int, ...]]
    input_dtypes: List[str]
    output_dtypes: List[str]
    
    # Compute metrics
    flops: Optional[int] = None
    bytes_accessed: Optional[int] = None
    
    # MXU metrics (for matmul/conv)
    mxu_utilization: Optional[float] = None
    
    # Padding info
    padding_waste_pct: Optional[float] = None
    padded_shapes: Optional[List[Tuple[int, ...]]] = None
    
    # Fusion info
    is_fused: bool = False
    fusion_group: Optional[str] = None
    fusion_failure_reason: Optional[str] = None
    
    # Memory info
    hbm_read_bytes: Optional[int] = None
    hbm_write_bytes: Optional[int] = None
    
    # Source location
    source_file: Optional[str] = None
    source_line: Optional[int] = None
    source_function: Optional[str] = None
    
    # HLO info
    hlo_opcode: Optional[str] = None
    hlo_name: Optional[str] = None
    
    def __post_init__(self):
        """Calculate derived metrics."""
        if self.flops and self.duration_ns > 0:
            self.flops_per_second = self.flops / (self.duration_ns / 1e9)
        else:
            self.flops_per_second = None


@dataclass
class CompilationRecord:
    """Record of a JIT compilation event."""
    
    function_name: str
    timestamp: float
    compilation_time_ms: float
    
    # Shape info that triggered compilation
    input_shapes: List[Tuple[int, ...]]
    input_dtypes: List[str]
    
    # Cache info
    cache_hit: bool
    cache_key: Optional[str] = None
    
    # Reason for recompilation
    recompilation_reason: Optional[str] = None
    
    # Compiled executable info
    executable_size_bytes: Optional[int] = None
    num_hlo_instructions: Optional[int] = None


@dataclass
class MemoryEvent:
    """Record of a memory allocation/deallocation event."""
    
    event_type: str  # "alloc" or "dealloc"
    timestamp: float
    size_bytes: int
    device: str
    
    # Tensor info
    shape: Optional[Tuple[int, ...]] = None
    dtype: Optional[str] = None
    
    # Memory stats at this point
    total_allocated: Optional[int] = None
    peak_allocated: Optional[int] = None


@dataclass 
class FusionGroup:
    """Information about a group of fused operations."""
    
    group_id: str
    operations: List[str]
    num_ops_fused: int
    
    # Fusion success metrics
    fusion_type: str  # "loop", "input", "output", etc.
    fusion_benefit: Optional[float] = None  # Estimated speedup
    
    # Failed fusion attempts
    failed_fusions: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class ProfileData:
    """Complete profiling data collected during a session."""
    
    # Session info
    session_id: str
    start_time: float
    end_time: Optional[float] = None
    
    # Device info
    device_type: str = "tpu"
    device_count: int = 1
    tpu_version: Optional[str] = None
    
    # Collected data
    operations: List[OperationRecord] = field(default_factory=list)
    compilations: List[CompilationRecord] = field(default_factory=list)
    memory_events: List[MemoryEvent] = field(default_factory=list)
    fusion_groups: List[FusionGroup] = field(default_factory=list)
    
    # Aggregate metrics
    total_ops: int = 0
    total_flops: int = 0
    total_bytes: int = 0
    total_time_ns: int = 0
    
    # Cache statistics
    cache_hits: int = 0
    cache_misses: int = 0
    
    # Memory statistics
    peak_memory_bytes: int = 0
    total_allocations: int = 0
    
    def add_operation(self, op: OperationRecord):
        """Add an operation record."""
        self.operations.append(op)
        self.total_ops += 1
        if op.flops:
            self.total_flops += op.flops
        if op.bytes_accessed:
            self.total_bytes += op.bytes_accessed
        self.total_time_ns += int(op.duration_ns)
    
    def add_compilation(self, comp: CompilationRecord):
        """Add a compilation record."""
        self.compilations.append(comp)
        if comp.cache_hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
    
    def add_memory_event(self, event: MemoryEvent):
        """Add a memory event."""
        self.memory_events.append(event)
        if event.event_type == "alloc":
            self.total_allocations += 1
            if event.total_allocated and event.total_allocated > self.peak_memory_bytes:
                self.peak_memory_bytes = event.total_allocated
    
    def finalize(self):
        """Finalize the profile data."""
        self.end_time = time.time()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the profiling data."""
        duration = (self.end_time - self.start_time) if self.end_time else 0
        
        return {
            "session_id": self.session_id,
            "duration_seconds": duration,
            "device": {
                "type": self.device_type,
                "count": self.device_count,
                "version": self.tpu_version,
            },
            "operations": {
                "total": self.total_ops,
                "total_flops": self.total_flops,
                "total_bytes": self.total_bytes,
                "total_time_ns": self.total_time_ns,
            },
            "compilation": {
                "total": len(self.compilations),
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "cache_hit_rate": (
                    self.cache_hits / (self.cache_hits + self.cache_misses) 
                    if (self.cache_hits + self.cache_misses) > 0 else 0
                ),
            },
            "memory": {
                "peak_bytes": self.peak_memory_bytes,
                "total_allocations": self.total_allocations,
            },
        }
    
    def get_operations_by_type(self) -> Dict[OperationType, List[OperationRecord]]:
        """Group operations by type."""
        by_type: Dict[OperationType, List[OperationRecord]] = {}
        for op in self.operations:
            if op.op_type not in by_type:
                by_type[op.op_type] = []
            by_type[op.op_type].append(op)
        return by_type
    
    def get_slowest_operations(self, n: int = 10) -> List[OperationRecord]:
        """Get the N slowest operations."""
        return sorted(self.operations, key=lambda x: x.duration_ns, reverse=True)[:n]
    
    def get_operations_with_low_mxu(self, threshold: float = 50.0) -> List[OperationRecord]:
        """Get operations with MXU utilization below threshold."""
        return [
            op for op in self.operations 
            if op.mxu_utilization is not None and op.mxu_utilization < threshold
        ]
    
    def get_operations_with_padding_waste(self, threshold: float = 10.0) -> List[OperationRecord]:
        """Get operations with significant padding waste."""
        return [
            op for op in self.operations
            if op.padding_waste_pct is not None and op.padding_waste_pct > threshold
        ]

