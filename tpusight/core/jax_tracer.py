"""JAX tracing and HLO analysis for TPUsight."""

from typing import Dict, List, Optional, Tuple, Any, Callable
from functools import wraps
import time
import uuid
import traceback
import re

try:
    import jax
    import jax.numpy as jnp
    from jax import core as jax_core
    from jax._src import traceback_util
    HAS_JAX = True
except ImportError:
    HAS_JAX = False

from tpusight.core.data_collector import (
    ProfileData, 
    OperationRecord, 
    CompilationRecord,
    OperationType,
)
from tpusight.utils.helpers import (
    estimate_mxu_utilization,
    calculate_padding_waste,
)


def get_op_type(op_name: str, input_shapes: Optional[List[Tuple[int, ...]]] = None) -> OperationType:
    """Determine operation type from name and/or shapes."""
    op_name_lower = op_name.lower()
    
    # Check by name first
    if any(x in op_name_lower for x in ["dot", "matmul", "gemm", "linear", "dense"]):
        return OperationType.MATMUL
    elif any(x in op_name_lower for x in ["conv", "convolution"]):
        return OperationType.CONV
    elif any(x in op_name_lower for x in ["reduce", "sum", "mean", "max", "min"]):
        return OperationType.REDUCE
    elif any(x in op_name_lower for x in ["add", "mul", "sub", "div", "exp", "log", "relu", "gelu"]):
        return OperationType.ELEMENTWISE
    elif "transpose" in op_name_lower:
        return OperationType.TRANSPOSE
    elif any(x in op_name_lower for x in ["reshape", "broadcast"]):
        return OperationType.RESHAPE
    elif "gather" in op_name_lower:
        return OperationType.GATHER
    elif "scatter" in op_name_lower:
        return OperationType.SCATTER
    elif any(x in op_name_lower for x in ["all-reduce", "all-gather", "collective"]):
        return OperationType.COLLECTIVE
    elif "custom" in op_name_lower:
        return OperationType.CUSTOM_CALL
    
    # If we have input shapes, try to infer from shapes
    # Matmul pattern: two 2D+ tensors where lhs[-1] == rhs[-2]
    if input_shapes and len(input_shapes) >= 2:
        lhs, rhs = input_shapes[0], input_shapes[1]
        if len(lhs) >= 2 and len(rhs) >= 2:
            if lhs[-1] == rhs[-2]:
                # Looks like a matmul pattern
                return OperationType.MATMUL
    
    return OperationType.OTHER


def calculate_matmul_flops(lhs_shape: Tuple[int, ...], rhs_shape: Tuple[int, ...]) -> int:
    """Calculate FLOPS for a matmul operation."""
    if len(lhs_shape) < 2 or len(rhs_shape) < 2:
        return 0
    
    # For batched matmul: (batch, M, K) x (batch, K, N) = (batch, M, N)
    # FLOPS = 2 * batch * M * K * N (multiply-add)
    
    m = lhs_shape[-2]
    k = lhs_shape[-1]
    n = rhs_shape[-1]
    
    # Calculate batch size
    batch_size = 1
    for dim in lhs_shape[:-2]:
        batch_size *= dim
    
    return 2 * batch_size * m * k * n


def estimate_bytes_accessed(
    input_shapes: List[Tuple[int, ...]],
    output_shapes: List[Tuple[int, ...]],
    dtype: str = "float32"
) -> int:
    """Estimate bytes accessed for an operation."""
    dtype_sizes = {
        "float32": 4,
        "float16": 2,
        "bfloat16": 2,
        "int32": 4,
        "int64": 8,
        "int8": 1,
    }
    
    bytes_per_elem = dtype_sizes.get(dtype, 4)
    
    total_elements = 0
    for shape in input_shapes + output_shapes:
        elements = 1
        for dim in shape:
            elements *= dim
        total_elements += elements
    
    return total_elements * bytes_per_elem


class JAXTracer:
    """Traces JAX operations for profiling."""
    
    def __init__(self, profile_data: ProfileData):
        self.profile_data = profile_data
        self._active = False
        self._original_jit = None
        self._traced_functions: Dict[str, Any] = {}
        self._compilation_cache: Dict[str, CompilationRecord] = {}
    
    def start(self):
        """Start tracing JAX operations."""
        if not HAS_JAX:
            raise RuntimeError("JAX is not installed. Please install jax and jaxlib.")
        
        self._active = True
        self._install_hooks()
    
    def stop(self):
        """Stop tracing JAX operations."""
        self._active = False
        self._remove_hooks()
    
    def _install_hooks(self):
        """Install tracing hooks into JAX."""
        # Store original jit for restoration
        self._original_jit = jax.jit
        
        # We'll use a custom wrapper that tracks compilation
        tracer = self
        original_jit = self._original_jit
        
        def traced_jit(fun, *args, **kwargs):
            """Wrapper around jax.jit that adds tracing."""
            jitted = original_jit(fun, *args, **kwargs)
            
            @wraps(fun)
            def wrapper(*call_args, **call_kwargs):
                return tracer._trace_call(
                    jitted, 
                    fun.__name__ if hasattr(fun, '__name__') else str(fun),
                    call_args, 
                    call_kwargs
                )
            
            return wrapper
        
        # Note: We don't actually replace jax.jit to avoid breaking things
        # Instead, we provide our own trace decorator
    
    def _remove_hooks(self):
        """Remove tracing hooks from JAX."""
        pass  # Currently no-op since we don't modify jax.jit
    
    def _trace_call(
        self, 
        jitted_fn: Callable,
        fn_name: str,
        args: tuple,
        kwargs: dict,
        force_trace: bool = False
    ) -> Any:
        """Trace a single JIT-compiled function call."""
        if not self._active and not force_trace:
            return jitted_fn(*args, **kwargs)
        
        # Get input shapes and dtypes
        input_shapes = []
        input_dtypes = []
        
        for arg in args:
            if hasattr(arg, 'shape'):
                input_shapes.append(tuple(arg.shape))
                input_dtypes.append(str(arg.dtype))
        
        for v in kwargs.values():
            if hasattr(v, 'shape'):
                input_shapes.append(tuple(v.shape))
                input_dtypes.append(str(v.dtype))
        
        # Generate cache key
        cache_key = f"{fn_name}_{input_shapes}_{input_dtypes}"
        
        # Check if this is a new compilation
        is_cache_hit = cache_key in self._compilation_cache
        
        # Time the execution
        start_time = time.perf_counter_ns()
        result = jitted_fn(*args, **kwargs)
        
        # Block until computation is complete (for accurate timing)
        if HAS_JAX and hasattr(result, 'block_until_ready'):
            result.block_until_ready()
        
        end_time = time.perf_counter_ns()
        duration_ns = end_time - start_time
        
        # Get output shapes
        output_shapes = []
        output_dtypes = []
        
        if hasattr(result, 'shape'):
            output_shapes.append(tuple(result.shape))
            output_dtypes.append(str(result.dtype))
        elif isinstance(result, (tuple, list)):
            for r in result:
                if hasattr(r, 'shape'):
                    output_shapes.append(tuple(r.shape))
                    output_dtypes.append(str(r.dtype))
        
        # Create operation record
        op_record = self._create_operation_record(
            name=fn_name,
            input_shapes=input_shapes,
            output_shapes=output_shapes,
            input_dtypes=input_dtypes,
            output_dtypes=output_dtypes,
            duration_ns=duration_ns,
        )
        
        self.profile_data.add_operation(op_record)
        
        # Record compilation if this was a cache miss
        if not is_cache_hit:
            comp_record = CompilationRecord(
                function_name=fn_name,
                timestamp=time.time(),
                compilation_time_ms=duration_ns / 1e6,  # Rough estimate
                input_shapes=input_shapes,
                input_dtypes=input_dtypes,
                cache_hit=False,
                cache_key=cache_key,
            )
            self.profile_data.add_compilation(comp_record)
            self._compilation_cache[cache_key] = comp_record
        else:
            # Record cache hit
            comp_record = CompilationRecord(
                function_name=fn_name,
                timestamp=time.time(),
                compilation_time_ms=0,
                input_shapes=input_shapes,
                input_dtypes=input_dtypes,
                cache_hit=True,
                cache_key=cache_key,
            )
            self.profile_data.add_compilation(comp_record)
        
        return result
    
    def _create_operation_record(
        self,
        name: str,
        input_shapes: List[Tuple[int, ...]],
        output_shapes: List[Tuple[int, ...]],
        input_dtypes: List[str],
        output_dtypes: List[str],
        duration_ns: float,
    ) -> OperationRecord:
        """Create an operation record with computed metrics."""
        
        # Pass input_shapes to help infer operation type
        op_type = get_op_type(name, input_shapes)
        
        # Calculate FLOPS for matmul
        flops = None
        mxu_util = None
        padding_waste = None
        
        if op_type == OperationType.MATMUL and len(input_shapes) >= 2:
            flops = calculate_matmul_flops(input_shapes[0], input_shapes[1])
            
            # Estimate MXU utilization
            if len(input_shapes[0]) >= 2 and len(input_shapes[1]) >= 2:
                m = input_shapes[0][-2]
                k = input_shapes[0][-1]
                n = input_shapes[1][-1]
                
                mxu_metrics = estimate_mxu_utilization(m, n, k)
                mxu_util = mxu_metrics["mxu_utilization_pct"]
                
                # Calculate padding waste for output
                if output_shapes:
                    padding_info = calculate_padding_waste(output_shapes[0])
                    padding_waste = padding_info["wasted_compute_pct"]
        
        # Estimate bytes accessed
        bytes_accessed = estimate_bytes_accessed(
            input_shapes, 
            output_shapes,
            input_dtypes[0] if input_dtypes else "float32"
        )
        
        # Get source location
        source_file = None
        source_line = None
        source_function = None
        
        try:
            # Get the caller's frame (skip internal frames)
            stack = traceback.extract_stack()
            for frame in reversed(stack):
                if 'tpusight' not in frame.filename and 'jax' not in frame.filename:
                    source_file = frame.filename
                    source_line = frame.lineno
                    source_function = frame.name
                    break
        except Exception:
            pass
        
        return OperationRecord(
            name=name,
            op_type=op_type,
            timestamp=time.time(),
            duration_ns=duration_ns,
            input_shapes=input_shapes,
            output_shapes=output_shapes,
            input_dtypes=input_dtypes,
            output_dtypes=output_dtypes,
            flops=flops,
            bytes_accessed=bytes_accessed,
            mxu_utilization=mxu_util,
            padding_waste_pct=padding_waste,
            source_file=source_file,
            source_line=source_line,
            source_function=source_function,
        )
    
    def trace_function(self, fn: Callable) -> Callable:
        """Decorator to trace a function."""
        if not HAS_JAX:
            return fn
        
        fn_name = fn.__name__ if hasattr(fn, '__name__') else str(fn)
        
        # JIT compile the function
        jitted = jax.jit(fn)
        
        @wraps(fn)
        def wrapper(*args, **kwargs):
            # force_trace=True ensures decorator always records operations
            return self._trace_call(jitted, fn_name, args, kwargs, force_trace=True)
        
        return wrapper


def analyze_hlo(hlo_text: str) -> Dict[str, Any]:
    """
    Analyze HLO (High Level Optimizer) text for insights.
    
    This extracts information about:
    - Operations and their types
    - Fusion opportunities
    - Memory layout
    - Potential inefficiencies
    """
    analysis = {
        "operations": [],
        "fusions": [],
        "potential_issues": [],
        "recommendations": [],
    }
    
    # Count operation types
    op_counts: Dict[str, int] = {}
    
    # Simple regex patterns for HLO ops
    op_pattern = r'%(\w+)\s*=\s*(\w+)'
    
    for match in re.finditer(op_pattern, hlo_text):
        op_name = match.group(2)
        op_counts[op_name] = op_counts.get(op_name, 0) + 1
    
    analysis["operations"] = op_counts
    
    # Look for fusion annotations
    fusion_pattern = r'fusion\s*\('
    fusion_count = len(re.findall(fusion_pattern, hlo_text))
    analysis["fusions"] = {"count": fusion_count}
    
    # Look for potential issues
    
    # 1. Many small operations (could indicate fusion failures)
    total_ops = sum(op_counts.values())
    if total_ops > 100 and fusion_count < total_ops // 10:
        analysis["potential_issues"].append({
            "type": "low_fusion",
            "message": f"Only {fusion_count} fusions for {total_ops} operations",
            "severity": "warning",
        })
    
    # 2. Transpose operations (memory layout issues)
    transpose_count = op_counts.get("transpose", 0)
    if transpose_count > 5:
        analysis["potential_issues"].append({
            "type": "many_transposes",
            "message": f"{transpose_count} transpose operations may indicate layout issues",
            "severity": "info",
        })
    
    # 3. Pad operations (padding inefficiency)
    pad_count = op_counts.get("pad", 0)
    if pad_count > 0:
        analysis["potential_issues"].append({
            "type": "padding",
            "message": f"{pad_count} padding operations detected",
            "severity": "info",
        })
    
    return analysis

