"""Helper utilities for TPUsight."""

from typing import Tuple, List, Optional
import math


def format_bytes(num_bytes: int) -> str:
    """Format bytes into human-readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} PB"


def format_flops(flops: float) -> str:
    """Format FLOPS into human-readable string."""
    for unit in ["FLOPS", "KFLOPS", "MFLOPS", "GFLOPS", "TFLOPS", "PFLOPS"]:
        if abs(flops) < 1000.0:
            return f"{flops:.2f} {unit}"
        flops /= 1000.0
    return f"{flops:.2f} EFLOPS"


def format_duration(seconds: float) -> str:
    """Format duration into human-readable string."""
    if seconds < 1e-6:
        return f"{seconds * 1e9:.2f} ns"
    elif seconds < 1e-3:
        return f"{seconds * 1e6:.2f} µs"
    elif seconds < 1:
        return f"{seconds * 1e3:.2f} ms"
    else:
        return f"{seconds:.2f} s"


# TPU v4 MXU dimensions (128x128 systolic array)
TPU_MXU_SIZE = 128


def calculate_padding_waste(shape: Tuple[int, ...], tile_size: int = TPU_MXU_SIZE) -> dict:
    """
    Calculate the amount of compute wasted due to padding.
    
    TPU MXU operations are most efficient when dimensions are multiples
    of the tile size (typically 128 for TPU v4).
    
    Args:
        shape: The tensor shape (at least 2D for matmul)
        tile_size: The TPU tile size (default 128)
    
    Returns:
        Dictionary with padding analysis
    """
    if len(shape) < 2:
        return {
            "original_shape": shape,
            "padded_shape": shape,
            "padding_overhead": 0.0,
            "wasted_compute_pct": 0.0,
            "recommendation": None,
        }
    
    # For matmul, we care about the last two dimensions
    m, n = shape[-2], shape[-1]
    
    # Calculate padded dimensions
    padded_m = math.ceil(m / tile_size) * tile_size
    padded_n = math.ceil(n / tile_size) * tile_size
    
    original_elements = m * n
    padded_elements = padded_m * padded_n
    
    wasted_compute = (padded_elements - original_elements) / padded_elements * 100
    
    # Calculate batch dimensions overhead
    batch_dims = shape[:-2]
    batch_size = 1
    for d in batch_dims:
        batch_size *= d
    
    padded_shape = batch_dims + (padded_m, padded_n)
    
    recommendation = None
    if wasted_compute > 10:
        optimal_m = round(m / tile_size) * tile_size
        optimal_n = round(n / tile_size) * tile_size
        recommendation = (
            f"Consider reshaping to ({optimal_m}, {optimal_n}) "
            f"to reduce padding waste from {wasted_compute:.1f}% to ~0%"
        )
    
    return {
        "original_shape": shape,
        "padded_shape": padded_shape,
        "padding_m": padded_m - m,
        "padding_n": padded_n - n,
        "padding_overhead": padded_elements - original_elements,
        "wasted_compute_pct": wasted_compute,
        "recommendation": recommendation,
    }


def get_optimal_tile_size(dim: int, max_tile: int = TPU_MXU_SIZE) -> int:
    """
    Get the optimal tile size for a given dimension.
    
    Args:
        dim: The dimension size
        max_tile: Maximum tile size (TPU MXU size)
    
    Returns:
        Optimal tile size
    """
    if dim >= max_tile:
        return max_tile
    
    # Find largest power of 2 that divides evenly
    for tile in [128, 64, 32, 16, 8]:
        if dim >= tile and dim % tile == 0:
            return tile
    
    # Fall back to largest divisor
    for tile in range(max_tile, 0, -1):
        if dim % tile == 0:
            return tile
    
    return 1


def estimate_mxu_utilization(
    m: int, 
    n: int, 
    k: int, 
    tile_size: int = TPU_MXU_SIZE
) -> dict:
    """
    Estimate MXU utilization for a matrix multiply operation.
    
    For a matmul of shape (M, K) x (K, N) = (M, N):
    - M: batch/rows of output
    - K: contraction dimension
    - N: columns of output
    
    TPU MXU is a 128x128 systolic array optimized for bf16/fp32 ops.
    
    Args:
        m: First dimension (rows)
        n: Second dimension (columns) 
        k: Contraction dimension
        tile_size: MXU tile size
    
    Returns:
        Dictionary with utilization metrics
    """
    # Theoretical peak utilization assumes perfect tiling
    # Calculate how well dimensions align with tile size
    
    m_tiles = math.ceil(m / tile_size)
    n_tiles = math.ceil(n / tile_size)
    k_tiles = math.ceil(k / tile_size)
    
    # Effective elements (padded)
    m_eff = m_tiles * tile_size
    n_eff = n_tiles * tile_size
    k_eff = k_tiles * tile_size
    
    # Actual vs padded compute
    actual_flops = 2 * m * n * k
    padded_flops = 2 * m_eff * n_eff * k_eff
    
    # Utilization is ratio of useful work to total work
    compute_utilization = actual_flops / padded_flops if padded_flops > 0 else 0
    
    # Dimension efficiency (how well each dim uses the tile)
    m_efficiency = m / m_eff if m_eff > 0 else 0
    n_efficiency = n / n_eff if n_eff > 0 else 0
    k_efficiency = k / k_eff if k_eff > 0 else 0
    
    # Overall MXU utilization estimate
    mxu_utilization = compute_utilization * 100
    
    # Bottleneck identification
    bottleneck = None
    min_eff = min(m_efficiency, n_efficiency, k_efficiency)
    if min_eff < 0.5:
        if m_efficiency == min_eff:
            bottleneck = f"M dimension ({m}) underutilizes MXU - consider batching"
        elif n_efficiency == min_eff:
            bottleneck = f"N dimension ({n}) underutilizes MXU - consider wider output"
        else:
            bottleneck = f"K dimension ({k}) underutilizes MXU - consider larger reduction"
    
    return {
        "mxu_utilization_pct": mxu_utilization,
        "compute_utilization": compute_utilization,
        "actual_flops": actual_flops,
        "padded_flops": padded_flops,
        "wasted_flops": padded_flops - actual_flops,
        "m_efficiency": m_efficiency,
        "n_efficiency": n_efficiency,
        "k_efficiency": k_efficiency,
        "tiles": {"m": m_tiles, "n": n_tiles, "k": k_tiles},
        "bottleneck": bottleneck,
    }


def analyze_tensor_layout(shape: Tuple[int, ...], strides: Optional[Tuple[int, ...]] = None) -> dict:
    """
    Analyze tensor layout for TPU memory efficiency.
    
    TPUs prefer contiguous memory access patterns. This analyzes
    whether the tensor layout is optimal for TPU operations.
    
    Args:
        shape: Tensor shape
        strides: Memory strides (if known)
    
    Returns:
        Layout analysis dictionary
    """
    ndim = len(shape)
    
    if ndim < 2:
        return {
            "is_contiguous": True,
            "layout": "contiguous",
            "efficiency": 1.0,
            "recommendation": None,
        }
    
    # Check if strides indicate contiguous layout
    if strides is not None:
        # Calculate expected contiguous strides (row-major)
        expected_strides = []
        stride = 1
        for s in reversed(shape):
            expected_strides.append(stride)
            stride *= s
        expected_strides = tuple(reversed(expected_strides))
        
        is_contiguous = strides == expected_strides
    else:
        is_contiguous = True  # Assume contiguous if no stride info
    
    # TPU prefers specific layouts for different operations
    # For matmul: (batch, M, K) x (batch, K, N) = (batch, M, N)
    
    # Check if innermost dimension is large enough for coalesced access
    inner_dim = shape[-1]
    min_efficient_inner = 8  # Minimum for efficient memory access
    
    efficiency = min(1.0, inner_dim / 128)  # Optimal at 128
    
    recommendation = None
    if inner_dim < min_efficient_inner:
        recommendation = (
            f"Inner dimension ({inner_dim}) is small - "
            "consider transposing or padding for better memory coalescing"
        )
    elif not is_contiguous:
        recommendation = "Tensor is not contiguous - consider using jnp.ascontiguousarray()"
    
    return {
        "is_contiguous": is_contiguous,
        "shape": shape,
        "inner_dim": inner_dim,
        "efficiency": efficiency,
        "recommendation": recommendation,
    }


def get_hbm_bandwidth_estimate(
    bytes_transferred: int,
    duration_seconds: float,
    peak_bandwidth_gbps: float = 1200.0  # TPU v4 HBM bandwidth
) -> dict:
    """
    Estimate HBM bandwidth utilization.
    
    Args:
        bytes_transferred: Total bytes moved
        duration_seconds: Time taken
        peak_bandwidth_gbps: Peak HBM bandwidth in GB/s
    
    Returns:
        Bandwidth analysis dictionary
    """
    if duration_seconds <= 0:
        return {
            "achieved_bandwidth_gbps": 0,
            "utilization_pct": 0,
            "bytes_transferred": bytes_transferred,
            "is_memory_bound": False,
        }
    
    achieved_bandwidth = bytes_transferred / duration_seconds / 1e9  # GB/s
    utilization = (achieved_bandwidth / peak_bandwidth_gbps) * 100
    
    # If we're achieving > 50% of peak, we're likely memory bound
    is_memory_bound = utilization > 50
    
    return {
        "achieved_bandwidth_gbps": achieved_bandwidth,
        "peak_bandwidth_gbps": peak_bandwidth_gbps,
        "utilization_pct": utilization,
        "bytes_transferred": bytes_transferred,
        "duration_seconds": duration_seconds,
        "is_memory_bound": is_memory_bound,
    }

