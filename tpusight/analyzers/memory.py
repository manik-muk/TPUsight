"""Memory traffic and layout diagnostics for TPUsight."""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from tpusight.core.data_collector import ProfileData, OperationRecord, MemoryEvent, OperationType
from tpusight.utils.helpers import (
    format_bytes, 
    analyze_tensor_layout, 
    get_hbm_bandwidth_estimate,
)


@dataclass
class MemoryMetrics:
    """Metrics for memory analysis."""
    
    # Overall stats
    peak_memory_bytes: int
    total_allocations: int
    total_bytes_allocated: int
    
    # Bandwidth
    total_bytes_transferred: int
    estimated_hbm_utilization: float  # 0-100%
    
    # Layout efficiency
    average_layout_efficiency: float  # 0-1
    operations_with_poor_layout: int
    
    # Memory patterns
    memory_bound_operations: List[Dict[str, Any]]
    layout_issues: List[Dict[str, Any]]


class MemoryAnalyzer:
    """
    Analyzes memory traffic and tensor layouts for TPU operations.
    
    TPU performance is often limited by HBM (High Bandwidth Memory) 
    bandwidth rather than compute. This analyzer identifies:
    
    - Memory-bound operations
    - Inefficient tensor layouts
    - Excessive memory traffic
    - Layout transformation overhead
    """
    
    def __init__(self, profile_data: ProfileData):
        self.profile_data = profile_data
        self._cached_analysis: Optional[Dict[str, Any]] = None
        
        # TPU v4 specs (configurable for other versions)
        self.hbm_bandwidth_gbps = 1200  # GB/s
        self.hbm_capacity_gb = 32  # GB per chip
    
    def analyze(self) -> Dict[str, Any]:
        """
        Perform complete memory analysis.
        
        Returns:
            Dictionary with memory metrics and recommendations
        """
        if self._cached_analysis is not None:
            return self._cached_analysis
        
        ops = self.profile_data.operations
        
        if not ops:
            return {
                "status": "no_data",
                "message": "No operations found to analyze",
                "metrics": None,
                "recommendations": [],
            }
        
        # Calculate memory metrics
        total_bytes = sum(op.bytes_accessed or 0 for op in ops)
        total_time_s = sum(op.duration_ns for op in ops) / 1e9
        
        # Estimate bandwidth utilization
        bandwidth_stats = get_hbm_bandwidth_estimate(
            total_bytes, 
            total_time_s,
            self.hbm_bandwidth_gbps
        )
        
        # Analyze layouts
        layout_results = self._analyze_layouts(ops)
        
        # Find memory-bound operations
        memory_bound_ops = self._find_memory_bound_operations(ops)
        
        metrics = MemoryMetrics(
            peak_memory_bytes=self.profile_data.peak_memory_bytes,
            total_allocations=self.profile_data.total_allocations,
            total_bytes_allocated=total_bytes,
            total_bytes_transferred=total_bytes,
            estimated_hbm_utilization=bandwidth_stats["utilization_pct"],
            average_layout_efficiency=layout_results["average_efficiency"],
            operations_with_poor_layout=layout_results["poor_layout_count"],
            memory_bound_operations=memory_bound_ops[:10],
            layout_issues=layout_results["issues"][:10],
        )
        
        recommendations = self._generate_recommendations(metrics, bandwidth_stats)
        
        self._cached_analysis = {
            "status": "ok",
            "metrics": metrics,
            "bandwidth": bandwidth_stats,
            "recommendations": recommendations,
            "summary": {
                "peak_memory": format_bytes(metrics.peak_memory_bytes),
                "total_transferred": format_bytes(total_bytes),
                "hbm_utilization": f"{bandwidth_stats['utilization_pct']:.1f}%",
                "layout_efficiency": f"{layout_results['average_efficiency'] * 100:.1f}%",
            },
        }
        
        return self._cached_analysis
    
    def _analyze_layouts(self, ops: List[OperationRecord]) -> Dict[str, Any]:
        """Analyze tensor layouts across operations."""
        total_efficiency = 0.0
        poor_layout_count = 0
        issues = []
        
        for op in ops:
            for shape in op.output_shapes:
                layout = analyze_tensor_layout(shape)
                total_efficiency += layout["efficiency"]
                
                if layout["efficiency"] < 0.5:
                    poor_layout_count += 1
                    if layout["recommendation"]:
                        issues.append({
                            "operation": op.name,
                            "shape": shape,
                            "efficiency": layout["efficiency"],
                            "issue": layout["recommendation"],
                            "source": f"{op.source_file}:{op.source_line}" if op.source_file else None,
                        })
        
        num_shapes = sum(len(op.output_shapes) for op in ops)
        
        return {
            "average_efficiency": total_efficiency / num_shapes if num_shapes > 0 else 1.0,
            "poor_layout_count": poor_layout_count,
            "issues": sorted(issues, key=lambda x: x["efficiency"]),
        }
    
    def _find_memory_bound_operations(self, ops: List[OperationRecord]) -> List[Dict[str, Any]]:
        """Identify operations that are likely memory-bound."""
        memory_bound = []
        
        for op in ops:
            if op.bytes_accessed and op.duration_ns > 0:
                # Calculate arithmetic intensity (FLOPS / bytes)
                flops = op.flops or 0
                intensity = flops / op.bytes_accessed if op.bytes_accessed > 0 else 0
                
                # Low intensity (<10 FLOPS/byte) suggests memory-bound
                # TPU MXU has ~150 TFLOPS, HBM is ~1.2 TB/s
                # Balance point is ~125 FLOPS/byte
                is_memory_bound = intensity < 50
                
                if is_memory_bound:
                    # Calculate achieved bandwidth
                    duration_s = op.duration_ns / 1e9
                    bandwidth_gbps = op.bytes_accessed / duration_s / 1e9 if duration_s > 0 else 0
                    
                    memory_bound.append({
                        "operation": op.name,
                        "type": op.op_type.value,
                        "bytes_accessed": op.bytes_accessed,
                        "flops": flops,
                        "arithmetic_intensity": intensity,
                        "achieved_bandwidth_gbps": bandwidth_gbps,
                        "duration_ns": op.duration_ns,
                        "shapes": op.input_shapes + op.output_shapes,
                        "is_memory_bound": True,
                    })
        
        # Sort by bandwidth (highest first - these are hitting limits)
        return sorted(memory_bound, key=lambda x: x["achieved_bandwidth_gbps"], reverse=True)
    
    def _generate_recommendations(
        self, 
        metrics: MemoryMetrics,
        bandwidth_stats: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate recommendations based on memory analysis."""
        recommendations = []
        
        # High memory usage
        peak_gb = metrics.peak_memory_bytes / 1e9
        if peak_gb > self.hbm_capacity_gb * 0.9:
            recommendations.append({
                "severity": "critical",
                "category": "memory_usage",
                "message": f"Peak memory ({peak_gb:.1f}GB) approaching HBM capacity ({self.hbm_capacity_gb}GB)",
                "impact": "high",
                "suggestion": "Risk of OOM. Consider gradient checkpointing or model parallelism.",
                "code_example": """
# Use gradient checkpointing to reduce memory
from jax.checkpoint import checkpoint
@checkpoint
def memory_heavy_layer(x):
    return expensive_computation(x)
""",
            })
        elif peak_gb > self.hbm_capacity_gb * 0.7:
            recommendations.append({
                "severity": "warning",
                "category": "memory_usage",
                "message": f"Peak memory ({peak_gb:.1f}GB) is high",
                "impact": "medium",
                "suggestion": "Monitor memory usage; consider optimization if approaching limits.",
            })
        
        # Memory-bound operations
        if len(metrics.memory_bound_operations) > len(self.profile_data.operations) * 0.3:
            recommendations.append({
                "severity": "warning",
                "category": "memory_bound",
                "message": f"{len(metrics.memory_bound_operations)} operations are memory-bound",
                "impact": "high",
                "suggestion": "Many operations are limited by memory bandwidth, not compute. Consider:\n"
                             "1. Fusing operations to reduce memory traffic\n"
                             "2. Using mixed precision (bfloat16) to halve memory bandwidth\n"
                             "3. Optimizing data layouts for sequential access",
            })
        
        # Low HBM utilization (compute-bound is good, but very low means issues)
        if bandwidth_stats["utilization_pct"] < 10 and metrics.total_bytes_transferred > 1e9:
            recommendations.append({
                "severity": "info",
                "category": "bandwidth_utilization",
                "message": f"HBM bandwidth utilization is low ({bandwidth_stats['utilization_pct']:.1f}%)",
                "impact": "low",
                "suggestion": "Operations may be compute-bound (good) or have memory access inefficiencies.",
            })
        
        # Layout issues
        if metrics.operations_with_poor_layout > 5:
            recommendations.append({
                "severity": "warning",
                "category": "tensor_layout",
                "message": f"{metrics.operations_with_poor_layout} operations have suboptimal layouts",
                "impact": "medium",
                "suggestion": "Poor tensor layouts cause uncoalesced memory access. Consider:\n"
                             "1. Using jnp.ascontiguousarray() after transposes\n"
                             "2. Ensuring innermost dimensions are large (>= 128)\n"
                             "3. Avoiding frequent layout changes between operations",
            })
        
        return recommendations
    
    def get_memory_timeline(self) -> List[Dict[str, Any]]:
        """Get memory usage over time for visualization."""
        timeline = []
        current_memory = 0
        
        for event in self.profile_data.memory_events:
            if event.event_type == "alloc":
                current_memory += event.size_bytes
            else:
                current_memory -= event.size_bytes
            
            timeline.append({
                "timestamp": event.timestamp,
                "memory_bytes": current_memory,
                "event_type": event.event_type,
                "size_bytes": event.size_bytes,
                "shape": event.shape,
                "dtype": event.dtype,
            })
        
        return timeline
    
    def get_bandwidth_by_operation(self) -> List[Dict[str, Any]]:
        """Get bandwidth metrics per operation for visualization."""
        results = []
        
        for op in self.profile_data.operations:
            if op.bytes_accessed and op.duration_ns > 0:
                duration_s = op.duration_ns / 1e9
                bandwidth = op.bytes_accessed / duration_s / 1e9  # GB/s
                
                results.append({
                    "name": op.name,
                    "type": op.op_type.value,
                    "bandwidth_gbps": bandwidth,
                    "bytes_accessed": op.bytes_accessed,
                    "duration_ns": op.duration_ns,
                    "is_memory_bound": bandwidth > self.hbm_bandwidth_gbps * 0.5,
                })
        
        return sorted(results, key=lambda x: x["bandwidth_gbps"], reverse=True)
    
    def analyze_operation(self, op_name: str) -> Dict[str, Any]:
        """Get detailed memory analysis for a specific operation."""
        op = next((o for o in self.profile_data.operations if o.name == op_name), None)
        
        if op is None:
            return {"status": "not_found", "message": f"Operation '{op_name}' not found"}
        
        # Analyze layouts
        input_layouts = [analyze_tensor_layout(s) for s in op.input_shapes]
        output_layouts = [analyze_tensor_layout(s) for s in op.output_shapes]
        
        # Calculate bandwidth
        duration_s = op.duration_ns / 1e9 if op.duration_ns > 0 else 1
        bandwidth_stats = get_hbm_bandwidth_estimate(
            op.bytes_accessed or 0,
            duration_s,
            self.hbm_bandwidth_gbps
        )
        
        # Calculate roofline metrics
        flops = op.flops or 0
        bytes_accessed = op.bytes_accessed or 1
        arithmetic_intensity = flops / bytes_accessed
        
        # Determine if memory or compute bound
        # TPU v4: ~275 TFLOPS bf16, ~1.2 TB/s HBM
        # Balance point: ~229 FLOPS/byte
        balance_point = 229  # FLOPS/byte
        bottleneck = "memory" if arithmetic_intensity < balance_point else "compute"
        
        return {
            "status": "ok",
            "operation": op_name,
            "type": op.op_type.value,
            "memory": {
                "bytes_accessed": op.bytes_accessed,
                "hbm_read": op.hbm_read_bytes,
                "hbm_write": op.hbm_write_bytes,
                "bandwidth": bandwidth_stats,
            },
            "compute": {
                "flops": flops,
                "arithmetic_intensity": arithmetic_intensity,
                "balance_point": balance_point,
            },
            "bottleneck": bottleneck,
            "layouts": {
                "inputs": input_layouts,
                "outputs": output_layouts,
            },
            "recommendation": self._get_op_recommendation(op, bottleneck, input_layouts, output_layouts),
        }
    
    def _get_op_recommendation(
        self,
        op: OperationRecord,
        bottleneck: str,
        input_layouts: List[Dict],
        output_layouts: List[Dict]
    ) -> Optional[str]:
        """Get specific recommendation for an operation."""
        recommendations = []
        
        if bottleneck == "memory":
            recommendations.append("Operation is memory-bound. Consider fusing with adjacent ops.")
        
        # Check for layout issues
        for layout in input_layouts + output_layouts:
            if not layout.get("is_contiguous", True):
                recommendations.append("Non-contiguous tensor detected. Use jnp.ascontiguousarray().")
            if layout.get("efficiency", 1) < 0.5:
                recommendations.append(f"Poor layout efficiency. {layout.get('recommendation', '')}")
        
        return " ".join(recommendations) if recommendations else None
    
    def get_visualization_data(self) -> Dict[str, Any]:
        """Get data for memory visualization."""
        analysis = self.analyze()
        if analysis["status"] == "no_data":
            return {"status": "no_data"}
        
        metrics = analysis["metrics"]
        
        return {
            "status": "ok",
            "memory_timeline": self.get_memory_timeline(),
            "bandwidth_by_op": self.get_bandwidth_by_operation()[:20],
            "peak_memory": metrics.peak_memory_bytes,
            "hbm_utilization": metrics.estimated_hbm_utilization,
            "memory_bound_count": len(metrics.memory_bound_operations),
            "layout_efficiency": metrics.average_layout_efficiency,
        }
    
    def __repr__(self) -> str:
        analysis = self.analyze()
        if analysis["status"] == "no_data":
            return "MemoryAnalyzer(no data)"
        metrics = analysis["metrics"]
        return (
            f"MemoryAnalyzer(peak={format_bytes(metrics.peak_memory_bytes)}, "
            f"hbm_util={metrics.estimated_hbm_utilization:.1f}%)"
        )

