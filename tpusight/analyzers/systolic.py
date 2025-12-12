"""Systolic array utilization analyzer for TPUsight."""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from tpusight.core.data_collector import ProfileData, OperationRecord, OperationType
from tpusight.utils.helpers import estimate_mxu_utilization, TPU_MXU_SIZE


@dataclass
class MXUUtilizationMetrics:
    """Metrics for MXU (Matrix Multiply Unit) utilization."""
    
    overall_utilization: float  # 0-100%
    matmul_utilization: float   # 0-100% for matmul ops only
    total_matmul_ops: int
    low_util_ops: int           # Ops with < 50% utilization
    wasted_flops: int
    
    # Breakdown by efficiency bucket
    efficiency_buckets: Dict[str, int]  # e.g., {"90-100%": 5, "70-90%": 3, ...}
    
    # Worst offenders
    low_util_operations: List[Dict[str, Any]]


class SystolicAnalyzer:
    """
    Analyzes systolic array (MXU) utilization for TPU operations.
    
    The TPU's Matrix Multiply Unit (MXU) is a 128x128 systolic array
    that performs matrix multiplications. This analyzer helps identify
    when the MXU is underutilized due to:
    
    - Small matrix dimensions
    - Dimensions not aligned to 128
    - Inefficient batching
    - Memory-bound operations
    """
    
    def __init__(self, profile_data: ProfileData):
        self.profile_data = profile_data
        self._cached_analysis: Optional[Dict[str, Any]] = None
    
    def analyze(self) -> Dict[str, Any]:
        """
        Perform complete MXU utilization analysis.
        
        Returns:
            Dictionary with utilization metrics and recommendations
        """
        if self._cached_analysis is not None:
            return self._cached_analysis
        
        matmul_ops = [
            op for op in self.profile_data.operations
            if op.op_type == OperationType.MATMUL
        ]
        
        if not matmul_ops:
            return {
                "status": "no_data",
                "message": "No matrix multiplication operations found",
                "metrics": None,
                "recommendations": [],
            }
        
        # Calculate utilization metrics
        total_util = 0.0
        total_flops = 0
        total_wasted = 0
        low_util_ops = []
        
        efficiency_buckets = {
            "90-100%": 0,
            "70-90%": 0,
            "50-70%": 0,
            "30-50%": 0,
            "0-30%": 0,
        }
        
        for op in matmul_ops:
            util = op.mxu_utilization or 0
            total_util += util
            
            if op.flops:
                total_flops += op.flops
                # Calculate wasted FLOPS based on utilization
                if util < 100:
                    wasted = int(op.flops * (100 - util) / util) if util > 0 else op.flops
                    total_wasted += wasted
            
            # Bucket the efficiency
            if util >= 90:
                efficiency_buckets["90-100%"] += 1
            elif util >= 70:
                efficiency_buckets["70-90%"] += 1
            elif util >= 50:
                efficiency_buckets["50-70%"] += 1
            elif util >= 30:
                efficiency_buckets["30-50%"] += 1
            else:
                efficiency_buckets["0-30%"] += 1
            
            # Track low utilization operations
            if util < 50:
                low_util_ops.append({
                    "name": op.name,
                    "utilization": util,
                    "input_shapes": op.input_shapes,
                    "output_shapes": op.output_shapes,
                    "flops": op.flops,
                    "source": f"{op.source_file}:{op.source_line}" if op.source_file else None,
                    "recommendation": self._get_recommendation_for_op(op),
                })
        
        avg_util = total_util / len(matmul_ops) if matmul_ops else 0
        
        metrics = MXUUtilizationMetrics(
            overall_utilization=avg_util,
            matmul_utilization=avg_util,
            total_matmul_ops=len(matmul_ops),
            low_util_ops=len(low_util_ops),
            wasted_flops=total_wasted,
            efficiency_buckets=efficiency_buckets,
            low_util_operations=sorted(low_util_ops, key=lambda x: x["utilization"])[:10],
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(metrics)
        
        self._cached_analysis = {
            "status": "ok",
            "metrics": metrics,
            "recommendations": recommendations,
            "summary": {
                "average_utilization": f"{avg_util:.1f}%",
                "total_matmul_ops": len(matmul_ops),
                "low_efficiency_ops": len(low_util_ops),
                "wasted_compute": f"{total_wasted:,} FLOPS",
            },
        }
        
        return self._cached_analysis
    
    def _get_recommendation_for_op(self, op: OperationRecord) -> Optional[str]:
        """Get a specific recommendation for an underutilized operation."""
        if not op.input_shapes or len(op.input_shapes) < 2:
            return None
        
        lhs_shape = op.input_shapes[0]
        rhs_shape = op.input_shapes[1]
        
        if len(lhs_shape) < 2 or len(rhs_shape) < 2:
            return None
        
        m = lhs_shape[-2]
        k = lhs_shape[-1]
        n = rhs_shape[-1]
        
        recommendations = []
        
        # Check each dimension
        if m < TPU_MXU_SIZE:
            if m < 32:
                recommendations.append(f"M={m} is very small. Consider batching multiple inputs together")
            else:
                pad_to = ((m + TPU_MXU_SIZE - 1) // TPU_MXU_SIZE) * TPU_MXU_SIZE
                recommendations.append(f"Pad M from {m} to {pad_to}")
        
        if k < TPU_MXU_SIZE:
            recommendations.append(f"K={k} is small. Consider combining with adjacent operations")
        
        if n < TPU_MXU_SIZE:
            if n < 32:
                recommendations.append(f"N={n} is very small. Consider wider output dimensions")
            else:
                pad_to = ((n + TPU_MXU_SIZE - 1) // TPU_MXU_SIZE) * TPU_MXU_SIZE
                recommendations.append(f"Pad N from {n} to {pad_to}")
        
        # Check alignment
        if m % 8 != 0:
            recommendations.append(f"M={m} is not aligned to 8. Consider padding")
        if n % 8 != 0:
            recommendations.append(f"N={n} is not aligned to 8. Consider padding")
        
        return "; ".join(recommendations) if recommendations else None
    
    def _generate_recommendations(self, metrics: MXUUtilizationMetrics) -> List[Dict[str, Any]]:
        """Generate overall recommendations based on metrics."""
        recommendations = []
        
        # Overall utilization recommendation
        if metrics.overall_utilization < 50:
            recommendations.append({
                "severity": "critical",
                "category": "mxu_utilization",
                "message": f"MXU utilization is critically low ({metrics.overall_utilization:.1f}%)",
                "impact": "high",
                "suggestion": "Review matrix dimensions and ensure they're multiples of 128 for optimal TPU performance",
            })
        elif metrics.overall_utilization < 70:
            recommendations.append({
                "severity": "warning",
                "category": "mxu_utilization", 
                "message": f"MXU utilization is below optimal ({metrics.overall_utilization:.1f}%)",
                "impact": "medium",
                "suggestion": "Consider adjusting batch sizes or padding dimensions to 128",
            })
        
        # Many low-efficiency ops
        if metrics.low_util_ops > metrics.total_matmul_ops * 0.3:
            recommendations.append({
                "severity": "warning",
                "category": "small_matmuls",
                "message": f"{metrics.low_util_ops}/{metrics.total_matmul_ops} operations have <50% utilization",
                "impact": "medium",
                "suggestion": "Many small matmuls detected. Consider operator fusion or batching",
            })
        
        # Wasted compute
        if metrics.wasted_flops > 1e9:  # More than 1 GFLOP wasted
            recommendations.append({
                "severity": "info",
                "category": "wasted_compute",
                "message": f"Approximately {metrics.wasted_flops / 1e9:.2f} GFLOPS wasted due to padding",
                "impact": "low",
                "suggestion": "Optimize tensor shapes to reduce padding overhead",
            })
        
        return recommendations
    
    def get_utilization_timeline(self) -> List[Dict[str, Any]]:
        """Get MXU utilization over time for timeline visualization."""
        timeline = []
        
        for op in self.profile_data.operations:
            if op.op_type == OperationType.MATMUL:
                timeline.append({
                    "timestamp": op.timestamp,
                    "utilization": op.mxu_utilization or 0,
                    "name": op.name,
                    "duration_ns": op.duration_ns,
                    "flops": op.flops,
                })
        
        return timeline
    
    def get_efficiency_heatmap_data(self) -> Dict[str, Any]:
        """Get data for efficiency heatmap visualization."""
        matmul_ops = [
            op for op in self.profile_data.operations
            if op.op_type == OperationType.MATMUL and op.input_shapes
        ]
        
        if not matmul_ops:
            return {"data": [], "x_label": "N", "y_label": "M"}
        
        heatmap_data = []
        for op in matmul_ops:
            if len(op.input_shapes) >= 2:
                m = op.input_shapes[0][-2] if len(op.input_shapes[0]) >= 2 else 1
                n = op.input_shapes[1][-1] if len(op.input_shapes[1]) >= 1 else 1
                
                heatmap_data.append({
                    "m": m,
                    "n": n,
                    "utilization": op.mxu_utilization or 0,
                    "name": op.name,
                })
        
        return {
            "data": heatmap_data,
            "x_label": "N (output columns)",
            "y_label": "M (output rows)",
        }
    
    def __repr__(self) -> str:
        analysis = self.analyze()
        if analysis["status"] == "no_data":
            return "SystolicAnalyzer(no data)"
        metrics = analysis["metrics"]
        return f"SystolicAnalyzer(utilization={metrics.overall_utilization:.1f}%, ops={metrics.total_matmul_ops})"

