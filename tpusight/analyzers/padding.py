"""Padding and tiling inefficiency analyzer for TPUsight."""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from tpusight.core.data_collector import ProfileData, OperationRecord, OperationType
from tpusight.utils.helpers import calculate_padding_waste, TPU_MXU_SIZE


@dataclass
class PaddingMetrics:
    """Metrics for padding inefficiency."""
    
    total_operations: int
    operations_with_padding: int
    total_wasted_compute_pct: float
    total_wasted_elements: int
    
    # By severity
    critical_ops: int  # >30% waste
    warning_ops: int   # 10-30% waste
    optimal_ops: int   # <10% waste
    
    # Worst offenders
    worst_operations: List[Dict[str, Any]]


class PaddingAnalyzer:
    """
    Analyzes padding and tiling inefficiency in TPU operations.
    
    TPUs perform best when tensor dimensions are multiples of the
    MXU tile size (128). This analyzer identifies operations where
    padding wastes compute and suggests optimal shapes.
    """
    
    def __init__(self, profile_data: ProfileData):
        self.profile_data = profile_data
        self._cached_analysis: Optional[Dict[str, Any]] = None
    
    def analyze(self) -> Dict[str, Any]:
        """
        Perform complete padding inefficiency analysis.
        
        Returns:
            Dictionary with padding metrics and recommendations
        """
        if self._cached_analysis is not None:
            return self._cached_analysis
        
        # Focus on matmul and conv operations where padding matters most
        relevant_ops = [
            op for op in self.profile_data.operations
            if op.op_type in [OperationType.MATMUL, OperationType.CONV]
        ]
        
        if not relevant_ops:
            return {
                "status": "no_data",
                "message": "No matrix or convolution operations found",
                "metrics": None,
                "recommendations": [],
            }
        
        padding_results = []
        total_waste_pct = 0.0
        total_wasted_elements = 0
        critical_count = 0
        warning_count = 0
        optimal_count = 0
        
        for op in relevant_ops:
            # Analyze output shapes for padding
            for shape in op.output_shapes:
                if len(shape) >= 2:
                    result = calculate_padding_waste(shape)
                    waste_pct = result["wasted_compute_pct"]
                    
                    total_waste_pct += waste_pct
                    total_wasted_elements += result["padding_overhead"]
                    
                    if waste_pct > 30:
                        critical_count += 1
                    elif waste_pct > 10:
                        warning_count += 1
                    else:
                        optimal_count += 1
                    
                    if waste_pct > 5:  # Only track if significant
                        padding_results.append({
                            "name": op.name,
                            "shape": shape,
                            "padded_shape": result["padded_shape"],
                            "waste_pct": waste_pct,
                            "wasted_elements": result["padding_overhead"],
                            "recommendation": result["recommendation"],
                            "source": f"{op.source_file}:{op.source_line}" if op.source_file else None,
                        })
        
        avg_waste_pct = total_waste_pct / len(relevant_ops) if relevant_ops else 0
        
        # Sort by waste percentage
        worst_ops = sorted(padding_results, key=lambda x: x["waste_pct"], reverse=True)[:10]
        
        metrics = PaddingMetrics(
            total_operations=len(relevant_ops),
            operations_with_padding=len(padding_results),
            total_wasted_compute_pct=avg_waste_pct,
            total_wasted_elements=total_wasted_elements,
            critical_ops=critical_count,
            warning_ops=warning_count,
            optimal_ops=optimal_count,
            worst_operations=worst_ops,
        )
        
        recommendations = self._generate_recommendations(metrics, worst_ops)
        
        self._cached_analysis = {
            "status": "ok",
            "metrics": metrics,
            "recommendations": recommendations,
            "summary": {
                "average_waste": f"{avg_waste_pct:.1f}%",
                "total_operations": len(relevant_ops),
                "operations_needing_attention": critical_count + warning_count,
            },
        }
        
        return self._cached_analysis
    
    def _generate_recommendations(
        self, 
        metrics: PaddingMetrics, 
        worst_ops: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate recommendations based on padding analysis."""
        recommendations = []
        
        # Overall padding efficiency
        if metrics.total_wasted_compute_pct > 20:
            recommendations.append({
                "severity": "critical",
                "category": "padding_efficiency",
                "message": f"High padding overhead ({metrics.total_wasted_compute_pct:.1f}% average waste)",
                "impact": "high",
                "suggestion": "Redesign tensor shapes to be multiples of 128 for TPU efficiency",
            })
        elif metrics.total_wasted_compute_pct > 10:
            recommendations.append({
                "severity": "warning",
                "category": "padding_efficiency",
                "message": f"Moderate padding overhead ({metrics.total_wasted_compute_pct:.1f}% average waste)",
                "impact": "medium",
                "suggestion": "Consider adjusting some tensor dimensions to reduce padding",
            })
        
        # Many critical operations
        if metrics.critical_ops > 5:
            recommendations.append({
                "severity": "critical",
                "category": "critical_shapes",
                "message": f"{metrics.critical_ops} operations have >30% compute waste from padding",
                "impact": "high",
                "suggestion": "These shapes severely underutilize TPU. Consider shape adjustments.",
            })
        
        # Specific shape recommendations
        for op in worst_ops[:3]:
            if op["recommendation"]:
                recommendations.append({
                    "severity": "warning",
                    "category": "shape_optimization",
                    "message": f"Operation '{op['name']}' has {op['waste_pct']:.1f}% waste",
                    "impact": "medium",
                    "suggestion": op["recommendation"],
                    "current_shape": op["shape"],
                    "optimal_shape": op["padded_shape"],
                })
        
        return recommendations
    
    def get_shape_efficiency_table(self) -> List[Dict[str, Any]]:
        """Get a table of all shapes and their efficiency."""
        table = []
        
        for op in self.profile_data.operations:
            if op.op_type in [OperationType.MATMUL, OperationType.CONV]:
                for i, shape in enumerate(op.output_shapes):
                    if len(shape) >= 2:
                        result = calculate_padding_waste(shape)
                        table.append({
                            "operation": op.name,
                            "shape": str(shape),
                            "m": shape[-2],
                            "n": shape[-1],
                            "waste_pct": result["wasted_compute_pct"],
                            "padding_m": result.get("padding_m", 0),
                            "padding_n": result.get("padding_n", 0),
                            "optimal": result["wasted_compute_pct"] < 10,
                        })
        
        return sorted(table, key=lambda x: x["waste_pct"], reverse=True)
    
    def suggest_optimal_shapes(self, shape: Tuple[int, ...]) -> Dict[str, Any]:
        """
        Suggest optimal shapes for a given tensor shape.
        
        Args:
            shape: The current tensor shape
            
        Returns:
            Dictionary with optimization suggestions
        """
        if len(shape) < 2:
            return {
                "original": shape,
                "suggestions": [],
                "message": "Shape has fewer than 2 dimensions, no optimization needed",
            }
        
        m, n = shape[-2], shape[-1]
        batch_dims = shape[:-2]
        
        suggestions = []
        
        # Option 1: Pad up to next multiple of 128
        pad_m = ((m + TPU_MXU_SIZE - 1) // TPU_MXU_SIZE) * TPU_MXU_SIZE
        pad_n = ((n + TPU_MXU_SIZE - 1) // TPU_MXU_SIZE) * TPU_MXU_SIZE
        
        if pad_m != m or pad_n != n:
            suggestions.append({
                "type": "pad_up",
                "shape": batch_dims + (pad_m, pad_n),
                "description": f"Pad to ({pad_m}, {pad_n}) - next multiple of 128",
                "waste_pct": 0.0,
            })
        
        # Option 2: Round down if close to multiple
        round_m = (m // TPU_MXU_SIZE) * TPU_MXU_SIZE
        round_n = (n // TPU_MXU_SIZE) * TPU_MXU_SIZE
        
        if round_m > 0 and round_n > 0 and (round_m != m or round_n != n):
            # Only suggest if we're not losing too much data
            data_retained = (round_m * round_n) / (m * n) * 100
            if data_retained > 80:  # Keep at least 80% of data
                suggestions.append({
                    "type": "round_down",
                    "shape": batch_dims + (round_m, round_n),
                    "description": f"Truncate to ({round_m}, {round_n}) - exact multiple of 128",
                    "data_retained_pct": data_retained,
                    "waste_pct": 0.0,
                })
        
        # Option 3: Smaller tile sizes if dimensions are small
        for tile in [64, 32]:
            if m < TPU_MXU_SIZE or n < TPU_MXU_SIZE:
                tile_m = ((m + tile - 1) // tile) * tile
                tile_n = ((n + tile - 1) // tile) * tile
                
                original_waste = calculate_padding_waste(shape)["wasted_compute_pct"]
                new_waste = calculate_padding_waste(batch_dims + (tile_m, tile_n))["wasted_compute_pct"]
                
                if new_waste < original_waste:
                    suggestions.append({
                        "type": f"tile_{tile}",
                        "shape": batch_dims + (tile_m, tile_n),
                        "description": f"Align to {tile}-element tiles: ({tile_m}, {tile_n})",
                        "waste_pct": new_waste,
                    })
        
        return {
            "original": shape,
            "original_waste_pct": calculate_padding_waste(shape)["wasted_compute_pct"],
            "suggestions": suggestions,
        }
    
    def get_visualization_data(self) -> Dict[str, Any]:
        """Get data for padding visualization."""
        analysis = self.analyze()
        if analysis["status"] == "no_data":
            return {"status": "no_data"}
        
        metrics = analysis["metrics"]
        
        return {
            "status": "ok",
            "pie_chart": {
                "labels": ["Optimal (<10%)", "Warning (10-30%)", "Critical (>30%)"],
                "values": [metrics.optimal_ops, metrics.warning_ops, metrics.critical_ops],
                "colors": ["#22c55e", "#f59e0b", "#ef4444"],
            },
            "worst_shapes": [
                {
                    "shape": str(op["shape"]),
                    "waste": op["waste_pct"],
                }
                for op in metrics.worst_operations[:5]
            ],
            "average_waste": metrics.total_wasted_compute_pct,
        }
    
    def __repr__(self) -> str:
        analysis = self.analyze()
        if analysis["status"] == "no_data":
            return "PaddingAnalyzer(no data)"
        metrics = analysis["metrics"]
        return f"PaddingAnalyzer(avg_waste={metrics.total_wasted_compute_pct:.1f}%, ops={metrics.total_operations})"

