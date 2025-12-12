"""Fusion failure analysis for TPUsight."""

from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
from enum import Enum

from tpusight.core.data_collector import ProfileData, OperationRecord, OperationType


class FusionBlocker(Enum):
    """Reasons why operation fusion might fail."""
    
    SHAPE_MISMATCH = "shape_mismatch"
    DTYPE_MISMATCH = "dtype_mismatch"
    MEMORY_CONSTRAINT = "memory_constraint"
    CONTROL_FLOW = "control_flow"
    SIDE_EFFECTS = "side_effects"
    UNSUPPORTED_OP = "unsupported_op"
    DATA_DEPENDENCY = "data_dependency"
    COLLECTIVE_OP = "collective_op"
    CUSTOM_CALL = "custom_call"
    UNKNOWN = "unknown"


@dataclass
class FusionOpportunity:
    """An opportunity for operation fusion."""
    
    operations: List[str]
    blocker: FusionBlocker
    explanation: str
    estimated_speedup: float  # Multiplier, e.g., 1.2 = 20% faster
    fix_suggestion: Optional[str]


@dataclass
class FusionMetrics:
    """Metrics about fusion in the program."""
    
    total_operations: int
    fused_operations: int
    fusion_rate: float  # 0-100%
    
    # Fusion groups
    num_fusion_groups: int
    largest_fusion_group: int
    average_fusion_size: float
    
    # Blockers
    blocker_counts: Dict[FusionBlocker, int]
    
    # Opportunities
    missed_opportunities: List[FusionOpportunity]


class FusionAnalyzer:
    """
    Analyzes operation fusion success and failure in TPU programs.
    
    XLA (the TPU compiler) attempts to fuse operations together to:
    - Reduce memory traffic
    - Enable kernel fusion
    - Minimize launch overhead
    
    This analyzer identifies when fusion fails and explains why.
    """
    
    def __init__(self, profile_data: ProfileData):
        self.profile_data = profile_data
        self._cached_analysis: Optional[Dict[str, Any]] = None
        
        # Operations that typically block fusion
        self._fusion_blockers = {
            OperationType.COLLECTIVE: FusionBlocker.COLLECTIVE_OP,
            OperationType.CUSTOM_CALL: FusionBlocker.CUSTOM_CALL,
        }
        
        # Operations that are good fusion candidates
        self._fusable_ops = {
            OperationType.ELEMENTWISE,
            OperationType.REDUCE,
            OperationType.TRANSPOSE,
            OperationType.RESHAPE,
        }
    
    def analyze(self) -> Dict[str, Any]:
        """
        Perform complete fusion analysis.
        
        Returns:
            Dictionary with fusion metrics and recommendations
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
        
        # Analyze fusion patterns
        fusion_groups = self._identify_fusion_groups(ops)
        missed_opportunities = self._find_missed_opportunities(ops, fusion_groups)
        blocker_counts = self._count_blockers(ops)
        
        fused_count = sum(len(g) for g in fusion_groups)
        
        metrics = FusionMetrics(
            total_operations=len(ops),
            fused_operations=fused_count,
            fusion_rate=(fused_count / len(ops) * 100) if ops else 0,
            num_fusion_groups=len(fusion_groups),
            largest_fusion_group=max(len(g) for g in fusion_groups) if fusion_groups else 0,
            average_fusion_size=sum(len(g) for g in fusion_groups) / len(fusion_groups) if fusion_groups else 0,
            blocker_counts=blocker_counts,
            missed_opportunities=missed_opportunities[:10],  # Top 10
        )
        
        recommendations = self._generate_recommendations(metrics)
        
        self._cached_analysis = {
            "status": "ok",
            "metrics": metrics,
            "fusion_groups": [
                {"id": i, "operations": list(g), "size": len(g)}
                for i, g in enumerate(fusion_groups)
            ],
            "recommendations": recommendations,
            "summary": {
                "fusion_rate": f"{metrics.fusion_rate:.1f}%",
                "num_groups": metrics.num_fusion_groups,
                "missed_opportunities": len(missed_opportunities),
            },
        }
        
        return self._cached_analysis
    
    def _identify_fusion_groups(self, ops: List[OperationRecord]) -> List[Set[str]]:
        """Identify groups of operations that were fused together."""
        fusion_groups = []
        current_group: Set[str] = set()
        
        for i, op in enumerate(ops):
            if op.is_fused and op.fusion_group:
                # Use explicit fusion group info
                existing = next(
                    (g for g in fusion_groups if any(op.fusion_group in n for n in g)),
                    None
                )
                if existing:
                    existing.add(op.name)
                else:
                    fusion_groups.append({op.name})
            elif op.op_type in self._fusable_ops:
                # Heuristic: consecutive fusable ops might be fused
                if current_group:
                    current_group.add(op.name)
                else:
                    current_group = {op.name}
            else:
                # Non-fusable op breaks the chain
                if len(current_group) > 1:
                    fusion_groups.append(current_group)
                current_group = set()
        
        if len(current_group) > 1:
            fusion_groups.append(current_group)
        
        return fusion_groups
    
    def _find_missed_opportunities(
        self, 
        ops: List[OperationRecord],
        existing_groups: List[Set[str]]
    ) -> List[FusionOpportunity]:
        """Find operations that could have been fused but weren't."""
        opportunities = []
        
        # Group operations by their location in code
        location_groups: Dict[str, List[OperationRecord]] = {}
        for op in ops:
            if op.source_file and op.source_line:
                key = f"{op.source_file}:{op.source_line // 10 * 10}"  # Group by ~10 lines
                if key not in location_groups:
                    location_groups[key] = []
                location_groups[key].append(op)
        
        # Look for patterns that suggest missed fusion
        for location, loc_ops in location_groups.items():
            # Pattern 1: Multiple elementwise ops that aren't fused
            elementwise_ops = [op for op in loc_ops if op.op_type == OperationType.ELEMENTWISE]
            if len(elementwise_ops) >= 3:
                # Check if they share similar shapes
                shapes = [op.output_shapes[0] if op.output_shapes else () for op in elementwise_ops]
                if len(set(shapes)) == 1 and shapes[0]:  # All same shape
                    opportunities.append(FusionOpportunity(
                        operations=[op.name for op in elementwise_ops],
                        blocker=FusionBlocker.UNKNOWN,
                        explanation="Multiple elementwise operations with same shape could potentially be fused",
                        estimated_speedup=1.1 + 0.05 * len(elementwise_ops),
                        fix_suggestion="Consider using jax.lax.fori_loop or combining operations into a single expression",
                    ))
            
            # Pattern 2: Matmul followed by elementwise not fused
            matmul_ops = [op for op in loc_ops if op.op_type == OperationType.MATMUL]
            for matmul in matmul_ops:
                # Find elementwise ops with matching output shape
                matching_elem = [
                    op for op in elementwise_ops
                    if op.input_shapes and matmul.output_shapes 
                    and op.input_shapes[0] == matmul.output_shapes[0]
                ]
                if matching_elem and not matmul.is_fused:
                    opportunities.append(FusionOpportunity(
                        operations=[matmul.name] + [op.name for op in matching_elem],
                        blocker=FusionBlocker.DATA_DEPENDENCY,
                        explanation="Matmul followed by elementwise ops could be fused but dependency prevents it",
                        estimated_speedup=1.15,
                        fix_suggestion="Ensure no intermediate results are used elsewhere; use jax.lax.fused_dot for explicit fusion",
                    ))
        
        # Pattern 3: Consecutive transposes (layout churn)
        prev_transpose = None
        for op in ops:
            if op.op_type == OperationType.TRANSPOSE:
                if prev_transpose is not None:
                    # Two consecutive transposes - might cancel out
                    opportunities.append(FusionOpportunity(
                        operations=[prev_transpose.name, op.name],
                        blocker=FusionBlocker.SHAPE_MISMATCH,
                        explanation="Consecutive transpose operations may indicate unnecessary layout changes",
                        estimated_speedup=1.3,
                        fix_suggestion="Review layout choices; consecutive transposes might cancel out or be combined",
                    ))
                prev_transpose = op
            else:
                prev_transpose = None
        
        # Sort by estimated speedup
        return sorted(opportunities, key=lambda x: x.estimated_speedup, reverse=True)
    
    def _count_blockers(self, ops: List[OperationRecord]) -> Dict[FusionBlocker, int]:
        """Count fusion blockers by type."""
        counts: Dict[FusionBlocker, int] = {b: 0 for b in FusionBlocker}
        
        for op in ops:
            if op.fusion_failure_reason:
                # Parse the reason if available
                reason_lower = op.fusion_failure_reason.lower()
                if "shape" in reason_lower:
                    counts[FusionBlocker.SHAPE_MISMATCH] += 1
                elif "dtype" in reason_lower or "type" in reason_lower:
                    counts[FusionBlocker.DTYPE_MISMATCH] += 1
                elif "memory" in reason_lower:
                    counts[FusionBlocker.MEMORY_CONSTRAINT] += 1
                elif "control" in reason_lower:
                    counts[FusionBlocker.CONTROL_FLOW] += 1
                else:
                    counts[FusionBlocker.UNKNOWN] += 1
            elif op.op_type in self._fusion_blockers:
                counts[self._fusion_blockers[op.op_type]] += 1
        
        return counts
    
    def _generate_recommendations(self, metrics: FusionMetrics) -> List[Dict[str, Any]]:
        """Generate recommendations based on fusion analysis."""
        recommendations = []
        
        # Low fusion rate
        if metrics.fusion_rate < 30:
            recommendations.append({
                "severity": "critical",
                "category": "fusion_rate",
                "message": f"Fusion rate is very low ({metrics.fusion_rate:.1f}%)",
                "impact": "high",
                "suggestion": "Many operations are not being fused, causing excessive memory traffic. Review operation patterns.",
            })
        elif metrics.fusion_rate < 50:
            recommendations.append({
                "severity": "warning",
                "category": "fusion_rate",
                "message": f"Fusion rate is below optimal ({metrics.fusion_rate:.1f}%)",
                "impact": "medium",
                "suggestion": "Consider restructuring operations to enable more fusion opportunities.",
            })
        
        # Small fusion groups
        if metrics.average_fusion_size < 3 and metrics.num_fusion_groups > 10:
            recommendations.append({
                "severity": "warning",
                "category": "small_groups",
                "message": f"Average fusion group size is small ({metrics.average_fusion_size:.1f} ops)",
                "impact": "medium",
                "suggestion": "Many small fusion groups indicate fragmented computation. Try to batch operations.",
            })
        
        # Specific blocker recommendations
        for blocker, count in metrics.blocker_counts.items():
            if count > 5:
                if blocker == FusionBlocker.SHAPE_MISMATCH:
                    recommendations.append({
                        "severity": "info",
                        "category": "shape_blockers",
                        "message": f"{count} fusion failures due to shape mismatches",
                        "impact": "low",
                        "suggestion": "Ensure tensors have compatible shapes for fusion; avoid unnecessary reshapes between operations.",
                    })
                elif blocker == FusionBlocker.COLLECTIVE_OP:
                    recommendations.append({
                        "severity": "info",
                        "category": "collective_blockers",
                        "message": f"{count} collective operations blocking fusion",
                        "impact": "low",
                        "suggestion": "Collective ops (all-reduce, etc.) naturally break fusion. This is expected in distributed training.",
                    })
        
        # Missed opportunities
        if metrics.missed_opportunities:
            top_opportunity = metrics.missed_opportunities[0]
            recommendations.append({
                "severity": "info",
                "category": "fusion_opportunity",
                "message": f"Potential fusion opportunity: {len(top_opportunity.operations)} operations",
                "impact": "medium",
                "suggestion": top_opportunity.fix_suggestion or "Review highlighted operations for fusion potential",
                "estimated_speedup": f"{(top_opportunity.estimated_speedup - 1) * 100:.0f}%",
            })
        
        return recommendations
    
    def explain_fusion_failure(self, op_name: str) -> Dict[str, Any]:
        """
        Get detailed explanation for why a specific operation wasn't fused.
        
        Args:
            op_name: Name of the operation
            
        Returns:
            Detailed explanation dictionary
        """
        op = next((o for o in self.profile_data.operations if o.name == op_name), None)
        
        if op is None:
            return {"status": "not_found", "message": f"Operation '{op_name}' not found"}
        
        if op.is_fused:
            return {
                "status": "fused",
                "message": f"Operation '{op_name}' was successfully fused",
                "fusion_group": op.fusion_group,
            }
        
        # Analyze why fusion might have failed
        reasons = []
        
        # Check operation type
        if op.op_type in self._fusion_blockers:
            reasons.append({
                "reason": self._fusion_blockers[op.op_type].value,
                "explanation": f"Operation type '{op.op_type.value}' typically blocks fusion",
                "is_fixable": False,
            })
        
        # Check for explicit failure reason
        if op.fusion_failure_reason:
            reasons.append({
                "reason": op.fusion_failure_reason,
                "explanation": "Compiler-reported fusion failure",
                "is_fixable": True,
            })
        
        # Check shape compatibility with neighbors
        idx = self.profile_data.operations.index(op)
        if idx > 0:
            prev_op = self.profile_data.operations[idx - 1]
            if prev_op.output_shapes != op.input_shapes:
                reasons.append({
                    "reason": "shape_mismatch",
                    "explanation": f"Output shape {prev_op.output_shapes} doesn't match input shape {op.input_shapes}",
                    "is_fixable": True,
                    "fix": "Ensure consistent shapes between adjacent operations",
                })
        
        if not reasons:
            reasons.append({
                "reason": "unknown",
                "explanation": "No clear fusion blocker identified",
                "is_fixable": True,
            })
        
        return {
            "status": "not_fused",
            "operation": op_name,
            "reasons": reasons,
            "suggestion": "Review operation placement and shapes for fusion opportunities",
        }
    
    def get_fusion_graph(self) -> Dict[str, Any]:
        """Get data for fusion graph visualization."""
        analysis = self.analyze()
        if analysis["status"] == "no_data":
            return {"status": "no_data"}
        
        nodes = []
        edges = []
        
        for i, op in enumerate(self.profile_data.operations):
            nodes.append({
                "id": i,
                "name": op.name,
                "type": op.op_type.value,
                "fused": op.is_fused,
                "group": op.fusion_group,
            })
            
            # Add edge to next operation
            if i < len(self.profile_data.operations) - 1:
                edges.append({
                    "from": i,
                    "to": i + 1,
                    "fused": op.fusion_group == self.profile_data.operations[i + 1].fusion_group
                            if op.fusion_group else False,
                })
        
        return {
            "status": "ok",
            "nodes": nodes,
            "edges": edges,
            "groups": analysis["fusion_groups"],
        }
    
    def __repr__(self) -> str:
        analysis = self.analyze()
        if analysis["status"] == "no_data":
            return "FusionAnalyzer(no data)"
        metrics = analysis["metrics"]
        return f"FusionAnalyzer(fusion_rate={metrics.fusion_rate:.1f}%, groups={metrics.num_fusion_groups})"

