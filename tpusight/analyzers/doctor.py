"""TPU Doctor - Actionable optimization suggestions for TPUsight."""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

from tpusight.core.data_collector import ProfileData, OperationType


class IssueSeverity(Enum):
    """Severity levels for optimization issues."""
    CRITICAL = "critical"  # Major performance impact, should fix
    WARNING = "warning"    # Moderate impact, recommended to fix
    INFO = "info"          # Minor impact, nice to have


class IssueCategory(Enum):
    """Categories of optimization issues."""
    MXU_UTILIZATION = "mxu_utilization"
    PADDING = "padding"
    FUSION = "fusion"
    COMPILATION = "compilation"
    MEMORY = "memory"
    LAYOUT = "layout"
    SHAPES = "shapes"
    DTYPE = "dtype"


@dataclass
class OptimizationIssue:
    """A single optimization issue/recommendation."""
    
    id: str
    severity: IssueSeverity
    category: IssueCategory
    title: str
    message: str
    impact_estimate: str  # e.g., "~20% speedup"
    suggestion: str
    code_example: Optional[str] = None
    affected_operations: Optional[List[str]] = None
    related_metrics: Optional[Dict[str, Any]] = None


class TPUDoctor:
    """
    TPU Doctor - Actionable optimization recommendations.
    
    Aggregates insights from all analyzers and provides a prioritized
    list of optimization opportunities with:
    - Clear explanations of the issue
    - Estimated performance impact
    - Specific code suggestions to fix
    - Examples where applicable
    """
    
    def __init__(self, profile_data: ProfileData):
        self.profile_data = profile_data
        self._cached_issues: Optional[List[OptimizationIssue]] = None
    
    def diagnose(self) -> Dict[str, Any]:
        """
        Perform comprehensive diagnosis and return all issues.
        
        Returns:
            Dictionary with diagnosis results
        """
        if self._cached_issues is not None:
            issues = self._cached_issues
        else:
            issues = self._gather_all_issues()
            self._cached_issues = issues
        
        # Group by severity
        critical = [i for i in issues if i.severity == IssueSeverity.CRITICAL]
        warnings = [i for i in issues if i.severity == IssueSeverity.WARNING]
        info = [i for i in issues if i.severity == IssueSeverity.INFO]
        
        # Calculate health score (0-100)
        health_score = self._calculate_health_score(issues)
        
        return {
            "health_score": health_score,
            "health_status": self._get_health_status(health_score),
            "total_issues": len(issues),
            "critical_count": len(critical),
            "warning_count": len(warnings),
            "info_count": len(info),
            "issues": {
                "critical": [self._issue_to_dict(i) for i in critical],
                "warning": [self._issue_to_dict(i) for i in warnings],
                "info": [self._issue_to_dict(i) for i in info],
            },
            "top_recommendations": [
                self._issue_to_dict(i) for i in issues[:5]
            ],
        }
    
    def _gather_all_issues(self) -> List[OptimizationIssue]:
        """Gather issues from all analyzers."""
        issues = []
        issue_id = 0
        
        # Import analyzers
        from tpusight.analyzers.systolic import SystolicAnalyzer
        from tpusight.analyzers.padding import PaddingAnalyzer
        from tpusight.analyzers.fusion import FusionAnalyzer
        from tpusight.analyzers.cache import CacheAnalyzer
        from tpusight.analyzers.memory import MemoryAnalyzer
        
        # Systolic/MXU analysis
        systolic = SystolicAnalyzer(self.profile_data)
        systolic_result = systolic.analyze()
        if systolic_result["status"] == "ok":
            for rec in systolic_result["recommendations"]:
                issues.append(OptimizationIssue(
                    id=f"mxu_{issue_id}",
                    severity=IssueSeverity[rec["severity"].upper()],
                    category=IssueCategory.MXU_UTILIZATION,
                    title="MXU Underutilization",
                    message=rec["message"],
                    impact_estimate=self._estimate_mxu_impact(
                        systolic_result["metrics"].overall_utilization
                    ),
                    suggestion=rec["suggestion"],
                    code_example=self._get_mxu_code_example(),
                    related_metrics={
                        "current_utilization": systolic_result["metrics"].overall_utilization,
                        "target_utilization": 80,
                    },
                ))
                issue_id += 1
        
        # Padding analysis
        padding = PaddingAnalyzer(self.profile_data)
        padding_result = padding.analyze()
        if padding_result["status"] == "ok":
            for rec in padding_result["recommendations"]:
                issues.append(OptimizationIssue(
                    id=f"pad_{issue_id}",
                    severity=IssueSeverity[rec["severity"].upper()],
                    category=IssueCategory.PADDING,
                    title="Padding Inefficiency",
                    message=rec["message"],
                    impact_estimate=f"~{rec.get('impact', 'medium')} compute savings",
                    suggestion=rec["suggestion"],
                    code_example=self._get_padding_code_example(rec),
                ))
                issue_id += 1
        
        # Fusion analysis
        fusion = FusionAnalyzer(self.profile_data)
        fusion_result = fusion.analyze()
        if fusion_result["status"] == "ok":
            for rec in fusion_result["recommendations"]:
                issues.append(OptimizationIssue(
                    id=f"fuse_{issue_id}",
                    severity=IssueSeverity[rec["severity"].upper()],
                    category=IssueCategory.FUSION,
                    title="Fusion Opportunity",
                    message=rec["message"],
                    impact_estimate=rec.get("estimated_speedup", "5-15% speedup"),
                    suggestion=rec["suggestion"],
                    code_example=self._get_fusion_code_example(),
                ))
                issue_id += 1
        
        # Cache analysis
        cache = CacheAnalyzer(self.profile_data)
        cache_result = cache.analyze()
        if cache_result["status"] == "ok":
            for rec in cache_result["recommendations"]:
                issues.append(OptimizationIssue(
                    id=f"cache_{issue_id}",
                    severity=IssueSeverity[rec["severity"].upper()],
                    category=IssueCategory.COMPILATION,
                    title="Compilation Cache Issue",
                    message=rec["message"],
                    impact_estimate=self._estimate_cache_impact(
                        cache_result["metrics"].total_compilation_time_ms
                    ),
                    suggestion=rec["suggestion"],
                    code_example=rec.get("code_example"),
                ))
                issue_id += 1
        
        # Memory analysis
        memory = MemoryAnalyzer(self.profile_data)
        memory_result = memory.analyze()
        if memory_result["status"] == "ok":
            for rec in memory_result["recommendations"]:
                issues.append(OptimizationIssue(
                    id=f"mem_{issue_id}",
                    severity=IssueSeverity[rec["severity"].upper()],
                    category=IssueCategory.MEMORY,
                    title="Memory Issue",
                    message=rec["message"],
                    impact_estimate=rec.get("impact", "varies"),
                    suggestion=rec["suggestion"],
                    code_example=rec.get("code_example"),
                ))
                issue_id += 1
        
        # Additional heuristic checks
        issues.extend(self._check_dtype_issues())
        issues.extend(self._check_shape_issues())
        
        # Sort by severity and impact
        severity_order = {IssueSeverity.CRITICAL: 0, IssueSeverity.WARNING: 1, IssueSeverity.INFO: 2}
        issues.sort(key=lambda x: (severity_order[x.severity], x.id))
        
        return issues
    
    def _check_dtype_issues(self) -> List[OptimizationIssue]:
        """Check for dtype-related optimization opportunities."""
        issues = []
        
        # Count operations by dtype
        dtype_counts: Dict[str, int] = {}
        for op in self.profile_data.operations:
            for dtype in op.input_dtypes:
                dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
        
        # Check for float32 when bfloat16 would work
        float32_count = dtype_counts.get("float32", 0)
        bfloat16_count = dtype_counts.get("bfloat16", 0)
        
        if float32_count > bfloat16_count * 2 and float32_count > 10:
            issues.append(OptimizationIssue(
                id="dtype_0",
                severity=IssueSeverity.INFO,
                category=IssueCategory.DTYPE,
                title="Consider Mixed Precision",
                message=f"Using float32 for {float32_count} operations. Consider bfloat16 for TPU.",
                impact_estimate="~2x memory bandwidth, faster matmuls",
                suggestion="TPUs are optimized for bfloat16. Consider using mixed precision training.",
                code_example="""
# Enable bfloat16 for better TPU performance
import jax.numpy as jnp

# Cast inputs to bfloat16
x = x.astype(jnp.bfloat16)

# Or use automatic mixed precision
from jax import config
config.update('jax_default_matmul_precision', 'bfloat16')
""",
            ))
        
        return issues
    
    def _check_shape_issues(self) -> List[OptimizationIssue]:
        """Check for shape-related optimization opportunities."""
        issues = []
        
        # Find very small operations
        small_ops = []
        for op in self.profile_data.operations:
            if op.op_type == OperationType.MATMUL:
                for shape in op.input_shapes:
                    if len(shape) >= 2:
                        if shape[-1] < 32 or shape[-2] < 32:
                            small_ops.append(op.name)
                            break
        
        if len(small_ops) > 5:
            issues.append(OptimizationIssue(
                id="shape_0",
                severity=IssueSeverity.WARNING,
                category=IssueCategory.SHAPES,
                title="Many Small Matrix Operations",
                message=f"Found {len(small_ops)} matmul operations with dimensions < 32",
                impact_estimate="Up to 10x speedup with batching",
                suggestion="Small matrix operations severely underutilize the MXU. Consider batching.",
                code_example="""
# Instead of many small matmuls:
# for x in xs:
#     result = jnp.dot(x, w)  # x is small

# Batch them together:
xs_batched = jnp.stack(xs)  # (batch, M, K)
results = jnp.dot(xs_batched, w)  # Single efficient matmul
""",
                affected_operations=small_ops[:5],
            ))
        
        return issues
    
    def _calculate_health_score(self, issues: List[OptimizationIssue]) -> int:
        """Calculate overall health score 0-100."""
        if not issues:
            return 100
        
        # Deduct points based on severity
        score = 100
        for issue in issues:
            if issue.severity == IssueSeverity.CRITICAL:
                score -= 15
            elif issue.severity == IssueSeverity.WARNING:
                score -= 5
            else:
                score -= 1
        
        return max(0, min(100, score))
    
    def _get_health_status(self, score: int) -> str:
        """Get health status string from score."""
        if score >= 90:
            return "excellent"
        elif score >= 70:
            return "good"
        elif score >= 50:
            return "fair"
        else:
            return "needs_attention"
    
    def _issue_to_dict(self, issue: OptimizationIssue) -> Dict[str, Any]:
        """Convert issue to dictionary."""
        return {
            "id": issue.id,
            "severity": issue.severity.value,
            "category": issue.category.value,
            "title": issue.title,
            "message": issue.message,
            "impact_estimate": issue.impact_estimate,
            "suggestion": issue.suggestion,
            "code_example": issue.code_example,
            "affected_operations": issue.affected_operations,
            "related_metrics": issue.related_metrics,
        }
    
    def _estimate_mxu_impact(self, current_util: float) -> str:
        """Estimate impact of improving MXU utilization."""
        if current_util < 30:
            return "3-5x potential speedup"
        elif current_util < 50:
            return "2-3x potential speedup"
        elif current_util < 70:
            return "1.5-2x potential speedup"
        else:
            return "10-30% potential speedup"
    
    def _estimate_cache_impact(self, compilation_time_ms: float) -> str:
        """Estimate impact of fixing cache issues."""
        if compilation_time_ms > 10000:
            return f"Save {compilation_time_ms/1000:.1f}s compilation time"
        elif compilation_time_ms > 1000:
            return f"Save ~{compilation_time_ms/1000:.1f}s per run"
        else:
            return "Minor compilation overhead"
    
    def _get_mxu_code_example(self) -> str:
        """Get code example for MXU optimization."""
        return """
# Ensure matrix dimensions are multiples of 128 for optimal MXU usage
import jax.numpy as jnp

def pad_to_multiple(x, multiple=128):
    \"\"\"Pad tensor to multiple of TPU tile size.\"\"\"
    pad_m = (multiple - x.shape[-2] % multiple) % multiple
    pad_n = (multiple - x.shape[-1] % multiple) % multiple
    if pad_m > 0 or pad_n > 0:
        x = jnp.pad(x, [(0, 0)] * (x.ndim - 2) + [(0, pad_m), (0, pad_n)])
    return x

# Example: Pad weight matrix for efficient matmul
w_padded = pad_to_multiple(w, 128)
result = jnp.dot(x, w_padded)[:, :original_n]  # Slice to original size
"""
    
    def _get_padding_code_example(self, rec: Dict[str, Any]) -> str:
        """Get code example for padding optimization."""
        current = rec.get("current_shape", "(M, N)")
        optimal = rec.get("optimal_shape", "(M', N')")
        
        return f"""
# Current shape: {current}
# Optimal shape: {optimal}

# Option 1: Pad input
x_padded = jnp.pad(x, [(0, pad_m), (0, pad_n)])

# Option 2: Adjust model architecture
# Use dimensions that are multiples of 128:
hidden_dim = 512  # Instead of 500
output_dim = 256  # Instead of 250
"""
    
    def _get_fusion_code_example(self) -> str:
        """Get code example for fusion optimization."""
        return """
# Instead of separate operations:
# y = jnp.exp(x)
# z = y * scale
# result = z + bias

# Combine into a single fused operation:
def fused_op(x, scale, bias):
    return jnp.exp(x) * scale + bias

# JIT compile to enable XLA fusion
fused_op_jit = jax.jit(fused_op)
result = fused_op_jit(x, scale, bias)
"""
    
    def get_recommendations(self, max_count: int = 10) -> List[Dict[str, Any]]:
        """
        Get top optimization recommendations.
        
        Args:
            max_count: Maximum number of recommendations to return
            
        Returns:
            List of recommendation dictionaries
        """
        diagnosis = self.diagnose()
        return diagnosis["top_recommendations"][:max_count]
    
    def get_quick_wins(self) -> List[Dict[str, Any]]:
        """Get easy-to-implement optimizations with good impact."""
        all_issues = self._cached_issues or self._gather_all_issues()
        
        # Quick wins: INFO or WARNING severity that have code examples
        quick_wins = [
            self._issue_to_dict(i) for i in all_issues
            if i.severity != IssueSeverity.CRITICAL
            and i.code_example is not None
        ]
        
        return quick_wins[:5]
    
    def get_critical_issues(self) -> List[Dict[str, Any]]:
        """Get only critical issues that need immediate attention."""
        all_issues = self._cached_issues or self._gather_all_issues()
        
        critical = [
            self._issue_to_dict(i) for i in all_issues
            if i.severity == IssueSeverity.CRITICAL
        ]
        
        return critical
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a brief summary of the TPU health check."""
        diagnosis = self.diagnose()
        
        return {
            "health_score": diagnosis["health_score"],
            "status": diagnosis["health_status"],
            "critical_issues": diagnosis["critical_count"],
            "warnings": diagnosis["warning_count"],
            "top_issue": (
                diagnosis["top_recommendations"][0]["message"]
                if diagnosis["top_recommendations"]
                else "No issues found"
            ),
        }
    
    def __repr__(self) -> str:
        diagnosis = self.diagnose()
        return (
            f"TPUDoctor(health={diagnosis['health_score']}/100, "
            f"issues={diagnosis['total_issues']})"
        )

