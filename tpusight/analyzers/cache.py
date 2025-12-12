"""Dynamic shape and executable cache profiler for TPUsight."""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

from tpusight.core.data_collector import ProfileData, CompilationRecord


@dataclass
class CacheMetrics:
    """Metrics for JIT compilation and caching."""
    
    total_calls: int
    cache_hits: int
    cache_misses: int
    cache_hit_rate: float  # 0-100%
    
    # Compilation time
    total_compilation_time_ms: float
    average_compilation_time_ms: float
    max_compilation_time_ms: float
    
    # Dynamic shapes
    unique_shapes: int
    shape_variations: Dict[str, int]  # function -> num unique shapes
    
    # Recompilation hotspots
    hotspots: List[Dict[str, Any]]


class CacheAnalyzer:
    """
    Analyzes JIT compilation caching and dynamic shape impact.
    
    JAX/XLA compiles functions lazily and caches executables based on:
    - Input shapes
    - Input dtypes
    - Function identity
    
    Dynamic shapes cause cache misses and recompilation, hurting performance.
    This analyzer identifies:
    - Cache hit/miss patterns
    - Functions with many shape variations
    - Recompilation hotspots
    """
    
    def __init__(self, profile_data: ProfileData):
        self.profile_data = profile_data
        self._cached_analysis: Optional[Dict[str, Any]] = None
    
    def analyze(self) -> Dict[str, Any]:
        """
        Perform complete cache analysis.
        
        Returns:
            Dictionary with cache metrics and recommendations
        """
        if self._cached_analysis is not None:
            return self._cached_analysis
        
        compilations = self.profile_data.compilations
        
        if not compilations:
            return {
                "status": "no_data",
                "message": "No compilation events recorded",
                "metrics": None,
                "recommendations": [],
            }
        
        # Calculate basic metrics
        cache_hits = sum(1 for c in compilations if c.cache_hit)
        cache_misses = len(compilations) - cache_hits
        
        # Compilation time (only for misses)
        miss_times = [c.compilation_time_ms for c in compilations if not c.cache_hit]
        total_comp_time = sum(miss_times)
        avg_comp_time = total_comp_time / len(miss_times) if miss_times else 0
        max_comp_time = max(miss_times) if miss_times else 0
        
        # Analyze shape variations
        shape_by_function = self._analyze_shape_variations(compilations)
        unique_shapes = sum(len(shapes) for shapes in shape_by_function.values())
        
        # Find hotspots (functions with many recompilations)
        hotspots = self._find_hotspots(compilations)
        
        metrics = CacheMetrics(
            total_calls=len(compilations),
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            cache_hit_rate=(cache_hits / len(compilations) * 100) if compilations else 0,
            total_compilation_time_ms=total_comp_time,
            average_compilation_time_ms=avg_comp_time,
            max_compilation_time_ms=max_comp_time,
            unique_shapes=unique_shapes,
            shape_variations={fn: len(shapes) for fn, shapes in shape_by_function.items()},
            hotspots=hotspots[:10],
        )
        
        recommendations = self._generate_recommendations(metrics, shape_by_function)
        
        self._cached_analysis = {
            "status": "ok",
            "metrics": metrics,
            "recommendations": recommendations,
            "compilation_timeline": self._get_compilation_timeline(compilations),
            "summary": {
                "cache_hit_rate": f"{metrics.cache_hit_rate:.1f}%",
                "total_compilation_time": f"{total_comp_time:.1f} ms",
                "shape_variations": unique_shapes,
                "hotspots": len(hotspots),
            },
        }
        
        return self._cached_analysis
    
    def _analyze_shape_variations(
        self, 
        compilations: List[CompilationRecord]
    ) -> Dict[str, List[Tuple[Tuple[int, ...], ...]]]:
        """Group compilations by function and track shape variations."""
        shape_by_function: Dict[str, List[Tuple[Tuple[int, ...], ...]]] = defaultdict(list)
        
        for comp in compilations:
            shapes = tuple(tuple(s) for s in comp.input_shapes)
            if shapes not in shape_by_function[comp.function_name]:
                shape_by_function[comp.function_name].append(shapes)
        
        return dict(shape_by_function)
    
    def _find_hotspots(self, compilations: List[CompilationRecord]) -> List[Dict[str, Any]]:
        """Find functions with excessive recompilation."""
        recomp_counts: Dict[str, Dict[str, Any]] = {}
        
        for comp in compilations:
            fn = comp.function_name
            if fn not in recomp_counts:
                recomp_counts[fn] = {
                    "function": fn,
                    "total_calls": 0,
                    "cache_misses": 0,
                    "total_time_ms": 0,
                    "shapes_seen": set(),
                }
            
            recomp_counts[fn]["total_calls"] += 1
            if not comp.cache_hit:
                recomp_counts[fn]["cache_misses"] += 1
                recomp_counts[fn]["total_time_ms"] += comp.compilation_time_ms
            recomp_counts[fn]["shapes_seen"].add(str(comp.input_shapes))
        
        # Convert to list and calculate severity
        hotspots = []
        for fn, data in recomp_counts.items():
            miss_rate = data["cache_misses"] / data["total_calls"] if data["total_calls"] > 0 else 0
            
            # High priority if many misses AND significant time
            if data["cache_misses"] > 2 or miss_rate > 0.3:
                hotspots.append({
                    "function": fn,
                    "total_calls": data["total_calls"],
                    "cache_misses": data["cache_misses"],
                    "miss_rate": miss_rate,
                    "compilation_time_ms": data["total_time_ms"],
                    "unique_shapes": len(data["shapes_seen"]),
                    "severity": "critical" if miss_rate > 0.5 else "warning",
                })
        
        # Sort by compilation time (highest impact first)
        return sorted(hotspots, key=lambda x: x["compilation_time_ms"], reverse=True)
    
    def _get_compilation_timeline(
        self, 
        compilations: List[CompilationRecord]
    ) -> List[Dict[str, Any]]:
        """Get timeline data for visualization."""
        timeline = []
        
        for comp in compilations:
            timeline.append({
                "timestamp": comp.timestamp,
                "function": comp.function_name,
                "cache_hit": comp.cache_hit,
                "compilation_time_ms": comp.compilation_time_ms if not comp.cache_hit else 0,
                "shapes": [list(s) for s in comp.input_shapes],
            })
        
        return timeline
    
    def _generate_recommendations(
        self, 
        metrics: CacheMetrics,
        shape_variations: Dict[str, List]
    ) -> List[Dict[str, Any]]:
        """Generate recommendations based on cache analysis."""
        recommendations = []
        
        # Low cache hit rate
        if metrics.cache_hit_rate < 50:
            recommendations.append({
                "severity": "critical",
                "category": "cache_hit_rate",
                "message": f"Cache hit rate is very low ({metrics.cache_hit_rate:.1f}%)",
                "impact": "high",
                "suggestion": "Excessive recompilation detected. Consider using static shapes or jax.ensure_compile_time_eval().",
                "code_example": """
# Use static shapes where possible
@jax.jit
def my_function(x):
    # x should have consistent shape across calls
    return x * 2

# Or use jax.ensure_compile_time_eval for dynamic computations
from jax import ensure_compile_time_eval
shape = ensure_compile_time_eval(lambda: get_shape())
""",
            })
        elif metrics.cache_hit_rate < 80:
            recommendations.append({
                "severity": "warning",
                "category": "cache_hit_rate",
                "message": f"Cache hit rate is below optimal ({metrics.cache_hit_rate:.1f}%)",
                "impact": "medium",
                "suggestion": "Some functions are being recompiled. Review input shapes for consistency.",
            })
        
        # High compilation time
        if metrics.total_compilation_time_ms > 5000:  # >5 seconds
            recommendations.append({
                "severity": "critical",
                "category": "compilation_time",
                "message": f"Total compilation time is {metrics.total_compilation_time_ms / 1000:.1f}s",
                "impact": "high",
                "suggestion": "Significant time spent compiling. Use persistent caching or AOT compilation.",
                "code_example": """
# Enable persistent compilation cache
import jax
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")

# Or use ahead-of-time compilation
from jax.experimental import aot
compiled = aot.save(jit_fn, args)
""",
            })
        
        # Functions with many shape variations
        for fn, shapes in shape_variations.items():
            if len(shapes) > 5:
                recommendations.append({
                    "severity": "warning",
                    "category": "dynamic_shapes",
                    "message": f"Function '{fn}' has {len(shapes)} shape variations",
                    "impact": "medium",
                    "suggestion": f"Consider padding inputs to fixed shapes or batching.",
                    "shapes_seen": [str(s) for s in shapes[:5]],
                })
        
        # Hotspot recommendations
        for hotspot in metrics.hotspots[:3]:
            if hotspot["severity"] == "critical":
                recommendations.append({
                    "severity": "critical",
                    "category": "recompilation_hotspot",
                    "message": f"Function '{hotspot['function']}' recompiles frequently ({hotspot['cache_misses']} times)",
                    "impact": "high",
                    "suggestion": f"This function spent {hotspot['compilation_time_ms']:.1f}ms compiling. Stabilize input shapes.",
                })
        
        return recommendations
    
    def get_function_cache_stats(self, function_name: str) -> Dict[str, Any]:
        """Get detailed cache statistics for a specific function."""
        compilations = [
            c for c in self.profile_data.compilations 
            if c.function_name == function_name
        ]
        
        if not compilations:
            return {
                "status": "not_found",
                "message": f"No compilation records for '{function_name}'",
            }
        
        hits = sum(1 for c in compilations if c.cache_hit)
        misses = len(compilations) - hits
        
        # Get all unique shapes
        shapes = list(set(str(c.input_shapes) for c in compilations))
        
        # Timeline of calls
        calls = [
            {
                "timestamp": c.timestamp,
                "cache_hit": c.cache_hit,
                "shapes": c.input_shapes,
                "compilation_time_ms": c.compilation_time_ms if not c.cache_hit else 0,
            }
            for c in compilations
        ]
        
        return {
            "status": "ok",
            "function": function_name,
            "total_calls": len(compilations),
            "cache_hits": hits,
            "cache_misses": misses,
            "hit_rate": (hits / len(compilations) * 100) if compilations else 0,
            "unique_shapes": len(shapes),
            "shapes": shapes,
            "calls": calls,
            "recommendation": self._get_function_recommendation(function_name, hits, misses, shapes),
        }
    
    def _get_function_recommendation(
        self, 
        fn: str, 
        hits: int, 
        misses: int, 
        shapes: List[str]
    ) -> Optional[str]:
        """Get recommendation for a specific function."""
        total = hits + misses
        if total == 0:
            return None
        
        miss_rate = misses / total
        
        if miss_rate > 0.5 and len(shapes) > 3:
            return (
                f"Function '{fn}' has high recompilation due to varying shapes. "
                "Consider:\n"
                "1. Padding inputs to a fixed maximum size\n"
                "2. Using jax.jit with static_argnums for shape parameters\n"
                "3. Batching operations to amortize compilation cost"
            )
        elif miss_rate > 0.3:
            return (
                f"Function '{fn}' has moderate recompilation. "
                "Review input shapes for consistency."
            )
        
        return None
    
    def get_visualization_data(self) -> Dict[str, Any]:
        """Get data for cache visualization."""
        analysis = self.analyze()
        if analysis["status"] == "no_data":
            return {"status": "no_data"}
        
        metrics = analysis["metrics"]
        
        return {
            "status": "ok",
            "cache_pie": {
                "labels": ["Cache Hits", "Cache Misses"],
                "values": [metrics.cache_hits, metrics.cache_misses],
                "colors": ["#22c55e", "#ef4444"],
            },
            "timeline": analysis["compilation_timeline"],
            "hotspots": metrics.hotspots,
            "hit_rate": metrics.cache_hit_rate,
            "compilation_time": metrics.total_compilation_time_ms,
        }
    
    def __repr__(self) -> str:
        analysis = self.analyze()
        if analysis["status"] == "no_data":
            return "CacheAnalyzer(no data)"
        metrics = analysis["metrics"]
        return f"CacheAnalyzer(hit_rate={metrics.cache_hit_rate:.1f}%, hotspots={len(metrics.hotspots)})"

