"""Time breakdown analyzer for TPUsight - tracks actual time spent in different categories."""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time
import os
import json
import tempfile

try:
    import jax
    import jax.numpy as jnp
    from jax import profiler as jax_profiler
    HAS_JAX = True
except ImportError:
    HAS_JAX = False

from tpusight.core.data_collector import ProfileData


class TimeCategory(Enum):
    """Categories of time spent during TPU execution."""
    COMPUTE = "compute"                    # Actual compute (matmul, conv, etc.)
    MEMORY_WAIT = "memory_wait"            # Waiting for HBM memory transfers
    REMATERIALIZATION = "rematerialization" # Recomputing values (gradient checkpointing)
    COMPILATION = "compilation"            # XLA/HLO compilation time
    HOST_TO_DEVICE = "host_to_device"      # Data transfer H2D
    DEVICE_TO_HOST = "device_to_host"      # Data transfer D2H
    COLLECTIVE = "collective"              # All-reduce, all-gather, etc.
    KERNEL_LAUNCH = "kernel_launch"        # Kernel dispatch overhead
    IDLE = "idle"                          # TPU idle time
    OTHER = "other"


@dataclass
class TimeBreakdown:
    """Breakdown of time spent in each category."""
    
    total_time_ms: float
    
    # Time per category in milliseconds
    compute_ms: float = 0.0
    memory_wait_ms: float = 0.0
    rematerialization_ms: float = 0.0
    compilation_ms: float = 0.0
    host_to_device_ms: float = 0.0
    device_to_host_ms: float = 0.0
    collective_ms: float = 0.0
    kernel_launch_ms: float = 0.0
    idle_ms: float = 0.0
    other_ms: float = 0.0
    
    # Detailed breakdown
    by_operation: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    def get_percentages(self) -> Dict[str, float]:
        """Get time as percentages of total."""
        if self.total_time_ms == 0:
            return {cat.value: 0.0 for cat in TimeCategory}
        
        return {
            "compute": self.compute_ms / self.total_time_ms * 100,
            "memory_wait": self.memory_wait_ms / self.total_time_ms * 100,
            "rematerialization": self.rematerialization_ms / self.total_time_ms * 100,
            "compilation": self.compilation_ms / self.total_time_ms * 100,
            "host_to_device": self.host_to_device_ms / self.total_time_ms * 100,
            "device_to_host": self.device_to_host_ms / self.total_time_ms * 100,
            "collective": self.collective_ms / self.total_time_ms * 100,
            "kernel_launch": self.kernel_launch_ms / self.total_time_ms * 100,
            "idle": self.idle_ms / self.total_time_ms * 100,
            "other": self.other_ms / self.total_time_ms * 100,
        }
    
    def get_top_categories(self, n: int = 5) -> List[Tuple[str, float, float]]:
        """Get top N categories by time spent. Returns (name, ms, percentage)."""
        categories = [
            ("Compute", self.compute_ms),
            ("Memory Wait", self.memory_wait_ms),
            ("Rematerialization", self.rematerialization_ms),
            ("Compilation", self.compilation_ms),
            ("Host→Device", self.host_to_device_ms),
            ("Device→Host", self.device_to_host_ms),
            ("Collective Ops", self.collective_ms),
            ("Kernel Launch", self.kernel_launch_ms),
            ("Idle", self.idle_ms),
            ("Other", self.other_ms),
        ]
        
        sorted_cats = sorted(categories, key=lambda x: x[1], reverse=True)
        
        return [
            (name, ms, ms / self.total_time_ms * 100 if self.total_time_ms > 0 else 0)
            for name, ms in sorted_cats[:n]
            if ms > 0
        ]


class TimeBreakdownAnalyzer:
    """
    Analyzes actual time breakdown for TPU operations.
    
    Uses JAX's built-in profiler to capture real timing data including:
    - Compute time (matmul, conv, elementwise)
    - Memory wait time (HBM bandwidth bottlenecks)
    - Rematerialization (recomputation from checkpointing)
    - Compilation/JIT overhead
    - Data transfer times
    - Collective operation times
    """
    
    def __init__(self, profile_data: ProfileData):
        self.profile_data = profile_data
        self._cached_breakdown: Optional[TimeBreakdown] = None
        self._trace_data: List[Dict[str, Any]] = []
        self._is_profiling = False
        self._profile_dir: Optional[str] = None
    
    def start_profiling(self):
        """Start JAX profiler to capture detailed timing."""
        if not HAS_JAX:
            return
        
        self._is_profiling = True
        self._trace_data = []
        
        # Create temp directory for trace
        self._profile_dir = tempfile.mkdtemp(prefix="tpusight_")
        
        # Start JAX profiler
        try:
            jax_profiler.start_trace(self._profile_dir)
        except Exception as e:
            print(f"Warning: Could not start JAX profiler: {e}")
            self._is_profiling = False
    
    def stop_profiling(self):
        """Stop JAX profiler and collect data."""
        if not self._is_profiling or not HAS_JAX:
            return
        
        try:
            jax_profiler.stop_trace()
            self._parse_trace_data()
        except Exception as e:
            print(f"Warning: Could not stop JAX profiler: {e}")
        finally:
            self._is_profiling = False
    
    def _parse_trace_data(self):
        """Parse the JAX trace data to extract timing information."""
        if not self._profile_dir:
            return
        
        # JAX writes trace data in various formats
        # Try to find and parse the trace file
        try:
            for filename in os.listdir(self._profile_dir):
                if filename.endswith('.json') or filename.endswith('.trace'):
                    filepath = os.path.join(self._profile_dir, filename)
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                        self._extract_timing_from_trace(data)
        except Exception:
            pass  # Trace parsing is best-effort
    
    def _extract_timing_from_trace(self, trace_data: Dict[str, Any]):
        """Extract timing information from trace JSON."""
        events = trace_data.get('traceEvents', [])
        
        for event in events:
            if event.get('ph') == 'X':  # Complete events
                name = event.get('name', '')
                dur = event.get('dur', 0) / 1000  # Convert µs to ms
                
                self._trace_data.append({
                    'name': name,
                    'duration_ms': dur,
                    'category': self._categorize_event(name),
                })
    
    def _categorize_event(self, event_name: str) -> TimeCategory:
        """Categorize a trace event by name."""
        name_lower = event_name.lower()
        
        # Memory operations
        if any(x in name_lower for x in ['infeed', 'outfeed', 'copy', 'transfer']):
            if 'host' in name_lower and 'device' in name_lower:
                return TimeCategory.HOST_TO_DEVICE
            return TimeCategory.MEMORY_WAIT
        
        # Collective operations
        if any(x in name_lower for x in ['all-reduce', 'all-gather', 'reduce-scatter', 'collective']):
            return TimeCategory.COLLECTIVE
        
        # Rematerialization
        if any(x in name_lower for x in ['remat', 'checkpoint', 'rematerialize']):
            return TimeCategory.REMATERIALIZATION
        
        # Compilation
        if any(x in name_lower for x in ['compile', 'xla', 'hlo', 'jit']):
            return TimeCategory.COMPILATION
        
        # Compute operations
        if any(x in name_lower for x in ['dot', 'conv', 'matmul', 'gemm', 'fusion', 
                                          'add', 'mul', 'reduce', 'softmax']):
            return TimeCategory.COMPUTE
        
        # Kernel launch
        if 'launch' in name_lower or 'dispatch' in name_lower:
            return TimeCategory.KERNEL_LAUNCH
        
        return TimeCategory.OTHER
    
    def analyze(self) -> Dict[str, Any]:
        """
        Analyze time breakdown from collected data.
        
        Returns detailed breakdown of where time is spent.
        """
        if self._cached_breakdown is not None:
            return self._format_analysis(self._cached_breakdown)
        
        # Calculate breakdown from trace data if available
        breakdown = self._calculate_breakdown_from_traces()
        
        # If no trace data, estimate from operation records
        if breakdown.total_time_ms == 0:
            breakdown = self._estimate_breakdown_from_operations()
        
        self._cached_breakdown = breakdown
        return self._format_analysis(breakdown)
    
    def _calculate_breakdown_from_traces(self) -> TimeBreakdown:
        """Calculate breakdown from actual trace data."""
        if not self._trace_data:
            return TimeBreakdown(total_time_ms=0)
        
        totals = {cat: 0.0 for cat in TimeCategory}
        
        for event in self._trace_data:
            category = event['category']
            totals[category] += event['duration_ms']
        
        total_ms = sum(totals.values())
        
        return TimeBreakdown(
            total_time_ms=total_ms,
            compute_ms=totals[TimeCategory.COMPUTE],
            memory_wait_ms=totals[TimeCategory.MEMORY_WAIT],
            rematerialization_ms=totals[TimeCategory.REMATERIALIZATION],
            compilation_ms=totals[TimeCategory.COMPILATION],
            host_to_device_ms=totals[TimeCategory.HOST_TO_DEVICE],
            device_to_host_ms=totals[TimeCategory.DEVICE_TO_HOST],
            collective_ms=totals[TimeCategory.COLLECTIVE],
            kernel_launch_ms=totals[TimeCategory.KERNEL_LAUNCH],
            idle_ms=totals[TimeCategory.IDLE],
            other_ms=totals[TimeCategory.OTHER],
        )
    
    def _estimate_breakdown_from_operations(self) -> TimeBreakdown:
        """
        Estimate time breakdown from operation records.
        
        This is a heuristic-based estimation when actual trace data isn't available.
        Uses the roofline model to estimate compute vs memory time.
        """
        if not self.profile_data.operations:
            return TimeBreakdown(total_time_ms=0)
        
        compute_ms = 0.0
        memory_ms = 0.0
        other_ms = 0.0
        compilation_ms = 0.0
        
        # TPU v4 specs for estimation
        peak_tflops = 275  # bf16
        peak_bandwidth_gbps = 1200
        balance_point = peak_tflops * 1e12 / (peak_bandwidth_gbps * 1e9)  # ~229 FLOPS/byte
        
        for op in self.profile_data.operations:
            duration_ms = op.duration_ns / 1e6
            
            if op.flops and op.bytes_accessed and op.bytes_accessed > 0:
                # Calculate arithmetic intensity
                intensity = op.flops / op.bytes_accessed
                
                # Estimate compute vs memory time based on roofline
                if intensity < balance_point:
                    # Memory-bound: most time is memory wait
                    memory_fraction = 1 - (intensity / balance_point)
                    memory_ms += duration_ms * memory_fraction
                    compute_ms += duration_ms * (1 - memory_fraction)
                else:
                    # Compute-bound: most time is compute
                    compute_ms += duration_ms
            else:
                other_ms += duration_ms
        
        # Add compilation time from compilation records
        for comp in self.profile_data.compilations:
            if not comp.cache_hit:
                compilation_ms += comp.compilation_time_ms
        
        total_ms = compute_ms + memory_ms + compilation_ms + other_ms
        
        # Build operation-level breakdown
        by_operation = {}
        for op in self.profile_data.operations:
            duration_ms = op.duration_ns / 1e6
            if op.flops and op.bytes_accessed and op.bytes_accessed > 0:
                intensity = op.flops / op.bytes_accessed
                if intensity < balance_point:
                    mem_frac = 1 - (intensity / balance_point)
                    by_operation[op.name] = {
                        "total_ms": duration_ms,
                        "compute_ms": duration_ms * (1 - mem_frac),
                        "memory_ms": duration_ms * mem_frac,
                    }
                else:
                    by_operation[op.name] = {
                        "total_ms": duration_ms,
                        "compute_ms": duration_ms,
                        "memory_ms": 0,
                    }
        
        return TimeBreakdown(
            total_time_ms=total_ms,
            compute_ms=compute_ms,
            memory_wait_ms=memory_ms,
            compilation_ms=compilation_ms,
            other_ms=other_ms,
            by_operation=by_operation,
        )
    
    def _format_analysis(self, breakdown: TimeBreakdown) -> Dict[str, Any]:
        """Format the breakdown into an analysis result."""
        percentages = breakdown.get_percentages()
        top_categories = breakdown.get_top_categories(5)
        
        # Generate recommendations based on breakdown
        recommendations = self._generate_recommendations(breakdown)
        
        # Determine if compute-bound or memory-bound
        if breakdown.compute_ms > breakdown.memory_wait_ms * 2:
            bottleneck = "compute"
            bottleneck_desc = "Your workload is compute-bound (good for TPU!)"
        elif breakdown.memory_wait_ms > breakdown.compute_ms * 2:
            bottleneck = "memory"
            bottleneck_desc = "Your workload is memory-bound - consider operation fusion or larger batch sizes"
        else:
            bottleneck = "balanced"
            bottleneck_desc = "Your workload is balanced between compute and memory"
        
        return {
            "status": "ok" if breakdown.total_time_ms > 0 else "no_data",
            "breakdown": breakdown,
            "percentages": percentages,
            "top_categories": top_categories,
            "bottleneck": bottleneck,
            "bottleneck_description": bottleneck_desc,
            "recommendations": recommendations,
            "summary": {
                "total_time_ms": breakdown.total_time_ms,
                "compute_pct": percentages.get("compute", 0),
                "memory_pct": percentages.get("memory_wait", 0),
                "compilation_pct": percentages.get("compilation", 0),
            },
        }
    
    def _generate_recommendations(self, breakdown: TimeBreakdown) -> List[Dict[str, Any]]:
        """Generate recommendations based on time breakdown."""
        recommendations = []
        pct = breakdown.get_percentages()
        
        # High memory wait time
        if pct.get("memory_wait", 0) > 30:
            recommendations.append({
                "severity": "warning",
                "category": "memory_bound",
                "message": f"Memory wait time is {pct['memory_wait']:.1f}% of total - workload is memory-bound",
                "impact": "high",
                "suggestion": "Consider: larger batch sizes, operation fusion, or mixed precision (bfloat16)",
            })
        
        # High compilation time
        if pct.get("compilation", 0) > 20 or breakdown.compilation_ms > 5000:
            recommendations.append({
                "severity": "warning",
                "category": "compilation_overhead",
                "message": f"Compilation takes {breakdown.compilation_ms:.0f}ms ({pct['compilation']:.1f}%)",
                "impact": "medium",
                "suggestion": "Use static shapes, enable persistent cache: jax.config.update('jax_compilation_cache_dir', '/tmp/jax_cache')",
            })
        
        # High rematerialization time
        if pct.get("rematerialization", 0) > 15:
            recommendations.append({
                "severity": "info",
                "category": "rematerialization",
                "message": f"Rematerialization takes {pct['rematerialization']:.1f}% of time",
                "impact": "medium",
                "suggestion": "Review gradient checkpointing strategy - you may be recomputing too much",
            })
        
        # High collective time (for multi-device)
        if pct.get("collective", 0) > 20:
            recommendations.append({
                "severity": "warning",
                "category": "collective_overhead",
                "message": f"Collective operations take {pct['collective']:.1f}% of time",
                "impact": "medium",
                "suggestion": "Consider gradient accumulation or reducing communication frequency",
            })
        
        # High data transfer time
        transfer_pct = pct.get("host_to_device", 0) + pct.get("device_to_host", 0)
        if transfer_pct > 10:
            recommendations.append({
                "severity": "warning",
                "category": "data_transfer",
                "message": f"Host↔Device transfers take {transfer_pct:.1f}% of time",
                "impact": "medium",
                "suggestion": "Prefetch data, use tf.data pipelines, or keep data on device longer",
            })
        
        # Low compute utilization
        if pct.get("compute", 0) < 50 and breakdown.total_time_ms > 100:
            recommendations.append({
                "severity": "info",
                "category": "low_compute",
                "message": f"Only {pct['compute']:.1f}% of time is spent on actual compute",
                "impact": "high",
                "suggestion": "TPU compute is underutilized - review the bottlenecks above",
            })
        
        return recommendations
    
    def get_visualization_data(self) -> Dict[str, Any]:
        """Get data formatted for visualization."""
        analysis = self.analyze()
        
        if analysis["status"] == "no_data":
            return {"status": "no_data"}
        
        breakdown = analysis["breakdown"]
        pct = analysis["percentages"]
        
        # Pie chart data
        pie_data = {
            "labels": [],
            "values": [],
            "colors": [],
        }
        
        color_map = {
            "compute": "#3fb950",
            "memory_wait": "#f85149",
            "rematerialization": "#a371f7",
            "compilation": "#d29922",
            "host_to_device": "#58a6ff",
            "device_to_host": "#58a6ff",
            "collective": "#f778ba",
            "kernel_launch": "#8b949e",
            "idle": "#6e7681",
            "other": "#484f58",
        }
        
        for cat, val in pct.items():
            if val > 0.5:  # Only show categories > 0.5%
                pie_data["labels"].append(cat.replace("_", " ").title())
                pie_data["values"].append(val)
                pie_data["colors"].append(color_map.get(cat, "#8b949e"))
        
        return {
            "status": "ok",
            "pie_chart": pie_data,
            "bar_chart": {
                "labels": [c[0] for c in analysis["top_categories"]],
                "values": [c[1] for c in analysis["top_categories"]],
            },
            "bottleneck": analysis["bottleneck"],
            "total_time_ms": breakdown.total_time_ms,
        }
    
    def print_breakdown(self):
        """Print a formatted breakdown to console."""
        analysis = self.analyze()
        
        if analysis["status"] == "no_data":
            print("No timing data available")
            return
        
        breakdown = analysis["breakdown"]
        pct = analysis["percentages"]
        
        print("\n" + "=" * 60)
        print("  TPUsight Time Breakdown")
        print("=" * 60)
        print(f"  Total Time: {breakdown.total_time_ms:.2f} ms")
        print("-" * 60)
        
        # Print categories sorted by time
        categories = [
            ("Compute", breakdown.compute_ms, pct["compute"], "🟢"),
            ("Memory Wait", breakdown.memory_wait_ms, pct["memory_wait"], "🔴"),
            ("Rematerialization", breakdown.rematerialization_ms, pct["rematerialization"], "🟣"),
            ("Compilation", breakdown.compilation_ms, pct["compilation"], "🟡"),
            ("Host→Device", breakdown.host_to_device_ms, pct["host_to_device"], "🔵"),
            ("Device→Host", breakdown.device_to_host_ms, pct["device_to_host"], "🔵"),
            ("Collective Ops", breakdown.collective_ms, pct["collective"], "🟠"),
            ("Kernel Launch", breakdown.kernel_launch_ms, pct["kernel_launch"], "⚪"),
            ("Idle", breakdown.idle_ms, pct["idle"], "⬛"),
            ("Other", breakdown.other_ms, pct["other"], "⬜"),
        ]
        
        # Sort by time and print non-zero categories
        for name, ms, percent, icon in sorted(categories, key=lambda x: x[1], reverse=True):
            if ms > 0.01:  # Only show if > 0.01ms
                bar_len = int(percent / 2)
                bar = "█" * bar_len + "░" * (50 - bar_len)
                print(f"  {icon} {name:18} {ms:10.2f} ms  {percent:5.1f}%  {bar}")
        
        print("-" * 60)
        print(f"  Bottleneck: {analysis['bottleneck_description']}")
        print("=" * 60 + "\n")
    
    def __repr__(self) -> str:
        analysis = self.analyze()
        if analysis["status"] == "no_data":
            return "TimeBreakdownAnalyzer(no data)"
        return (
            f"TimeBreakdownAnalyzer(total={analysis['breakdown'].total_time_ms:.1f}ms, "
            f"compute={analysis['percentages']['compute']:.1f}%, "
            f"memory={analysis['percentages']['memory_wait']:.1f}%)"
        )

