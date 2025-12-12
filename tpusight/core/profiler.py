"""Main TPUsight profiler class."""

from typing import Optional, Callable, Any, Dict
from contextlib import contextmanager
from functools import wraps
import time
import uuid

try:
    import jax
    HAS_JAX = True
except ImportError:
    HAS_JAX = False

from tpusight.core.data_collector import ProfileData
from tpusight.core.jax_tracer import JAXTracer


class TPUsight:
    """
    TPUsight - A comprehensive TPU profiler.
    
    Provides deep visibility into TPU workloads with actionable
    optimization insights, viewable through an interactive Jupyter GUI.
    
    Example usage:
        >>> profiler = TPUsight()
        >>> @profiler.trace
        ... def my_model(x, w):
        ...     return jnp.dot(x, w)
        >>> 
        >>> result = my_model(x, w)
        >>> profiler.dashboard()
    """
    
    def __init__(
        self,
        collect_hlo: bool = True,
        collect_memory: bool = True,
        cache_analysis: bool = True,
        sample_rate: float = 1.0,
        session_name: Optional[str] = None,
    ):
        """
        Initialize the TPUsight profiler.
        
        Args:
            collect_hlo: Whether to collect HLO IR information
            collect_memory: Whether to track memory allocations
            cache_analysis: Whether to monitor executable cache
            sample_rate: Sampling rate for operations (1.0 = all)
            session_name: Optional name for this profiling session
        """
        self.collect_hlo = collect_hlo
        self.collect_memory = collect_memory
        self.cache_analysis = cache_analysis
        self.sample_rate = sample_rate
        
        # Generate session ID
        self.session_id = session_name or f"tpusight_{uuid.uuid4().hex[:8]}"
        
        # Initialize profile data
        self.profile_data = ProfileData(
            session_id=self.session_id,
            start_time=time.time(),
        )
        
        # Detect device info
        self._detect_device()
        
        # Initialize tracer
        self._tracer = JAXTracer(self.profile_data)
        
        # Dashboard reference
        self._dashboard = None
        
        # Analyzers (lazy loaded)
        self._analyzers: Dict[str, Any] = {}
    
    def _detect_device(self):
        """Detect TPU/GPU/CPU device information."""
        if not HAS_JAX:
            self.profile_data.device_type = "unknown"
            self.profile_data.device_count = 0
            return
        
        try:
            devices = jax.devices()
            if devices:
                device = devices[0]
                self.profile_data.device_type = device.platform
                self.profile_data.device_count = len(devices)
                
                # Try to get TPU version
                if device.platform == "tpu":
                    # Device kind contains version info like "TPU v4"
                    self.profile_data.tpu_version = getattr(device, 'device_kind', None)
        except Exception:
            self.profile_data.device_type = "unknown"
            self.profile_data.device_count = 0
    
    def trace(self, fn: Callable) -> Callable:
        """
        Decorator to trace a function's execution.
        
        Args:
            fn: The function to trace
            
        Returns:
            Wrapped function with tracing enabled
        """
        return self._tracer.trace_function(fn)
    
    @contextmanager
    def trace_context(self, name: Optional[str] = None):
        """
        Context manager for tracing a block of code.
        
        Args:
            name: Optional name for this trace context
            
        Example:
            >>> with profiler.trace_context("forward_pass"):
            ...     result = model(x)
        """
        self._tracer.start()
        try:
            yield self
        finally:
            self._tracer.stop()
    
    def start(self):
        """Start profiling."""
        self._tracer.start()
    
    def stop(self):
        """Stop profiling and finalize data."""
        self._tracer.stop()
        self.profile_data.finalize()
    
    def reset(self):
        """Reset profiling data for a new session."""
        self.session_id = f"tpusight_{uuid.uuid4().hex[:8]}"
        self.profile_data = ProfileData(
            session_id=self.session_id,
            start_time=time.time(),
        )
        self._detect_device()
        self._tracer = JAXTracer(self.profile_data)
        self._analyzers = {}
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the profiling session."""
        return self.profile_data.get_summary()
    
    def _get_analyzer(self, name: str) -> Any:
        """Lazily load and cache an analyzer."""
        if name not in self._analyzers:
            if name == "systolic":
                from tpusight.analyzers.systolic import SystolicAnalyzer
                self._analyzers[name] = SystolicAnalyzer(self.profile_data)
            elif name == "padding":
                from tpusight.analyzers.padding import PaddingAnalyzer
                self._analyzers[name] = PaddingAnalyzer(self.profile_data)
            elif name == "fusion":
                from tpusight.analyzers.fusion import FusionAnalyzer
                self._analyzers[name] = FusionAnalyzer(self.profile_data)
            elif name == "cache":
                from tpusight.analyzers.cache import CacheAnalyzer
                self._analyzers[name] = CacheAnalyzer(self.profile_data)
            elif name == "memory":
                from tpusight.analyzers.memory import MemoryAnalyzer
                self._analyzers[name] = MemoryAnalyzer(self.profile_data)
            elif name == "doctor":
                from tpusight.analyzers.doctor import TPUDoctor
                self._analyzers[name] = TPUDoctor(self.profile_data)
        
        return self._analyzers[name]
    
    @property
    def systolic(self):
        """Access the systolic array utilization analyzer."""
        return self._get_analyzer("systolic")
    
    @property
    def padding(self):
        """Access the padding/tiling inefficiency analyzer."""
        return self._get_analyzer("padding")
    
    @property
    def fusion(self):
        """Access the fusion failure analyzer."""
        return self._get_analyzer("fusion")
    
    @property
    def cache(self):
        """Access the cache profiler."""
        return self._get_analyzer("cache")
    
    @property
    def memory(self):
        """Access the memory diagnostics analyzer."""
        return self._get_analyzer("memory")
    
    @property
    def doctor(self):
        """Access the TPU Doctor for optimization suggestions."""
        return self._get_analyzer("doctor")
    
    def dashboard(self, height: int = 800) -> Any:
        """
        Display the interactive profiling dashboard.
        
        Args:
            height: Height of the dashboard in pixels
            
        Returns:
            Dashboard widget (in Jupyter) or None (in terminal)
        """
        from tpusight.visualization.dashboard import Dashboard
        
        self._dashboard = Dashboard(self)
        return self._dashboard.display(height=height)
    
    def report(self) -> str:
        """
        Generate a text report of profiling results.
        
        Returns:
            Formatted text report
        """
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        from io import StringIO
        
        output = StringIO()
        console = Console(file=output, force_terminal=True)
        
        summary = self.get_summary()
        
        # Header
        console.print(Panel.fit(
            f"[bold blue]TPUsight Profiling Report[/bold blue]\n"
            f"Session: {self.session_id}",
            border_style="blue"
        ))
        
        # Device info
        device = summary["device"]
        console.print(f"\n[bold]Device:[/bold] {device['type']} x{device['count']}")
        if device.get('version'):
            console.print(f"[bold]Version:[/bold] {device['version']}")
        
        # Operations summary
        ops = summary["operations"]
        console.print(f"\n[bold]Operations:[/bold]")
        console.print(f"  Total: {ops['total']}")
        console.print(f"  Total FLOPS: {ops['total_flops']:,}")
        console.print(f"  Total time: {ops['total_time_ns'] / 1e6:.2f} ms")
        
        # Compilation summary
        comp = summary["compilation"]
        console.print(f"\n[bold]Compilation:[/bold]")
        console.print(f"  Total compilations: {comp['total']}")
        console.print(f"  Cache hit rate: {comp['cache_hit_rate'] * 100:.1f}%")
        
        # Memory summary
        mem = summary["memory"]
        console.print(f"\n[bold]Memory:[/bold]")
        console.print(f"  Peak usage: {mem['peak_bytes'] / 1e9:.2f} GB")
        console.print(f"  Total allocations: {mem['total_allocations']}")
        
        # Top issues from doctor
        console.print("\n[bold]Top Recommendations:[/bold]")
        recommendations = self.doctor.get_recommendations()
        for i, rec in enumerate(recommendations[:5], 1):
            severity_color = {
                "critical": "red",
                "warning": "yellow", 
                "info": "blue"
            }.get(rec.get("severity", "info"), "white")
            console.print(f"  {i}. [{severity_color}]{rec['message']}[/{severity_color}]")
        
        return output.getvalue()
    
    def export(self, filepath: str, format: str = "json"):
        """
        Export profiling data to a file.
        
        Args:
            filepath: Path to save the exported data
            format: Export format ("json", "csv", "html")
        """
        import json
        
        if format == "json":
            data = {
                "session": self.get_summary(),
                "operations": [
                    {
                        "name": op.name,
                        "type": op.op_type.value,
                        "duration_ns": op.duration_ns,
                        "flops": op.flops,
                        "mxu_utilization": op.mxu_utilization,
                        "padding_waste_pct": op.padding_waste_pct,
                        "input_shapes": [list(s) for s in op.input_shapes],
                        "output_shapes": [list(s) for s in op.output_shapes],
                    }
                    for op in self.profile_data.operations
                ],
                "compilations": [
                    {
                        "function": comp.function_name,
                        "cache_hit": comp.cache_hit,
                        "compilation_time_ms": comp.compilation_time_ms,
                    }
                    for comp in self.profile_data.compilations
                ],
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        
        elif format == "csv":
            import csv
            
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "name", "type", "duration_ns", "flops", 
                    "mxu_utilization", "padding_waste_pct"
                ])
                for op in self.profile_data.operations:
                    writer.writerow([
                        op.name, op.op_type.value, op.duration_ns,
                        op.flops, op.mxu_utilization, op.padding_waste_pct
                    ])
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def __repr__(self) -> str:
        return (
            f"TPUsight(session='{self.session_id}', "
            f"device='{self.profile_data.device_type}', "
            f"ops={self.profile_data.total_ops})"
        )

