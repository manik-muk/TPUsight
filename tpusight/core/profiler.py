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
        
        elif format == "html":
            html_content = self._generate_html_report()
            with open(filepath, 'w') as f:
                f.write(html_content)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _generate_html_report(self) -> str:
        """Generate a standalone HTML report with charts."""
        from tpusight.utils.helpers import format_bytes, format_flops, format_duration
        
        summary = self.get_summary()
        doctor_diagnosis = self.doctor.diagnose()
        systolic_analysis = self.systolic.analyze()
        padding_analysis = self.padding.analyze()
        cache_analysis = self.cache.analyze()
        
        # Build operations table rows
        ops_rows = ""
        for op in self.profile_data.operations[:50]:  # Limit to 50
            mxu = f"{op.mxu_utilization:.1f}%" if op.mxu_utilization else "-"
            padding = f"{op.padding_waste_pct:.1f}%" if op.padding_waste_pct else "-"
            flops = format_flops(op.flops) if op.flops else "-"
            ops_rows += f"""
            <tr>
                <td>{op.name}</td>
                <td>{op.op_type.value}</td>
                <td>{format_duration(op.duration_ns / 1e9)}</td>
                <td>{flops}</td>
                <td>{mxu}</td>
                <td>{padding}</td>
            </tr>"""
        
        # Build recommendations
        recs_html = ""
        for rec in doctor_diagnosis["top_recommendations"][:10]:
            severity_colors = {"critical": "#f85149", "warning": "#d29922", "info": "#58a6ff"}
            color = severity_colors.get(rec["severity"], "#58a6ff")
            recs_html += f"""
            <div class="issue" style="border-left-color: {color}">
                <div class="issue-header">
                    <span class="severity" style="background: {color}">{rec['severity'].upper()}</span>
                    <strong>{rec['title']}</strong>
                </div>
                <p>{rec['message']}</p>
                <p class="suggestion"><strong>Suggestion:</strong> {rec['suggestion']}</p>
                {f'<p class="impact">Impact: {rec["impact_estimate"]}</p>' if rec.get("impact_estimate") else ''}
            </div>"""
        
        # MXU metrics
        mxu_util = systolic_analysis["metrics"].overall_utilization if systolic_analysis["status"] == "ok" else 0
        
        # Cache metrics  
        cache_hit_rate = cache_analysis["metrics"].cache_hit_rate if cache_analysis["status"] == "ok" else 0
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>TPUsight Report - {self.session_id}</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: 'SF Mono', 'Fira Code', 'JetBrains Mono', monospace;
            background: #0f1419;
            color: #e6edf3;
            padding: 24px;
            line-height: 1.6;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        .header {{
            display: flex;
            align-items: center;
            gap: 20px;
            margin-bottom: 32px;
            padding-bottom: 20px;
            border-bottom: 1px solid #30363d;
        }}
        .logo {{
            font-size: 32px;
            font-weight: 700;
            background: linear-gradient(135deg, #58a6ff, #a371f7);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .meta {{ color: #8b949e; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 16px; margin-bottom: 32px; }}
        .card {{
            background: #232a33;
            border: 1px solid #30363d;
            border-radius: 8px;
            padding: 20px;
        }}
        .card-title {{ color: #8b949e; font-size: 12px; text-transform: uppercase; margin-bottom: 8px; }}
        .card-value {{ font-size: 28px; font-weight: 700; }}
        .card-subtitle {{ color: #8b949e; font-size: 12px; margin-top: 4px; }}
        .progress {{ height: 8px; background: #1a1f26; border-radius: 4px; margin-top: 12px; overflow: hidden; }}
        .progress-bar {{ height: 100%; border-radius: 4px; }}
        .section {{ margin-bottom: 32px; }}
        .section-title {{ font-size: 20px; font-weight: 600; margin-bottom: 16px; }}
        table {{ width: 100%; border-collapse: collapse; background: #232a33; border-radius: 8px; overflow: hidden; }}
        th {{ background: #1a1f26; text-align: left; padding: 12px; color: #8b949e; font-weight: 600; }}
        td {{ padding: 12px; border-bottom: 1px solid #30363d; }}
        tr:hover {{ background: #1a1f26; }}
        .health-score {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 80px;
            height: 80px;
            border-radius: 50%;
            font-size: 28px;
            font-weight: 700;
            background: {'linear-gradient(135deg, #3fb950, #2ea043)' if doctor_diagnosis['health_score'] >= 70 else 'linear-gradient(135deg, #d29922, #bb8009)' if doctor_diagnosis['health_score'] >= 50 else 'linear-gradient(135deg, #f85149, #cf222e)'};
        }}
        .issue {{
            background: #1a1f26;
            border-left: 3px solid;
            padding: 16px;
            margin-bottom: 12px;
            border-radius: 0 8px 8px 0;
        }}
        .issue-header {{ display: flex; align-items: center; gap: 12px; margin-bottom: 8px; }}
        .severity {{ padding: 2px 8px; border-radius: 4px; font-size: 11px; color: white; }}
        .suggestion {{ color: #8b949e; margin-top: 8px; }}
        .impact {{ color: #58a6ff; font-size: 13px; margin-top: 4px; }}
        .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }}
        @media (max-width: 768px) {{ .two-col {{ grid-template-columns: 1fr; }} }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <span class="logo">TPUsight</span>
            <span class="meta">
                Session: {self.session_id} |
                Device: {summary['device']['type']} x{summary['device']['count']} |
                Duration: {summary['duration_seconds']:.2f}s
            </span>
        </div>
        
        <div class="section">
            <div class="grid">
                <div class="card">
                    <div class="card-title">Health Score</div>
                    <div style="display: flex; align-items: center; gap: 16px;">
                        <div class="health-score">{doctor_diagnosis['health_score']}</div>
                        <div>
                            <div class="card-value" style="font-size: 18px;">{doctor_diagnosis['health_status'].replace('_', ' ').title()}</div>
                            <div class="card-subtitle">{doctor_diagnosis['critical_count']} critical, {doctor_diagnosis['warning_count']} warnings</div>
                        </div>
                    </div>
                </div>
                <div class="card">
                    <div class="card-title">Total Operations</div>
                    <div class="card-value">{summary['operations']['total']:,}</div>
                    <div class="card-subtitle">{format_flops(summary['operations']['total_flops'])} total</div>
                </div>
                <div class="card">
                    <div class="card-title">MXU Utilization</div>
                    <div class="card-value">{mxu_util:.1f}%</div>
                    <div class="progress">
                        <div class="progress-bar" style="width: {mxu_util}%; background: {'#3fb950' if mxu_util >= 70 else '#d29922' if mxu_util >= 50 else '#f85149'};"></div>
                    </div>
                </div>
                <div class="card">
                    <div class="card-title">Cache Hit Rate</div>
                    <div class="card-value">{cache_hit_rate:.1f}%</div>
                    <div class="progress">
                        <div class="progress-bar" style="width: {cache_hit_rate}%; background: {'#3fb950' if cache_hit_rate >= 80 else '#d29922'};"></div>
                    </div>
                </div>
                <div class="card">
                    <div class="card-title">Peak Memory</div>
                    <div class="card-value">{format_bytes(summary['memory']['peak_bytes'])}</div>
                    <div class="card-subtitle">{summary['memory']['total_allocations']} allocations</div>
                </div>
                <div class="card">
                    <div class="card-title">Total Time</div>
                    <div class="card-value">{format_duration(summary['operations']['total_time_ns'] / 1e9)}</div>
                    <div class="card-subtitle">Across all operations</div>
                </div>
            </div>
        </div>
        
        <div class="two-col">
            <div class="section">
                <div class="section-title">🩺 Recommendations</div>
                {recs_html if recs_html else '<p style="color: #8b949e;">No issues found - your code is well optimized!</p>'}
            </div>
            
            <div class="section">
                <div class="section-title">📊 Operations</div>
                <div style="overflow-x: auto;">
                    <table>
                        <thead>
                            <tr>
                                <th>Name</th>
                                <th>Type</th>
                                <th>Duration</th>
                                <th>FLOPS</th>
                                <th>MXU Util</th>
                                <th>Padding</th>
                            </tr>
                        </thead>
                        <tbody>
                            {ops_rows if ops_rows else '<tr><td colspan="6" style="text-align:center; color:#8b949e;">No operations recorded</td></tr>'}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
        
        <div style="text-align: center; margin-top: 40px; color: #8b949e; font-size: 12px;">
            Generated by TPUsight v0.1.0
        </div>
    </div>
</body>
</html>"""
        return html
    
    def summary(self) -> None:
        """Print a quick text summary to the console (works in any environment)."""
        from tpusight.utils.helpers import format_bytes, format_flops, format_duration
        
        s = self.get_summary()
        d = self.doctor.diagnose()
        
        print("=" * 60)
        print(f"  TPUsight Summary - {self.session_id}")
        print("=" * 60)
        print(f"  Device: {s['device']['type']} x{s['device']['count']}")
        print(f"  Health Score: {d['health_score']}/100 ({d['health_status']})")
        print("-" * 60)
        print(f"  Operations: {s['operations']['total']}")
        print(f"  Total FLOPS: {format_flops(s['operations']['total_flops'])}")
        print(f"  Total Time: {format_duration(s['operations']['total_time_ns'] / 1e9)}")
        print(f"  Cache Hit Rate: {s['compilation']['cache_hit_rate'] * 100:.1f}%")
        print(f"  Peak Memory: {format_bytes(s['memory']['peak_bytes'])}")
        print("-" * 60)
        print(f"  Issues: {d['critical_count']} critical, {d['warning_count']} warnings")
        
        if d['top_recommendations']:
            print("\n  Top Recommendations:")
            for i, rec in enumerate(d['top_recommendations'][:3], 1):
                icon = {"critical": "🔴", "warning": "🟡", "info": "🔵"}.get(rec['severity'], "⚪")
                print(f"    {i}. {icon} {rec['message'][:60]}...")
        
        print("=" * 60)
    
    def __repr__(self) -> str:
        return (
            f"TPUsight(session='{self.session_id}', "
            f"device='{self.profile_data.device_type}', "
            f"ops={self.profile_data.total_ops})"
        )

