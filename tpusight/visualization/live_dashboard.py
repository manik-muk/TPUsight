"""Live dashboard for real-time TPU profiling visualization."""

from typing import Dict, List, Any, Optional, TYPE_CHECKING
from threading import Thread
import time

try:
    import ipywidgets as widgets
    from IPython.display import display, HTML, clear_output
    HAS_WIDGETS = True
except ImportError:
    HAS_WIDGETS = False

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

if TYPE_CHECKING:
    from tpusight.core.live_profiler import LiveProfiler, LiveMetrics, LiveAlert


# Colors
COLORS = {
    "bg": "#0f1419",
    "card": "#232a33",
    "border": "#30363d",
    "text": "#e6edf3",
    "text_dim": "#8b949e",
    "green": "#3fb950",
    "yellow": "#d29922",
    "red": "#f85149",
    "blue": "#58a6ff",
    "purple": "#a371f7",
}


class LiveDashboard:
    """
    Real-time dashboard for live profiling.
    
    Updates automatically as operations are profiled, showing:
    - Live metrics (ops/sec, utilization, etc.)
    - Real-time charts
    - Alert feed
    - Operation log
    """
    
    def __init__(self, live_profiler: "LiveProfiler"):
        self.profiler = live_profiler
        self._widgets: Dict[str, widgets.Widget] = {}
        self._is_running = False
        self._update_thread: Optional[Thread] = None
        
        # Chart data
        self._time_data: List[float] = []
        self._mxu_data: List[float] = []
        self._ops_data: List[float] = []
    
    def display(self, update_interval: float = 1.0) -> Optional[widgets.Widget]:
        """
        Display the live dashboard.
        
        Args:
            update_interval: How often to refresh (seconds)
            
        Returns:
            Dashboard widget
        """
        if not HAS_WIDGETS:
            print("ipywidgets not available. Use text-based monitoring instead:")
            print("  live_profiler.on_alert(lambda a: print(f'ALERT: {a.message}'))")
            return None
        
        # Create widgets
        self._create_widgets()
        
        # Register callbacks
        self.profiler.on_metrics(self._on_metrics_update)
        self.profiler.on_alert(self._on_alert)
        
        # Layout
        dashboard = widgets.VBox([
            self._widgets["header"],
            widgets.HBox([
                self._widgets["metrics_panel"],
                self._widgets["chart_panel"],
            ]),
            widgets.HBox([
                self._widgets["alerts_panel"],
                self._widgets["ops_panel"],
            ]),
        ], layout=widgets.Layout(width="100%"))
        
        # Start update loop
        self._is_running = True
        self._update_thread = Thread(target=self._update_loop, args=(update_interval,), daemon=True)
        self._update_thread.start()
        
        display(HTML(self._get_styles()))
        display(dashboard)
        
        return dashboard
    
    def stop(self):
        """Stop the live dashboard updates."""
        self._is_running = False
        if self._update_thread:
            self._update_thread.join(timeout=2.0)
    
    def _get_styles(self) -> str:
        """Get CSS styles for the dashboard."""
        return f"""
        <style>
            .live-header {{
                font-family: 'SF Mono', monospace;
                padding: 16px;
                background: {COLORS['card']};
                border-radius: 8px;
                margin-bottom: 16px;
            }}
            .live-title {{
                font-size: 24px;
                font-weight: 700;
                background: linear-gradient(135deg, {COLORS['red']}, {COLORS['yellow']});
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }}
            .live-status {{
                display: inline-block;
                padding: 4px 12px;
                border-radius: 12px;
                font-size: 12px;
                font-weight: 600;
                animation: pulse 2s infinite;
            }}
            .live-status-active {{
                background: {COLORS['red']};
                color: white;
            }}
            @keyframes pulse {{
                0%, 100% {{ opacity: 1; }}
                50% {{ opacity: 0.6; }}
            }}
            .metric-card {{
                background: {COLORS['card']};
                border: 1px solid {COLORS['border']};
                border-radius: 8px;
                padding: 16px;
                margin: 8px;
                min-width: 150px;
            }}
            .metric-value {{
                font-size: 28px;
                font-weight: 700;
                color: {COLORS['text']};
            }}
            .metric-label {{
                font-size: 11px;
                color: {COLORS['text_dim']};
                text-transform: uppercase;
            }}
            .alert-item {{
                padding: 8px 12px;
                margin: 4px 0;
                border-radius: 4px;
                font-size: 12px;
                border-left: 3px solid;
            }}
            .alert-critical {{
                background: rgba(248, 81, 73, 0.15);
                border-color: {COLORS['red']};
            }}
            .alert-warning {{
                background: rgba(210, 153, 34, 0.15);
                border-color: {COLORS['yellow']};
            }}
            .alert-info {{
                background: rgba(88, 166, 255, 0.15);
                border-color: {COLORS['blue']};
            }}
        </style>
        """
    
    def _create_widgets(self):
        """Create all dashboard widgets."""
        # Header
        self._widgets["header"] = widgets.HTML(self._render_header())
        
        # Metrics panel
        self._widgets["ops_metric"] = widgets.HTML(self._render_metric("0", "Total Ops"))
        self._widgets["rate_metric"] = widgets.HTML(self._render_metric("0", "Ops/sec"))
        self._widgets["mxu_metric"] = widgets.HTML(self._render_metric("0%", "MXU Util"))
        self._widgets["alerts_metric"] = widgets.HTML(self._render_metric("0", "Alerts"))
        
        self._widgets["metrics_panel"] = widgets.VBox([
            widgets.HTML('<div style="font-weight: 600; margin: 8px;">📊 Live Metrics</div>'),
            widgets.HBox([
                self._widgets["ops_metric"],
                self._widgets["rate_metric"],
            ]),
            widgets.HBox([
                self._widgets["mxu_metric"],
                self._widgets["alerts_metric"],
            ]),
        ], layout=widgets.Layout(width="40%", padding="8px"))
        
        # Chart panel
        self._widgets["chart_output"] = widgets.Output()
        self._widgets["chart_panel"] = widgets.VBox([
            widgets.HTML('<div style="font-weight: 600; margin: 8px;">📈 Real-time MXU Utilization</div>'),
            self._widgets["chart_output"],
        ], layout=widgets.Layout(width="60%", padding="8px"))
        
        # Alerts panel
        self._widgets["alerts_output"] = widgets.Output()
        self._widgets["alerts_panel"] = widgets.VBox([
            widgets.HTML('<div style="font-weight: 600; margin: 8px;">🚨 Live Alerts</div>'),
            self._widgets["alerts_output"],
        ], layout=widgets.Layout(width="50%", padding="8px", max_height="200px", overflow="auto"))
        
        # Operations panel
        self._widgets["ops_output"] = widgets.Output()
        self._widgets["ops_panel"] = widgets.VBox([
            widgets.HTML('<div style="font-weight: 600; margin: 8px;">⚡ Recent Operations</div>'),
            self._widgets["ops_output"],
        ], layout=widgets.Layout(width="50%", padding="8px", max_height="200px", overflow="auto"))
    
    def _render_header(self) -> str:
        """Render the header HTML."""
        return f"""
        <div class="live-header">
            <span class="live-title">TPUsight Live</span>
            <span class="live-status live-status-active">● RECORDING</span>
            <span style="margin-left: 16px; color: {COLORS['text_dim']};">
                Session: {self.profiler.session_id}
            </span>
        </div>
        """
    
    def _render_metric(self, value: str, label: str, color: str = COLORS["text"]) -> str:
        """Render a metric card."""
        return f"""
        <div class="metric-card">
            <div class="metric-value" style="color: {color};">{value}</div>
            <div class="metric-label">{label}</div>
        </div>
        """
    
    def _on_metrics_update(self, metrics: "LiveMetrics"):
        """Handle metrics update callback."""
        # Store for charting
        self._time_data.append(metrics.timestamp)
        self._mxu_data.append(metrics.mxu_utilization)
        self._ops_data.append(metrics.ops_per_second)
        
        # Keep last 60 points
        if len(self._time_data) > 60:
            self._time_data = self._time_data[-60:]
            self._mxu_data = self._mxu_data[-60:]
            self._ops_data = self._ops_data[-60:]
    
    def _on_alert(self, alert: "LiveAlert"):
        """Handle alert callback."""
        pass  # Updates happen in _update_loop
    
    def _update_loop(self, interval: float):
        """Background loop for updating dashboard."""
        while self._is_running:
            try:
                self._refresh_widgets()
            except Exception:
                pass
            time.sleep(interval)
    
    def _refresh_widgets(self):
        """Refresh all widget contents."""
        metrics = self.profiler.get_current_metrics()
        
        # Update metrics
        mxu_color = COLORS["green"] if metrics.mxu_utilization >= 70 else \
                    COLORS["yellow"] if metrics.mxu_utilization >= 50 else COLORS["red"]
        
        self._widgets["ops_metric"].value = self._render_metric(
            f"{metrics.total_ops:,}", "Total Ops"
        )
        self._widgets["rate_metric"].value = self._render_metric(
            f"{metrics.ops_per_second:.1f}", "Ops/sec"
        )
        self._widgets["mxu_metric"].value = self._render_metric(
            f"{metrics.mxu_utilization:.0f}%", "MXU Util", mxu_color
        )
        self._widgets["alerts_metric"].value = self._render_metric(
            str(metrics.active_alerts), "Alerts (60s)",
            COLORS["red"] if metrics.active_alerts > 0 else COLORS["text"]
        )
        
        # Update chart
        if HAS_PLOTLY and self._mxu_data:
            with self._widgets["chart_output"]:
                clear_output(wait=True)
                
                fig = go.Figure()
                
                # Relative time (seconds ago)
                if self._time_data:
                    now = self._time_data[-1]
                    x_data = [t - now for t in self._time_data]
                else:
                    x_data = []
                
                fig.add_trace(go.Scatter(
                    x=x_data,
                    y=self._mxu_data,
                    mode='lines',
                    fill='tozeroy',
                    line=dict(color=COLORS["blue"], width=2),
                    fillcolor='rgba(88, 166, 255, 0.2)',
                ))
                
                # Add threshold lines
                fig.add_hline(y=70, line_dash="dash", line_color=COLORS["green"],
                             annotation_text="Good (70%)")
                fig.add_hline(y=50, line_dash="dash", line_color=COLORS["yellow"],
                             annotation_text="Warning (50%)")
                
                fig.update_layout(
                    height=200,
                    margin=dict(l=40, r=20, t=20, b=40),
                    paper_bgcolor=COLORS["bg"],
                    plot_bgcolor=COLORS["bg"],
                    font=dict(color=COLORS["text"]),
                    xaxis=dict(
                        title="Time (seconds)",
                        gridcolor=COLORS["border"],
                        range=[-60, 0],
                    ),
                    yaxis=dict(
                        title="MXU %",
                        gridcolor=COLORS["border"],
                        range=[0, 105],
                    ),
                    showlegend=False,
                )
                
                fig.show()
        
        # Update alerts
        alerts = self.profiler.get_recent_alerts(10)
        with self._widgets["alerts_output"]:
            clear_output(wait=True)
            if alerts:
                for alert in reversed(alerts):
                    severity_class = f"alert-{alert.severity}"
                    print_html = f"""
                    <div class="alert-item {severity_class}">
                        <strong>{alert.category}</strong>: {alert.message}
                        <div style="color: {COLORS['text_dim']}; font-size: 10px;">
                            {alert.operation or ''} • {time.strftime('%H:%M:%S', time.localtime(alert.timestamp))}
                        </div>
                    </div>
                    """
                    display(HTML(print_html))
            else:
                display(HTML(f'<div style="color: {COLORS["text_dim"]}; padding: 16px;">No alerts yet</div>'))
        
        # Update recent ops
        with self._widgets["ops_output"]:
            clear_output(wait=True)
            recent_ops = list(self.profiler._recent_ops)[-10:]
            if recent_ops:
                for op in reversed(recent_ops):
                    mxu_str = f"{op.mxu_utilization:.0f}%" if op.mxu_utilization else "-"
                    display(HTML(f"""
                    <div style="padding: 4px 8px; border-bottom: 1px solid {COLORS['border']}; font-size: 12px;">
                        <span style="color: {COLORS['blue']};">{op.name}</span>
                        <span style="color: {COLORS['text_dim']}; margin-left: 8px;">
                            {op.duration_ns/1e6:.2f}ms | MXU: {mxu_str}
                        </span>
                    </div>
                    """))
            else:
                display(HTML(f'<div style="color: {COLORS["text_dim"]}; padding: 16px;">No operations yet</div>'))


def print_live_status(profiler: "LiveProfiler"):
    """Print live status to console (works without widgets)."""
    metrics = profiler.get_current_metrics()
    alerts = profiler.get_recent_alerts(5)
    
    # Clear screen (works in most terminals)
    print("\033[2J\033[H", end="")
    
    print("=" * 60)
    print(f"  🔴 TPUsight Live - {profiler.session_id}")
    print("=" * 60)
    print(f"  Total Ops: {metrics.total_ops:,}")
    print(f"  Ops/sec:   {metrics.ops_per_second:.1f}")
    print(f"  MXU Util:  {metrics.mxu_utilization:.1f}%")
    print(f"  Last Op:   {metrics.last_op_name} ({metrics.last_op_duration_ms:.2f}ms)")
    print("-" * 60)
    
    if alerts:
        print("  Recent Alerts:")
        for alert in alerts[-5:]:
            icon = {"critical": "🔴", "warning": "🟡", "info": "🔵"}.get(alert.severity, "⚪")
            print(f"    {icon} {alert.message}")
    else:
        print("  No alerts")
    
    print("=" * 60)

