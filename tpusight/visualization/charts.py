"""Chart components for TPUsight visualization using Plotly."""

from typing import Dict, List, Any, Optional, Tuple
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Dark theme colors
THEME = {
    "bg": "#0f1419",
    "paper_bg": "#1a1f26",
    "grid": "#30363d",
    "text": "#e6edf3",
    "text_secondary": "#8b949e",
    "blue": "#58a6ff",
    "green": "#3fb950",
    "yellow": "#d29922",
    "red": "#f85149",
    "purple": "#a371f7",
}

# Common layout settings for all charts
COMMON_LAYOUT = dict(
    paper_bgcolor=THEME["paper_bg"],
    plot_bgcolor=THEME["bg"],
    font=dict(family="SF Mono, Fira Code, monospace", color=THEME["text"]),
    margin=dict(l=60, r=40, t=50, b=50),
    xaxis=dict(
        gridcolor=THEME["grid"],
        zerolinecolor=THEME["grid"],
    ),
    yaxis=dict(
        gridcolor=THEME["grid"],
        zerolinecolor=THEME["grid"],
    ),
)


def create_utilization_chart(
    data: Dict[str, Any],
    title: str = "MXU Utilization"
) -> go.Figure:
    """
    Create a utilization gauge chart.
    
    Args:
        data: Dictionary with 'value' (0-100) and optional 'label'
        title: Chart title
    
    Returns:
        Plotly figure
    """
    value = data.get("value", 0)
    
    # Determine color based on value
    if value >= 80:
        color = THEME["green"]
    elif value >= 50:
        color = THEME["yellow"]
    else:
        color = THEME["red"]
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        domain=dict(x=[0, 1], y=[0, 1]),
        title=dict(text=title, font=dict(size=16, color=THEME["text"])),
        number=dict(suffix="%", font=dict(size=36, color=THEME["text"])),
        gauge=dict(
            axis=dict(
                range=[0, 100],
                tickcolor=THEME["text_secondary"],
                tickwidth=1,
            ),
            bar=dict(color=color),
            bgcolor=THEME["bg"],
            borderwidth=0,
            steps=[
                dict(range=[0, 50], color="rgba(248, 81, 73, 0.2)"),
                dict(range=[50, 80], color="rgba(210, 153, 34, 0.2)"),
                dict(range=[80, 100], color="rgba(63, 185, 80, 0.2)"),
            ],
            threshold=dict(
                line=dict(color=THEME["text"], width=2),
                thickness=0.75,
                value=value,
            ),
        ),
    ))
    
    fig.update_layout(
        **COMMON_LAYOUT,
        height=250,
    )
    
    return fig


def create_timeline_chart(
    data: List[Dict[str, Any]],
    x_field: str = "timestamp",
    y_field: str = "utilization",
    title: str = "Utilization Over Time"
) -> go.Figure:
    """
    Create a timeline chart showing metric over time.
    
    Args:
        data: List of data points with timestamp and value
        x_field: Field name for x-axis
        y_field: Field name for y-axis
        title: Chart title
    
    Returns:
        Plotly figure
    """
    if not data:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color=THEME["text_secondary"])
        )
        fig.update_layout(**COMMON_LAYOUT, height=300, title=title)
        return fig
    
    x_values = [d.get(x_field, i) for i, d in enumerate(data)]
    y_values = [d.get(y_field, 0) for d in data]
    names = [d.get("name", "") for d in data]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=x_values,
        y=y_values,
        mode="lines+markers",
        name=y_field,
        line=dict(color=THEME["blue"], width=2),
        marker=dict(size=6, color=THEME["blue"]),
        hovertemplate="%{text}<br>%{y:.1f}%<extra></extra>",
        text=names,
    ))
    
    # Add threshold lines
    fig.add_hline(y=80, line_dash="dash", line_color=THEME["green"], 
                  annotation_text="Target (80%)")
    fig.add_hline(y=50, line_dash="dash", line_color=THEME["yellow"],
                  annotation_text="Warning (50%)")
    
    fig.update_layout(
        **COMMON_LAYOUT,
        title=title,
        height=300,
        xaxis_title="Time",
        yaxis_title="Utilization (%)",
        yaxis=dict(range=[0, 105], **COMMON_LAYOUT["yaxis"]),
        showlegend=False,
    )
    
    return fig


def create_memory_chart(
    data: List[Dict[str, Any]],
    title: str = "Memory Usage"
) -> go.Figure:
    """
    Create a memory usage chart over time.
    
    Args:
        data: List of memory events with timestamp and memory_bytes
        title: Chart title
    
    Returns:
        Plotly figure
    """
    if not data:
        fig = go.Figure()
        fig.add_annotation(
            text="No memory data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color=THEME["text_secondary"])
        )
        fig.update_layout(**COMMON_LAYOUT, height=300, title=title)
        return fig
    
    timestamps = [d["timestamp"] for d in data]
    memory_gb = [d["memory_bytes"] / 1e9 for d in data]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=memory_gb,
        fill="tozeroy",
        line=dict(color=THEME["purple"], width=2),
        fillcolor="rgba(163, 113, 247, 0.2)",
        name="Memory (GB)",
    ))
    
    # Mark peak
    peak_idx = memory_gb.index(max(memory_gb))
    fig.add_annotation(
        x=timestamps[peak_idx],
        y=memory_gb[peak_idx],
        text=f"Peak: {memory_gb[peak_idx]:.2f} GB",
        showarrow=True,
        arrowhead=2,
        arrowcolor=THEME["purple"],
        font=dict(color=THEME["text"]),
    )
    
    fig.update_layout(
        **COMMON_LAYOUT,
        title=title,
        height=300,
        xaxis_title="Time",
        yaxis_title="Memory (GB)",
        showlegend=False,
    )
    
    return fig


def create_heatmap(
    data: List[Dict[str, Any]],
    x_field: str = "n",
    y_field: str = "m",
    z_field: str = "utilization",
    title: str = "Efficiency Heatmap"
) -> go.Figure:
    """
    Create a heatmap showing efficiency by dimensions.
    
    Args:
        data: List of data points with x, y, and z values
        x_field: Field name for x-axis
        y_field: Field name for y-axis
        z_field: Field name for color (z)
        title: Chart title
    
    Returns:
        Plotly figure
    """
    if not data:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color=THEME["text_secondary"])
        )
        fig.update_layout(**COMMON_LAYOUT, height=400, title=title)
        return fig
    
    x_values = [d.get(x_field, 0) for d in data]
    y_values = [d.get(y_field, 0) for d in data]
    z_values = [d.get(z_field, 0) for d in data]
    names = [d.get("name", "") for d in data]
    
    fig = go.Figure(go.Scatter(
        x=x_values,
        y=y_values,
        mode="markers",
        marker=dict(
            size=12,
            color=z_values,
            colorscale=[
                [0, THEME["red"]],
                [0.5, THEME["yellow"]],
                [1, THEME["green"]],
            ],
            colorbar=dict(
                title=z_field.title(),
                ticksuffix="%",
            ),
            showscale=True,
        ),
        text=[f"{name}<br>M={y}, N={x}<br>Util: {z:.1f}%" 
              for name, x, y, z in zip(names, x_values, y_values, z_values)],
        hovertemplate="%{text}<extra></extra>",
    ))
    
    # Add 128 boundary lines
    fig.add_vline(x=128, line_dash="dash", line_color=THEME["text_secondary"],
                  annotation_text="N=128")
    fig.add_hline(y=128, line_dash="dash", line_color=THEME["text_secondary"],
                  annotation_text="M=128")
    
    fig.update_layout(
        **COMMON_LAYOUT,
        title=title,
        height=400,
        xaxis_title="N (output columns)",
        yaxis_title="M (output rows)",
    )
    
    return fig


def create_pie_chart(
    labels: List[str],
    values: List[float],
    colors: Optional[List[str]] = None,
    title: str = "Distribution"
) -> go.Figure:
    """
    Create a pie/donut chart.
    
    Args:
        labels: Segment labels
        values: Segment values
        colors: Optional custom colors
        title: Chart title
    
    Returns:
        Plotly figure
    """
    if colors is None:
        colors = [THEME["blue"], THEME["green"], THEME["yellow"], 
                  THEME["red"], THEME["purple"]]
    
    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.5,
        marker=dict(colors=colors),
        textinfo="label+percent",
        textfont=dict(color=THEME["text"]),
        hovertemplate="%{label}: %{value}<extra></extra>",
    ))
    
    fig.update_layout(
        **COMMON_LAYOUT,
        title=title,
        height=300,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5,
            font=dict(color=THEME["text"]),
        ),
    )
    
    return fig


def create_bar_chart(
    labels: List[str],
    values: List[float],
    colors: Optional[List[str]] = None,
    title: str = "Comparison",
    horizontal: bool = False
) -> go.Figure:
    """
    Create a bar chart.
    
    Args:
        labels: Bar labels
        values: Bar values
        colors: Optional custom colors per bar
        title: Chart title
        horizontal: Whether to use horizontal bars
    
    Returns:
        Plotly figure
    """
    if colors is None:
        colors = [THEME["blue"]] * len(values)
    
    orientation = "h" if horizontal else "v"
    
    if horizontal:
        fig = go.Figure(go.Bar(
            x=values,
            y=labels,
            orientation=orientation,
            marker=dict(color=colors),
            text=[f"{v:.1f}" for v in values],
            textposition="outside",
            textfont=dict(color=THEME["text"]),
        ))
    else:
        fig = go.Figure(go.Bar(
            x=labels,
            y=values,
            orientation=orientation,
            marker=dict(color=colors),
            text=[f"{v:.1f}" for v in values],
            textposition="outside",
            textfont=dict(color=THEME["text"]),
        ))
    
    fig.update_layout(
        **COMMON_LAYOUT,
        title=title,
        height=300,
        showlegend=False,
    )
    
    return fig


def create_compilation_timeline(
    data: List[Dict[str, Any]],
    title: str = "Compilation Events"
) -> go.Figure:
    """
    Create a compilation timeline showing cache hits/misses.
    
    Args:
        data: List of compilation events
        title: Chart title
    
    Returns:
        Plotly figure
    """
    if not data:
        fig = go.Figure()
        fig.add_annotation(
            text="No compilation data",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color=THEME["text_secondary"])
        )
        fig.update_layout(**COMMON_LAYOUT, height=300, title=title)
        return fig
    
    # Separate hits and misses
    hits = [d for d in data if d.get("cache_hit", False)]
    misses = [d for d in data if not d.get("cache_hit", False)]
    
    fig = go.Figure()
    
    # Cache hits (small markers)
    if hits:
        fig.add_trace(go.Scatter(
            x=[h["timestamp"] for h in hits],
            y=[0.2] * len(hits),
            mode="markers",
            name="Cache Hit",
            marker=dict(size=8, color=THEME["green"], symbol="circle"),
            text=[h.get("function", "") for h in hits],
            hovertemplate="%{text}<br>Cache Hit<extra></extra>",
        ))
    
    # Cache misses (larger markers with compilation time)
    if misses:
        sizes = [max(8, min(30, m.get("compilation_time_ms", 10) / 10)) for m in misses]
        fig.add_trace(go.Scatter(
            x=[m["timestamp"] for m in misses],
            y=[0.8] * len(misses),
            mode="markers",
            name="Recompilation",
            marker=dict(size=sizes, color=THEME["red"], symbol="square"),
            text=[f"{m.get('function', '')}<br>{m.get('compilation_time_ms', 0):.1f}ms" 
                  for m in misses],
            hovertemplate="%{text}<extra></extra>",
        ))
    
    fig.update_layout(
        **COMMON_LAYOUT,
        title=title,
        height=200,
        xaxis_title="Time",
        yaxis=dict(
            visible=False,
            range=[0, 1],
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )
    
    return fig


def create_roofline_chart(
    operations: List[Dict[str, Any]],
    peak_compute_tflops: float = 275,
    peak_bandwidth_gbps: float = 1200,
    title: str = "Roofline Model"
) -> go.Figure:
    """
    Create a roofline model chart.
    
    The roofline model visualizes whether operations are
    compute-bound or memory-bound.
    
    Args:
        operations: List of operations with flops, bytes, duration
        peak_compute_tflops: Peak compute throughput (TFLOPS)
        peak_bandwidth_gbps: Peak memory bandwidth (GB/s)
        title: Chart title
    
    Returns:
        Plotly figure
    """
    # Calculate balance point (FLOPS/byte where lines meet)
    balance_point = peak_compute_tflops * 1e12 / (peak_bandwidth_gbps * 1e9)
    
    # Create roofline
    x_roof = [0.1, balance_point, 1000]
    y_roof = [
        0.1 * peak_bandwidth_gbps * 1e9 / 1e12,  # Memory-bound region
        peak_compute_tflops,
        peak_compute_tflops,  # Compute-bound region
    ]
    
    fig = go.Figure()
    
    # Roofline
    fig.add_trace(go.Scatter(
        x=x_roof,
        y=y_roof,
        mode="lines",
        name="Roofline",
        line=dict(color=THEME["yellow"], width=3),
    ))
    
    # Add operations
    if operations:
        x_ops = []
        y_ops = []
        texts = []
        
        for op in operations:
            flops = op.get("flops", 0)
            bytes_accessed = op.get("bytes_accessed", 1)
            duration_s = op.get("duration_ns", 1e9) / 1e9
            
            if bytes_accessed > 0 and duration_s > 0:
                intensity = flops / bytes_accessed
                achieved_tflops = flops / duration_s / 1e12
                
                x_ops.append(intensity)
                y_ops.append(achieved_tflops)
                texts.append(f"{op.get('name', 'op')}<br>"
                           f"Intensity: {intensity:.1f} FLOPS/byte<br>"
                           f"Achieved: {achieved_tflops:.2f} TFLOPS")
        
        if x_ops:
            fig.add_trace(go.Scatter(
                x=x_ops,
                y=y_ops,
                mode="markers",
                name="Operations",
                marker=dict(size=10, color=THEME["blue"]),
                text=texts,
                hovertemplate="%{text}<extra></extra>",
            ))
    
    # Add balance point annotation
    fig.add_vline(x=balance_point, line_dash="dash", line_color=THEME["text_secondary"])
    fig.add_annotation(
        x=balance_point, y=peak_compute_tflops * 0.5,
        text=f"Balance: {balance_point:.0f} FLOPS/byte",
        showarrow=False,
        font=dict(size=10, color=THEME["text_secondary"]),
    )
    
    fig.update_layout(
        **COMMON_LAYOUT,
        title=title,
        height=400,
        xaxis_title="Arithmetic Intensity (FLOPS/byte)",
        yaxis_title="Performance (TFLOPS)",
        xaxis_type="log",
        yaxis_type="log",
    )
    
    return fig


def create_operations_breakdown(
    operations: List[Dict[str, Any]],
    title: str = "Operations by Type"
) -> go.Figure:
    """
    Create a breakdown chart of operations by type.
    
    Args:
        operations: List of operations with type and duration
        title: Chart title
    
    Returns:
        Plotly figure
    """
    # Group by type
    type_times: Dict[str, float] = {}
    type_counts: Dict[str, int] = {}
    
    for op in operations:
        op_type = op.get("type", "other")
        duration = op.get("duration_ns", 0)
        
        type_times[op_type] = type_times.get(op_type, 0) + duration
        type_counts[op_type] = type_counts.get(op_type, 0) + 1
    
    # Sort by time
    sorted_types = sorted(type_times.items(), key=lambda x: x[1], reverse=True)
    
    labels = [t[0] for t in sorted_types]
    values = [t[1] / 1e6 for t in sorted_types]  # Convert to ms
    counts = [type_counts[t[0]] for t in sorted_types]
    
    # Color by type
    type_colors = {
        "matmul": THEME["blue"],
        "convolution": THEME["purple"],
        "reduce": THEME["green"],
        "elementwise": THEME["yellow"],
        "transpose": THEME["red"],
        "other": THEME["text_secondary"],
    }
    colors = [type_colors.get(l, THEME["text_secondary"]) for l in labels]
    
    fig = go.Figure(go.Bar(
        x=labels,
        y=values,
        marker=dict(color=colors),
        text=[f"{v:.1f}ms ({c})" for v, c in zip(values, counts)],
        textposition="outside",
        textfont=dict(color=THEME["text"], size=10),
    ))
    
    fig.update_layout(
        **COMMON_LAYOUT,
        title=title,
        height=300,
        xaxis_title="Operation Type",
        yaxis_title="Time (ms)",
    )
    
    return fig

