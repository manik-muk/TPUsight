"""Jupyter widgets for TPUsight visualization."""

from typing import Dict, List, Any, Optional, Callable
import ipywidgets as widgets
from IPython.display import display, HTML

# Color scheme - Dark theme inspired by professional profilers
COLORS = {
    "bg_primary": "#0f1419",
    "bg_secondary": "#1a1f26",
    "bg_card": "#232a33",
    "text_primary": "#e6edf3",
    "text_secondary": "#8b949e",
    "accent_blue": "#58a6ff",
    "accent_green": "#3fb950",
    "accent_yellow": "#d29922",
    "accent_red": "#f85149",
    "accent_purple": "#a371f7",
    "border": "#30363d",
}

# Severity colors
SEVERITY_COLORS = {
    "critical": COLORS["accent_red"],
    "warning": COLORS["accent_yellow"],
    "info": COLORS["accent_blue"],
}


def get_base_styles() -> str:
    """Get base CSS styles for the dashboard."""
    return f"""
    <style>
        .tpusight-container {{
            font-family: 'SF Mono', 'Fira Code', 'JetBrains Mono', monospace;
            background: {COLORS['bg_primary']};
            color: {COLORS['text_primary']};
            padding: 20px;
            border-radius: 12px;
        }}
        
        .tpusight-header {{
            display: flex;
            align-items: center;
            gap: 16px;
            margin-bottom: 24px;
            padding-bottom: 16px;
            border-bottom: 1px solid {COLORS['border']};
        }}
        
        .tpusight-logo {{
            font-size: 28px;
            font-weight: 700;
            background: linear-gradient(135deg, {COLORS['accent_blue']}, {COLORS['accent_purple']});
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        
        .tpusight-card {{
            background: {COLORS['bg_card']};
            border: 1px solid {COLORS['border']};
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 16px;
        }}
        
        .tpusight-card-title {{
            font-size: 14px;
            font-weight: 600;
            color: {COLORS['text_secondary']};
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 12px;
        }}
        
        .tpusight-metric {{
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            margin-bottom: 8px;
        }}
        
        .tpusight-metric-value {{
            font-size: 24px;
            font-weight: 700;
            color: {COLORS['text_primary']};
        }}
        
        .tpusight-metric-label {{
            font-size: 12px;
            color: {COLORS['text_secondary']};
        }}
        
        .tpusight-progress {{
            height: 8px;
            background: {COLORS['bg_secondary']};
            border-radius: 4px;
            overflow: hidden;
            margin-top: 8px;
        }}
        
        .tpusight-progress-bar {{
            height: 100%;
            border-radius: 4px;
            transition: width 0.3s ease;
        }}
        
        .tpusight-issue {{
            padding: 12px;
            border-radius: 6px;
            margin-bottom: 8px;
            border-left: 3px solid;
        }}
        
        .tpusight-issue-critical {{
            background: rgba(248, 81, 73, 0.1);
            border-left-color: {COLORS['accent_red']};
        }}
        
        .tpusight-issue-warning {{
            background: rgba(210, 153, 34, 0.1);
            border-left-color: {COLORS['accent_yellow']};
        }}
        
        .tpusight-issue-info {{
            background: rgba(88, 166, 255, 0.1);
            border-left-color: {COLORS['accent_blue']};
        }}
        
        .tpusight-code {{
            background: {COLORS['bg_secondary']};
            border: 1px solid {COLORS['border']};
            border-radius: 6px;
            padding: 12px;
            font-size: 12px;
            overflow-x: auto;
            white-space: pre;
            margin-top: 8px;
        }}
        
        .tpusight-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }}
        
        .tpusight-table th {{
            text-align: left;
            padding: 8px 12px;
            background: {COLORS['bg_secondary']};
            color: {COLORS['text_secondary']};
            font-weight: 600;
            border-bottom: 1px solid {COLORS['border']};
        }}
        
        .tpusight-table td {{
            padding: 8px 12px;
            border-bottom: 1px solid {COLORS['border']};
        }}
        
        .tpusight-table tr:hover {{
            background: {COLORS['bg_secondary']};
        }}
        
        .tpusight-badge {{
            display: inline-block;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 600;
        }}
        
        .tpusight-tabs {{
            display: flex;
            gap: 4px;
            margin-bottom: 16px;
            border-bottom: 1px solid {COLORS['border']};
            padding-bottom: 4px;
        }}
        
        .tpusight-tab {{
            padding: 8px 16px;
            border-radius: 6px 6px 0 0;
            cursor: pointer;
            color: {COLORS['text_secondary']};
            transition: all 0.2s ease;
        }}
        
        .tpusight-tab:hover {{
            background: {COLORS['bg_secondary']};
            color: {COLORS['text_primary']};
        }}
        
        .tpusight-tab-active {{
            background: {COLORS['accent_blue']};
            color: white;
        }}
        
        .health-score {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 80px;
            height: 80px;
            border-radius: 50%;
            font-size: 28px;
            font-weight: 700;
        }}
        
        .health-excellent {{ background: linear-gradient(135deg, {COLORS['accent_green']}, #2ea043); }}
        .health-good {{ background: linear-gradient(135deg, {COLORS['accent_blue']}, #1f6feb); }}
        .health-fair {{ background: linear-gradient(135deg, {COLORS['accent_yellow']}, #bb8009); }}
        .health-poor {{ background: linear-gradient(135deg, {COLORS['accent_red']}, #cf222e); }}
    </style>
    """


class MetricCard(widgets.VBox):
    """A card widget displaying a single metric."""
    
    def __init__(
        self,
        title: str,
        value: str,
        subtitle: Optional[str] = None,
        progress: Optional[float] = None,
        progress_color: str = COLORS["accent_blue"],
        **kwargs
    ):
        super().__init__(**kwargs)
        
        progress_html = ""
        if progress is not None:
            progress_html = f"""
            <div class="tpusight-progress">
                <div class="tpusight-progress-bar" 
                     style="width: {min(100, progress)}%; 
                            background: {progress_color};">
                </div>
            </div>
            """
        
        subtitle_html = f'<div class="tpusight-metric-label">{subtitle}</div>' if subtitle else ""
        
        html = f"""
        <div class="tpusight-card">
            <div class="tpusight-card-title">{title}</div>
            <div class="tpusight-metric-value">{value}</div>
            {subtitle_html}
            {progress_html}
        </div>
        """
        
        self.children = [widgets.HTML(html)]


class AnalysisPanel(widgets.VBox):
    """A panel for displaying analysis results."""
    
    def __init__(
        self,
        title: str,
        content: widgets.Widget,
        collapsible: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        if collapsible:
            accordion = widgets.Accordion(children=[content])
            accordion.set_title(0, title)
            self.children = [accordion]
        else:
            header = widgets.HTML(f'<div class="tpusight-card-title">{title}</div>')
            self.children = [header, content]


def create_tabs(
    tab_names: List[str],
    tab_contents: List[widgets.Widget]
) -> widgets.Tab:
    """Create a tabbed interface."""
    tab = widgets.Tab(children=tab_contents)
    for i, name in enumerate(tab_names):
        tab.set_title(i, name)
    
    # Style the tab
    tab.layout = widgets.Layout(
        width="100%",
    )
    
    return tab


def create_issue_card(issue: Dict[str, Any]) -> widgets.HTML:
    """Create a card for displaying an optimization issue."""
    severity = issue.get("severity", "info")
    severity_class = f"tpusight-issue-{severity}"
    severity_color = SEVERITY_COLORS.get(severity, COLORS["accent_blue"])
    
    code_html = ""
    if issue.get("code_example"):
        code_html = f'<div class="tpusight-code">{issue["code_example"]}</div>'
    
    affected_ops = ""
    if issue.get("affected_operations"):
        ops = ", ".join(issue["affected_operations"][:3])
        affected_ops = f'<div style="margin-top: 8px; font-size: 11px; color: {COLORS["text_secondary"]};">Affected: {ops}</div>'
    
    return widgets.HTML(f"""
    <div class="tpusight-issue {severity_class}">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div style="font-weight: 600; color: {severity_color};">{issue.get("title", "Issue")}</div>
            <span class="tpusight-badge" style="background: {severity_color}; color: white;">
                {issue.get("impact_estimate", "")}
            </span>
        </div>
        <div style="margin-top: 8px; color: {COLORS['text_primary']};">{issue.get("message", "")}</div>
        <div style="margin-top: 8px; color: {COLORS['text_secondary']};">
            <strong>Suggestion:</strong> {issue.get("suggestion", "")}
        </div>
        {affected_ops}
        {code_html}
    </div>
    """)


def create_health_score_widget(score: int, status: str) -> widgets.HTML:
    """Create a health score display widget."""
    if score >= 90:
        health_class = "health-excellent"
    elif score >= 70:
        health_class = "health-good"
    elif score >= 50:
        health_class = "health-fair"
    else:
        health_class = "health-poor"
    
    status_text = {
        "excellent": "Excellent",
        "good": "Good",
        "fair": "Fair",
        "needs_attention": "Needs Attention",
    }.get(status, status.title())
    
    return widgets.HTML(f"""
    <div style="display: flex; align-items: center; gap: 20px;">
        <div class="health-score {health_class}">{score}</div>
        <div>
            <div style="font-size: 18px; font-weight: 600;">TPU Health Score</div>
            <div style="color: {COLORS['text_secondary']};">{status_text}</div>
        </div>
    </div>
    """)


def create_data_table(
    columns: List[str],
    data: List[List[Any]],
    max_rows: int = 20
) -> widgets.HTML:
    """Create a data table widget."""
    if not data:
        return widgets.HTML(f'<div style="color: {COLORS["text_secondary"]};">No data available</div>')
    
    headers = "".join(f"<th>{col}</th>" for col in columns)
    
    rows = ""
    for row in data[:max_rows]:
        cells = "".join(f"<td>{cell}</td>" for cell in row)
        rows += f"<tr>{cells}</tr>"
    
    if len(data) > max_rows:
        rows += f'<tr><td colspan="{len(columns)}" style="text-align: center; color: {COLORS["text_secondary"]};">... and {len(data) - max_rows} more rows</td></tr>'
    
    return widgets.HTML(f"""
    <table class="tpusight-table">
        <thead><tr>{headers}</tr></thead>
        <tbody>{rows}</tbody>
    </table>
    """)


def format_value_with_color(value: float, thresholds: Dict[str, float]) -> str:
    """Format a value with color based on thresholds."""
    good = thresholds.get("good", 80)
    warning = thresholds.get("warning", 50)
    
    if value >= good:
        color = COLORS["accent_green"]
    elif value >= warning:
        color = COLORS["accent_yellow"]
    else:
        color = COLORS["accent_red"]
    
    return f'<span style="color: {color}; font-weight: 600;">{value:.1f}%</span>'

