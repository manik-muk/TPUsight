"""Main dashboard for TPUsight visualization."""

from typing import Dict, List, Any, Optional, TYPE_CHECKING
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output

if TYPE_CHECKING:
    from tpusight.core.profiler import TPUsight

from tpusight.visualization.widgets import (
    get_base_styles,
    MetricCard,
    AnalysisPanel,
    create_tabs,
    create_issue_card,
    create_health_score_widget,
    create_data_table,
    format_value_with_color,
    COLORS,
)
from tpusight.visualization.charts import (
    create_utilization_chart,
    create_timeline_chart,
    create_memory_chart,
    create_heatmap,
    create_pie_chart,
    create_bar_chart,
    create_compilation_timeline,
    create_roofline_chart,
    create_operations_breakdown,
)
from tpusight.utils.helpers import format_bytes, format_flops, format_duration


class Dashboard:
    """
    Interactive dashboard for TPUsight profiling results.
    
    Displays profiling data across multiple tabs:
    - Overview: Summary metrics and health score
    - Systolic: MXU utilization analysis
    - Padding: Shape efficiency analysis  
    - Fusion: Operation fusion insights
    - Cache: JIT compilation analysis
    - Memory: Memory traffic analysis
    - Doctor: Optimization recommendations
    """
    
    def __init__(self, profiler: "TPUsight"):
        self.profiler = profiler
        self._widgets: Dict[str, widgets.Widget] = {}
    
    def display(self, height: int = 800) -> widgets.Widget:
        """
        Display the interactive dashboard.
        
        Args:
            height: Dashboard height in pixels
            
        Returns:
            The dashboard widget
        """
        # Inject styles
        display(HTML(get_base_styles()))
        
        # Create main container
        main = widgets.VBox(
            layout=widgets.Layout(
                width="100%",
                min_height=f"{height}px",
            )
        )
        
        # Header
        header = self._create_header()
        
        # Tabs
        tabs = self._create_tabs()
        
        main.children = [header, tabs]
        
        display(main)
        return main
    
    def _create_header(self) -> widgets.Widget:
        """Create the dashboard header."""
        summary = self.profiler.get_summary()
        
        return widgets.HTML(f"""
        <div class="tpusight-container">
            <div class="tpusight-header">
                <span class="tpusight-logo">TPUsight</span>
                <span style="color: {COLORS['text_secondary']};">
                    Session: {summary['session_id']} | 
                    Device: {summary['device']['type']} x{summary['device']['count']} |
                    Duration: {summary['duration_seconds']:.2f}s
                </span>
            </div>
        </div>
        """)
    
    def _create_tabs(self) -> widgets.Tab:
        """Create the main tab interface."""
        tab_names = [
            "🎯 Overview",
            "⚡ Systolic",
            "📐 Padding",
            "🔗 Fusion",
            "💾 Cache",
            "🧠 Memory",
            "🩺 Doctor",
        ]
        
        tab_contents = [
            self._create_overview_tab(),
            self._create_systolic_tab(),
            self._create_padding_tab(),
            self._create_fusion_tab(),
            self._create_cache_tab(),
            self._create_memory_tab(),
            self._create_doctor_tab(),
        ]
        
        return create_tabs(tab_names, tab_contents)
    
    def _create_overview_tab(self) -> widgets.Widget:
        """Create the overview tab content."""
        summary = self.profiler.get_summary()
        doctor_diagnosis = self.profiler.doctor.diagnose()
        
        # Health score
        health_widget = create_health_score_widget(
            doctor_diagnosis["health_score"],
            doctor_diagnosis["health_status"]
        )
        
        # Key metrics row
        ops = summary["operations"]
        comp = summary["compilation"]
        mem = summary["memory"]
        
        metrics_row = widgets.HBox([
            widgets.VBox([
                MetricCard(
                    "Total Operations",
                    f"{ops['total']:,}",
                    subtitle=f"{format_flops(ops['total_flops'])} total",
                ),
            ]),
            widgets.VBox([
                MetricCard(
                    "Total Time",
                    format_duration(ops['total_time_ns'] / 1e9),
                    subtitle=f"Across all operations",
                ),
            ]),
            widgets.VBox([
                MetricCard(
                    "Cache Hit Rate",
                    f"{comp['cache_hit_rate'] * 100:.1f}%",
                    progress=comp['cache_hit_rate'] * 100,
                    progress_color=COLORS['accent_green'] if comp['cache_hit_rate'] > 0.8 else COLORS['accent_yellow'],
                ),
            ]),
            widgets.VBox([
                MetricCard(
                    "Peak Memory",
                    format_bytes(mem['peak_bytes']),
                    subtitle=f"{mem['total_allocations']} allocations",
                ),
            ]),
        ], layout=widgets.Layout(width="100%", justify_content="space-around"))
        
        # Operations breakdown chart
        ops_data = [
            {
                "type": op.op_type.value,
                "duration_ns": op.duration_ns,
            }
            for op in self.profiler.profile_data.operations
        ]
        
        ops_chart = widgets.Output()
        with ops_chart:
            fig = create_operations_breakdown(ops_data)
            fig.show()
        
        # Top issues
        issues_html = widgets.HTML(f"""
        <div class="tpusight-card">
            <div class="tpusight-card-title">Top Issues</div>
        </div>
        """)
        
        issue_cards = widgets.VBox([
            create_issue_card(issue) 
            for issue in doctor_diagnosis["top_recommendations"][:3]
        ])
        
        return widgets.VBox([
            health_widget,
            widgets.HTML("<div style='height: 20px;'></div>"),
            metrics_row,
            widgets.HTML("<div style='height: 20px;'></div>"),
            widgets.HBox([
                widgets.VBox([ops_chart], layout=widgets.Layout(width="50%")),
                widgets.VBox([issues_html, issue_cards], layout=widgets.Layout(width="50%")),
            ]),
        ])
    
    def _create_systolic_tab(self) -> widgets.Widget:
        """Create the systolic array analysis tab."""
        analysis = self.profiler.systolic.analyze()
        
        if analysis["status"] == "no_data":
            return widgets.HTML(f"""
            <div class="tpusight-card">
                <div style="text-align: center; padding: 40px; color: {COLORS['text_secondary']};">
                    No matrix multiplication operations found to analyze.
                </div>
            </div>
            """)
        
        metrics = analysis["metrics"]
        
        # Utilization gauge
        gauge_output = widgets.Output()
        with gauge_output:
            fig = create_utilization_chart(
                {"value": metrics.overall_utilization},
                title="MXU Utilization"
            )
            fig.show()
        
        # Efficiency distribution
        buckets = metrics.efficiency_buckets
        pie_output = widgets.Output()
        with pie_output:
            fig = create_pie_chart(
                labels=list(buckets.keys()),
                values=list(buckets.values()),
                colors=[COLORS["accent_green"], COLORS["accent_blue"], 
                       COLORS["accent_yellow"], COLORS["accent_yellow"], COLORS["accent_red"]],
                title="Efficiency Distribution"
            )
            fig.show()
        
        # Timeline
        timeline_data = self.profiler.systolic.get_utilization_timeline()
        timeline_output = widgets.Output()
        with timeline_output:
            fig = create_timeline_chart(
                timeline_data,
                title="MXU Utilization Over Time"
            )
            fig.show()
        
        # Heatmap
        heatmap_data = self.profiler.systolic.get_efficiency_heatmap_data()
        heatmap_output = widgets.Output()
        with heatmap_output:
            fig = create_heatmap(
                heatmap_data.get("data", []),
                title="Efficiency by Matrix Dimensions"
            )
            fig.show()
        
        # Low utilization operations table
        low_util_ops = metrics.low_util_operations
        if low_util_ops:
            table_data = [
                [
                    op["name"][:30],
                    f"{op['utilization']:.1f}%",
                    str(op["input_shapes"]),
                    op.get("recommendation", "-")[:50],
                ]
                for op in low_util_ops[:10]
            ]
            table = create_data_table(
                ["Operation", "Utilization", "Shapes", "Recommendation"],
                table_data
            )
        else:
            table = widgets.HTML(f"""
            <div style="color: {COLORS['accent_green']}; padding: 20px; text-align: center;">
                ✓ All operations have good MXU utilization
            </div>
            """)
        
        # Recommendations
        rec_cards = widgets.VBox([
            create_issue_card({
                "severity": rec["severity"],
                "title": "MXU Recommendation",
                "message": rec["message"],
                "suggestion": rec["suggestion"],
                "impact_estimate": rec.get("impact", ""),
            })
            for rec in analysis["recommendations"]
        ])
        
        return widgets.VBox([
            widgets.HBox([
                widgets.VBox([gauge_output], layout=widgets.Layout(width="30%")),
                widgets.VBox([pie_output], layout=widgets.Layout(width="30%")),
                widgets.VBox([
                    MetricCard("Total MatMul Ops", str(metrics.total_matmul_ops)),
                    MetricCard("Low Efficiency Ops", str(metrics.low_util_ops),
                              subtitle=f"{format_flops(metrics.wasted_flops)} wasted"),
                ], layout=widgets.Layout(width="40%")),
            ]),
            timeline_output,
            widgets.HBox([
                widgets.VBox([heatmap_output], layout=widgets.Layout(width="50%")),
                widgets.VBox([
                    widgets.HTML('<div class="tpusight-card-title">Low Utilization Operations</div>'),
                    table,
                ], layout=widgets.Layout(width="50%")),
            ]),
            widgets.HTML('<div class="tpusight-card-title">Recommendations</div>'),
            rec_cards,
        ])
    
    def _create_padding_tab(self) -> widgets.Widget:
        """Create the padding analysis tab."""
        analysis = self.profiler.padding.analyze()
        
        if analysis["status"] == "no_data":
            return widgets.HTML(f"""
            <div class="tpusight-card">
                <div style="text-align: center; padding: 40px; color: {COLORS['text_secondary']};">
                    No operations found to analyze for padding efficiency.
                </div>
            </div>
            """)
        
        metrics = analysis["metrics"]
        vis_data = self.profiler.padding.get_visualization_data()
        
        # Pie chart of efficiency buckets
        pie_output = widgets.Output()
        with pie_output:
            pie_data = vis_data.get("pie_chart", {})
            fig = create_pie_chart(
                labels=pie_data.get("labels", []),
                values=pie_data.get("values", []),
                colors=pie_data.get("colors", []),
                title="Shape Efficiency Distribution"
            )
            fig.show()
        
        # Metrics
        metrics_col = widgets.VBox([
            MetricCard(
                "Average Waste",
                f"{metrics.total_wasted_compute_pct:.1f}%",
                progress=100 - metrics.total_wasted_compute_pct,
                progress_color=COLORS["accent_green"] if metrics.total_wasted_compute_pct < 10 else COLORS["accent_yellow"],
            ),
            MetricCard("Critical Shapes", str(metrics.critical_ops), 
                      subtitle=">30% waste"),
            MetricCard("Warning Shapes", str(metrics.warning_ops),
                      subtitle="10-30% waste"),
        ])
        
        # Shape efficiency table
        shape_table = self.profiler.padding.get_shape_efficiency_table()
        if shape_table:
            table_data = [
                [
                    row["operation"][:25],
                    row["shape"],
                    f"{row['waste_pct']:.1f}%",
                    "✓" if row["optimal"] else "✗",
                ]
                for row in shape_table[:15]
            ]
            table = create_data_table(
                ["Operation", "Shape", "Waste %", "Optimal"],
                table_data
            )
        else:
            table = widgets.HTML("No shape data available")
        
        # Recommendations
        rec_cards = widgets.VBox([
            create_issue_card({
                "severity": rec["severity"],
                "title": rec.get("category", "Padding"),
                "message": rec["message"],
                "suggestion": rec["suggestion"],
                "impact_estimate": rec.get("impact", ""),
            })
            for rec in analysis["recommendations"]
        ])
        
        return widgets.VBox([
            widgets.HBox([
                widgets.VBox([pie_output], layout=widgets.Layout(width="50%")),
                metrics_col,
            ]),
            widgets.HTML('<div class="tpusight-card-title">Shape Efficiency Details</div>'),
            table,
            widgets.HTML('<div class="tpusight-card-title">Recommendations</div>'),
            rec_cards,
        ])
    
    def _create_fusion_tab(self) -> widgets.Widget:
        """Create the fusion analysis tab."""
        analysis = self.profiler.fusion.analyze()
        
        if analysis["status"] == "no_data":
            return widgets.HTML(f"""
            <div class="tpusight-card">
                <div style="text-align: center; padding: 40px; color: {COLORS['text_secondary']};">
                    No operations found to analyze for fusion.
                </div>
            </div>
            """)
        
        metrics = analysis["metrics"]
        
        # Fusion rate gauge
        gauge_output = widgets.Output()
        with gauge_output:
            fig = create_utilization_chart(
                {"value": metrics.fusion_rate},
                title="Fusion Rate"
            )
            fig.show()
        
        # Metrics
        metrics_col = widgets.VBox([
            MetricCard("Fusion Groups", str(metrics.num_fusion_groups)),
            MetricCard("Avg Group Size", f"{metrics.average_fusion_size:.1f}"),
            MetricCard("Largest Group", str(metrics.largest_fusion_group)),
        ])
        
        # Fusion blockers
        blocker_labels = []
        blocker_values = []
        for blocker, count in metrics.blocker_counts.items():
            if count > 0:
                blocker_labels.append(blocker.value)
                blocker_values.append(count)
        
        blockers_output = widgets.Output()
        with blockers_output:
            if blocker_labels:
                fig = create_bar_chart(
                    blocker_labels,
                    blocker_values,
                    title="Fusion Blockers by Type"
                )
                fig.show()
        
        # Missed opportunities
        opportunities = metrics.missed_opportunities
        if opportunities:
            opp_cards = widgets.VBox([
                widgets.HTML(f"""
                <div class="tpusight-issue tpusight-issue-info">
                    <div style="font-weight: 600;">Potential Fusion: {len(op.operations)} ops</div>
                    <div style="margin-top: 4px;">{op.explanation}</div>
                    <div style="margin-top: 8px; color: {COLORS['text_secondary']};">
                        <strong>Suggested fix:</strong> {op.fix_suggestion or 'Review operations'}
                    </div>
                    <div class="tpusight-badge" style="background: {COLORS['accent_blue']}; margin-top: 8px;">
                        ~{(op.estimated_speedup - 1) * 100:.0f}% speedup
                    </div>
                </div>
                """)
                for op in opportunities[:5]
            ])
        else:
            opp_cards = widgets.HTML(f"""
            <div style="color: {COLORS['accent_green']}; padding: 20px;">
                ✓ No obvious missed fusion opportunities detected
            </div>
            """)
        
        # Recommendations
        rec_cards = widgets.VBox([
            create_issue_card({
                "severity": rec["severity"],
                "title": "Fusion",
                "message": rec["message"],
                "suggestion": rec["suggestion"],
                "impact_estimate": rec.get("estimated_speedup", ""),
            })
            for rec in analysis["recommendations"]
        ])
        
        return widgets.VBox([
            widgets.HBox([
                widgets.VBox([gauge_output], layout=widgets.Layout(width="30%")),
                metrics_col,
                widgets.VBox([blockers_output], layout=widgets.Layout(width="40%")),
            ]),
            widgets.HTML('<div class="tpusight-card-title">Missed Fusion Opportunities</div>'),
            opp_cards,
            widgets.HTML('<div class="tpusight-card-title">Recommendations</div>'),
            rec_cards,
        ])
    
    def _create_cache_tab(self) -> widgets.Widget:
        """Create the cache analysis tab."""
        analysis = self.profiler.cache.analyze()
        
        if analysis["status"] == "no_data":
            return widgets.HTML(f"""
            <div class="tpusight-card">
                <div style="text-align: center; padding: 40px; color: {COLORS['text_secondary']};">
                    No compilation events recorded.
                </div>
            </div>
            """)
        
        metrics = analysis["metrics"]
        vis_data = self.profiler.cache.get_visualization_data()
        
        # Cache hit rate pie
        pie_output = widgets.Output()
        with pie_output:
            pie_data = vis_data.get("cache_pie", {})
            fig = create_pie_chart(
                labels=pie_data.get("labels", []),
                values=pie_data.get("values", []),
                colors=pie_data.get("colors", []),
                title="Cache Performance"
            )
            fig.show()
        
        # Metrics
        metrics_col = widgets.VBox([
            MetricCard(
                "Cache Hit Rate",
                f"{metrics.cache_hit_rate:.1f}%",
                progress=metrics.cache_hit_rate,
                progress_color=COLORS["accent_green"] if metrics.cache_hit_rate > 80 else COLORS["accent_yellow"],
            ),
            MetricCard(
                "Total Compilation Time",
                f"{metrics.total_compilation_time_ms:.1f} ms",
            ),
            MetricCard("Shape Variations", str(metrics.unique_shapes)),
        ])
        
        # Compilation timeline
        timeline_output = widgets.Output()
        with timeline_output:
            fig = create_compilation_timeline(
                analysis.get("compilation_timeline", []),
                title="Compilation Events"
            )
            fig.show()
        
        # Hotspots table
        hotspots = metrics.hotspots
        if hotspots:
            table_data = [
                [
                    h["function"][:30],
                    str(h["cache_misses"]),
                    f"{h['miss_rate'] * 100:.1f}%",
                    f"{h['compilation_time_ms']:.1f} ms",
                    str(h["unique_shapes"]),
                ]
                for h in hotspots[:10]
            ]
            table = create_data_table(
                ["Function", "Recompiles", "Miss Rate", "Time", "Shapes"],
                table_data
            )
        else:
            table = widgets.HTML(f"""
            <div style="color: {COLORS['accent_green']}; padding: 20px;">
                ✓ No recompilation hotspots detected
            </div>
            """)
        
        # Recommendations
        rec_cards = widgets.VBox([
            create_issue_card({
                "severity": rec["severity"],
                "title": rec.get("category", "Cache"),
                "message": rec["message"],
                "suggestion": rec["suggestion"],
                "impact_estimate": rec.get("impact", ""),
                "code_example": rec.get("code_example"),
            })
            for rec in analysis["recommendations"]
        ])
        
        return widgets.VBox([
            widgets.HBox([
                widgets.VBox([pie_output], layout=widgets.Layout(width="40%")),
                metrics_col,
            ]),
            timeline_output,
            widgets.HTML('<div class="tpusight-card-title">Recompilation Hotspots</div>'),
            table,
            widgets.HTML('<div class="tpusight-card-title">Recommendations</div>'),
            rec_cards,
        ])
    
    def _create_memory_tab(self) -> widgets.Widget:
        """Create the memory analysis tab."""
        analysis = self.profiler.memory.analyze()
        
        if analysis["status"] == "no_data":
            return widgets.HTML(f"""
            <div class="tpusight-card">
                <div style="text-align: center; padding: 40px; color: {COLORS['text_secondary']};">
                    No memory data available.
                </div>
            </div>
            """)
        
        metrics = analysis["metrics"]
        vis_data = self.profiler.memory.get_visualization_data()
        
        # Memory timeline
        timeline_output = widgets.Output()
        with timeline_output:
            fig = create_memory_chart(
                vis_data.get("memory_timeline", []),
                title="Memory Usage Over Time"
            )
            fig.show()
        
        # Metrics
        metrics_row = widgets.HBox([
            MetricCard("Peak Memory", format_bytes(metrics.peak_memory_bytes)),
            MetricCard("HBM Utilization", f"{metrics.estimated_hbm_utilization:.1f}%",
                      progress=metrics.estimated_hbm_utilization),
            MetricCard("Layout Efficiency", f"{metrics.average_layout_efficiency * 100:.1f}%",
                      progress=metrics.average_layout_efficiency * 100),
            MetricCard("Memory-Bound Ops", str(len(metrics.memory_bound_operations))),
        ])
        
        # Roofline model
        roofline_output = widgets.Output()
        with roofline_output:
            ops_for_roofline = [
                {
                    "name": op.name,
                    "flops": op.flops or 0,
                    "bytes_accessed": op.bytes_accessed or 0,
                    "duration_ns": op.duration_ns,
                }
                for op in self.profiler.profile_data.operations
                if op.flops and op.bytes_accessed
            ]
            fig = create_roofline_chart(ops_for_roofline)
            fig.show()
        
        # Memory-bound operations table
        mem_bound = metrics.memory_bound_operations
        if mem_bound:
            table_data = [
                [
                    op["operation"][:25],
                    op["type"],
                    f"{op['arithmetic_intensity']:.1f}",
                    f"{op['achieved_bandwidth_gbps']:.1f} GB/s",
                ]
                for op in mem_bound[:10]
            ]
            table = create_data_table(
                ["Operation", "Type", "Arith. Intensity", "Bandwidth"],
                table_data
            )
        else:
            table = widgets.HTML(f"""
            <div style="color: {COLORS['accent_green']}; padding: 20px;">
                ✓ No memory-bound operations detected
            </div>
            """)
        
        # Recommendations
        rec_cards = widgets.VBox([
            create_issue_card({
                "severity": rec["severity"],
                "title": rec.get("category", "Memory"),
                "message": rec["message"],
                "suggestion": rec["suggestion"],
                "impact_estimate": rec.get("impact", ""),
                "code_example": rec.get("code_example"),
            })
            for rec in analysis["recommendations"]
        ])
        
        return widgets.VBox([
            metrics_row,
            timeline_output,
            widgets.HBox([
                widgets.VBox([roofline_output], layout=widgets.Layout(width="50%")),
                widgets.VBox([
                    widgets.HTML('<div class="tpusight-card-title">Memory-Bound Operations</div>'),
                    table,
                ], layout=widgets.Layout(width="50%")),
            ]),
            widgets.HTML('<div class="tpusight-card-title">Recommendations</div>'),
            rec_cards,
        ])
    
    def _create_doctor_tab(self) -> widgets.Widget:
        """Create the TPU Doctor tab with all recommendations."""
        diagnosis = self.profiler.doctor.diagnose()
        
        # Health overview
        health_widget = create_health_score_widget(
            diagnosis["health_score"],
            diagnosis["health_status"]
        )
        
        # Summary stats
        stats_row = widgets.HBox([
            MetricCard("Critical Issues", str(diagnosis["critical_count"]),
                      subtitle="Need immediate attention"),
            MetricCard("Warnings", str(diagnosis["warning_count"]),
                      subtitle="Recommended to fix"),
            MetricCard("Info", str(diagnosis["info_count"]),
                      subtitle="Nice to have"),
        ])
        
        # All issues grouped by severity
        def create_issue_section(title: str, issues: List[Dict], color: str) -> widgets.Widget:
            if not issues:
                return widgets.VBox([])
            
            header = widgets.HTML(f"""
            <div style="margin: 20px 0 10px 0; font-size: 16px; font-weight: 600; color: {color};">
                {title} ({len(issues)})
            </div>
            """)
            
            cards = widgets.VBox([
                create_issue_card(issue) for issue in issues
            ])
            
            return widgets.VBox([header, cards])
        
        critical_section = create_issue_section(
            "🔴 Critical Issues",
            diagnosis["issues"]["critical"],
            COLORS["accent_red"]
        )
        
        warning_section = create_issue_section(
            "🟡 Warnings",
            diagnosis["issues"]["warning"],
            COLORS["accent_yellow"]
        )
        
        info_section = create_issue_section(
            "🔵 Suggestions",
            diagnosis["issues"]["info"],
            COLORS["accent_blue"]
        )
        
        # Quick wins section
        quick_wins = self.profiler.doctor.get_quick_wins()
        quick_wins_section = widgets.VBox([])
        if quick_wins:
            quick_wins_header = widgets.HTML(f"""
            <div style="margin: 20px 0 10px 0; font-size: 16px; font-weight: 600; color: {COLORS['accent_green']};">
                ⚡ Quick Wins (Easy to implement)
            </div>
            """)
            quick_wins_cards = widgets.VBox([
                create_issue_card(qw) for qw in quick_wins[:3]
            ])
            quick_wins_section = widgets.VBox([quick_wins_header, quick_wins_cards])
        
        return widgets.VBox([
            health_widget,
            widgets.HTML("<div style='height: 20px;'></div>"),
            stats_row,
            widgets.HTML("<div style='height: 20px;'></div>"),
            quick_wins_section,
            critical_section,
            warning_section,
            info_section,
        ])

