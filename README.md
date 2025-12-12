# TPUsight

**A comprehensive TPU profiler inspired by NVIDIA Nsight**

TPUsight provides deep visibility into TPU workloads with actionable optimization insights, all viewable through an interactive Jupyter GUI or real-time live dashboard.

![TPUsight Dashboard](https://img.shields.io/badge/TPUsight-v0.1.0-blue)
![Python](https://img.shields.io/badge/python-3.9+-green)
![JAX](https://img.shields.io/badge/JAX-0.4.20+-orange)

## Features

### Systolic Array Utilization Estimator
- Real-time MXU (Matrix Multiply Unit) utilization tracking
- Operation-level efficiency breakdown
- Identifies underutilized matrix operations

### Padding/Tiling Inefficiency Analyzer
- Detects suboptimal tensor shapes
- Calculates wasted compute from padding
- Suggests optimal tile sizes for TPU architecture

### Fusion Failure Explanations
- Explains why operations couldn't be fused
- Identifies fusion-blocking patterns
- Suggests code changes to enable fusion

### Dynamic Shape + Executable Cache Profiler
- Tracks JIT recompilation events
- Monitors executable cache hit/miss rates
- Identifies dynamic shape bottlenecks

### Memory Traffic + Layout Diagnostics
- HBM bandwidth utilization
- Tensor layout efficiency analysis
- Memory bottleneck detection

### Time Breakdown Analysis
- Compute vs memory wait time breakdown
- Rematerialization overhead tracking
- Compilation time analysis
- Roofline model visualization

### Live Profiling Mode
- Real-time metrics streaming
- Live alerts for inefficiencies
- Auto-updating dashboards
- Custom alert thresholds

### TPU Doctor (Optimization Suggestions)
- Actionable recommendations ranked by impact
- Code snippets for fixes
- Performance impact estimates

## Quick Start

### Installation

```bash
pip install -e .
```

For TPU support:
```bash
pip install -e ".[tpu]"
```

### Basic Usage

```python
import jax.numpy as jnp
from tpusight import TPUsight

# Create profiler instance
profiler = TPUsight()

# Profile a function
@profiler.trace
def my_model(x, w):
    return jnp.dot(x, w)

# Run profiled code
x = jnp.ones((1024, 512))
w = jnp.ones((512, 256))
result = my_model(x, w)

# View summary
profiler.summary()

# Export HTML report
profiler.export("report.html", format="html")
```

### Live Profiling

```python
from tpusight import LiveProfiler
from tpusight.visualization.live_dashboard import SimpleLiveDashboard

live = LiveProfiler()

@live.on_alert
def handle_alert(alert):
    print(f"[{alert.severity}] {alert.message}")

@live.trace
def my_function(x, w):
    return jnp.dot(x, w)

# Start live monitoring
live.start()
dashboard = SimpleLiveDashboard(live)
dashboard.start()

# Run your workload - dashboard updates in real-time
for batch in data:
    result = my_function(batch, weights)

# Stop when done
dashboard.stop()
live.stop()
```

### Time Breakdown

```python
# See where time is spent
profiler.time_breakdown.print_breakdown()

# Output:
# ============================================================
#   TPUsight Time Breakdown
# ============================================================
#   Total Time: 125.32 ms
# ------------------------------------------------------------
#   Compute            85.20 ms   68.0%
#   Memory Wait        25.50 ms   20.3%
#   Compilation        10.12 ms    8.1%
#   Rematerialization   4.50 ms    3.6%
# ------------------------------------------------------------
#   Bottleneck: Your workload is compute-bound (good for TPU!)
# ============================================================
```

## Dashboard Overview

The TPUsight dashboard provides eight main views:

| View | Description |
|------|-------------|
| **Overview** | Health score and summary metrics |
| **Systolic** | MXU utilization heatmaps and timelines |
| **Padding** | Tensor shape efficiency analysis |
| **Fusion** | Operation fusion success/failure breakdown |
| **Cache** | JIT compilation and cache statistics |
| **Memory** | HBM bandwidth and layout diagnostics |
| **Time** | Compute/memory/compilation breakdown |
| **Doctor** | Prioritized optimization recommendations |

## Architecture

```
tpusight/
├── core/           # Core profiling infrastructure
│   ├── profiler.py       # Main TPUsight class
│   ├── live_profiler.py  # Live profiling mode
│   ├── jax_tracer.py     # JAX tracing utilities
│   └── data_collector.py # Data structures
├── analyzers/      # Analysis modules
│   ├── systolic.py       # MXU utilization
│   ├── padding.py        # Shape efficiency
│   ├── fusion.py         # Fusion analysis
│   ├── cache.py          # Cache profiling
│   ├── memory.py         # Memory analysis
│   ├── time_breakdown.py # Time breakdown
│   └── doctor.py         # Recommendations
├── visualization/  # Jupyter widgets and charts
│   ├── dashboard.py      # Main dashboard
│   ├── live_dashboard.py # Live dashboard
│   ├── widgets.py        # UI components
│   └── charts.py         # Plotly charts
└── utils/          # Helper utilities
```

## Configuration

```python
# Post-hoc profiler
profiler = TPUsight(
    collect_hlo=True,          # Collect HLO IR
    collect_memory=True,       # Track memory allocations
    cache_analysis=True,       # Monitor executable cache
    sample_rate=1.0,           # Sampling rate (1.0 = all ops)
)

# Live profiler with custom thresholds
live = LiveProfiler(
    alert_thresholds={
        "mxu_utilization_low": 30.0,      # Critical if MXU < 30%
        "mxu_utilization_warning": 50.0,  # Warning if MXU < 50%
        "padding_waste_high": 30.0,       # Alert if padding > 30%
        "compilation_time_high": 5000.0,  # Alert if compile > 5s
    },
    update_interval=0.5,  # Metrics update frequency
)
```

## Output Formats

```python
# Quick summary to console
profiler.summary()

# Full text report
print(profiler.report())

# Export to HTML (works in any browser)
profiler.export("report.html", format="html")

# Export to JSON (for programmatic access)
profiler.export("results.json", format="json")

# Export to CSV
profiler.export("operations.csv", format="csv")
```

## Documentation

See the `notebooks/demo.ipynb` for detailed examples and tutorials.

## Contributing

Contributions welcome! Please read our contributing guidelines.

## License

MIT License - see LICENSE file for details.
