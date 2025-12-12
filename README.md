# TPUsight 🔬

**A comprehensive TPU profiler inspired by NVIDIA Nsight**

TPUsight provides deep visibility into TPU workloads with actionable optimization insights, all viewable through an interactive Jupyter GUI.

![TPUsight Dashboard](https://img.shields.io/badge/TPUsight-v0.1.0-blue)
![Python](https://img.shields.io/badge/python-3.9+-green)
![JAX](https://img.shields.io/badge/JAX-0.4.20+-orange)

## ✨ Features

### 🎯 Systolic Array Utilization Estimator
- Real-time MXU (Matrix Multiply Unit) utilization tracking
- Operation-level efficiency breakdown
- Identifies underutilized matrix operations

### 📐 Padding/Tiling Inefficiency Analyzer
- Detects suboptimal tensor shapes
- Calculates wasted compute from padding
- Suggests optimal tile sizes for TPU architecture

### 🔗 Fusion Failure Explanations
- Explains why operations couldn't be fused
- Identifies fusion-blocking patterns
- Suggests code changes to enable fusion

### ⚡ Dynamic Shape + Executable Cache Profiler
- Tracks JIT recompilation events
- Monitors executable cache hit/miss rates
- Identifies dynamic shape bottlenecks

### 🧠 Memory Traffic + Layout Diagnostics
- HBM bandwidth utilization
- Tensor layout efficiency analysis
- Memory bottleneck detection

### 🩺 TPU Doctor (Optimization Suggestions)
- Actionable recommendations ranked by impact
- Code snippets for fixes
- Performance impact estimates

## 🚀 Quick Start

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
import jax
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

# Launch interactive dashboard
profiler.dashboard()
```

### Jupyter Notebook

```python
from tpusight import TPUsight

profiler = TPUsight()

# Profile your JAX code
with profiler.trace_context():
    # Your JAX operations here
    pass

# Display interactive dashboard
profiler.dashboard()
```

## 📊 Dashboard Overview

The TPUsight dashboard provides six main views:

| View | Description |
|------|-------------|
| **Systolic** | MXU utilization heatmaps and timelines |
| **Padding** | Tensor shape efficiency analysis |
| **Fusion** | Operation fusion success/failure breakdown |
| **Cache** | JIT compilation and cache statistics |
| **Memory** | HBM bandwidth and layout diagnostics |
| **Doctor** | Prioritized optimization recommendations |

## 🏗️ Architecture

```
tpusight/
├── core/           # Core profiling infrastructure
├── analyzers/      # Analysis modules for each feature
├── visualization/  # Jupyter widgets and charts
└── utils/          # Helper utilities
```

## 🔧 Configuration

```python
profiler = TPUsight(
    collect_hlo=True,          # Collect HLO IR
    collect_memory=True,       # Track memory allocations
    cache_analysis=True,       # Monitor executable cache
    sample_rate=1.0,           # Sampling rate (1.0 = all ops)
)
```

## 📖 Documentation

See the `notebooks/` directory for detailed examples and tutorials.

## 🤝 Contributing

Contributions welcome! Please read our contributing guidelines.

## 📄 License

MIT License - see LICENSE file for details.

