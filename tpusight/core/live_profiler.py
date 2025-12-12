"""Live profiling mode for TPUsight with real-time updates."""

from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from collections import deque
from threading import Thread, Lock, Event
from functools import wraps
import time
import uuid

try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False

from tpusight.core.data_collector import (
    ProfileData, 
    OperationRecord, 
    CompilationRecord,
    OperationType,
)
from tpusight.core.jax_tracer import get_op_type, calculate_matmul_flops, estimate_bytes_accessed
from tpusight.utils.helpers import estimate_mxu_utilization, calculate_padding_waste


@dataclass
class LiveAlert:
    """A real-time alert for performance issues."""
    
    id: str
    timestamp: float
    severity: str  # "critical", "warning", "info"
    category: str
    message: str
    operation: Optional[str] = None
    value: Optional[float] = None
    threshold: Optional[float] = None


@dataclass
class LiveMetrics:
    """Current live metrics snapshot."""
    
    timestamp: float
    
    # Counts
    total_ops: int = 0
    ops_per_second: float = 0.0
    
    # Utilization (rolling average)
    mxu_utilization: float = 0.0
    memory_utilization: float = 0.0
    
    # Time breakdown (rolling)
    compute_pct: float = 0.0
    memory_pct: float = 0.0
    
    # Recent operation
    last_op_name: str = ""
    last_op_duration_ms: float = 0.0
    
    # Alerts
    active_alerts: int = 0


class LiveProfiler:
    """
    Live profiling mode with real-time updates.
    
    Provides:
    - Real-time metric streaming
    - Live alerts for inefficiencies
    - Callback hooks for custom handling
    - Auto-updating dashboards
    
    Example:
        >>> live = LiveProfiler()
        >>> live.on_alert(lambda alert: print(f"ALERT: {alert.message}"))
        >>> 
        >>> @live.trace
        ... def my_function(x):
        ...     return jnp.dot(x, x.T)
        >>> 
        >>> live.start()
        >>> # ... run workload ...
        >>> live.stop()
    """
    
    def __init__(
        self,
        window_size: int = 100,
        alert_thresholds: Optional[Dict[str, float]] = None,
        update_interval: float = 0.5,
    ):
        """
        Initialize live profiler.
        
        Args:
            window_size: Number of recent operations to keep for rolling stats
            alert_thresholds: Custom thresholds for alerts
            update_interval: How often to compute rolling stats (seconds)
        """
        self.window_size = window_size
        self.update_interval = update_interval
        
        # Default alert thresholds
        self.thresholds = {
            "mxu_utilization_low": 30.0,      # Alert if MXU < 30%
            "mxu_utilization_warning": 50.0,  # Warning if MXU < 50%
            "padding_waste_high": 30.0,       # Alert if padding > 30%
            "cache_miss_rate_high": 50.0,     # Alert if cache miss > 50%
            "memory_bound_threshold": 70.0,   # Alert if memory-bound > 70%
            "compilation_time_high": 5000.0,  # Alert if compilation > 5s
        }
        if alert_thresholds:
            self.thresholds.update(alert_thresholds)
        
        # Data storage
        self._recent_ops: deque = deque(maxlen=window_size)
        self._recent_compilations: deque = deque(maxlen=window_size)
        self._alerts: deque = deque(maxlen=1000)
        self._metrics_history: deque = deque(maxlen=1000)
        
        # Current state
        self._current_metrics = LiveMetrics(timestamp=time.time())
        self._start_time: Optional[float] = None
        self._total_ops = 0
        
        # Thread safety
        self._lock = Lock()
        self._stop_event = Event()
        self._update_thread: Optional[Thread] = None
        
        # Callbacks
        self._on_operation_callbacks: List[Callable[[OperationRecord], None]] = []
        self._on_alert_callbacks: List[Callable[[LiveAlert], None]] = []
        self._on_metrics_callbacks: List[Callable[[LiveMetrics], None]] = []
        
        # Session ID
        self.session_id = f"live_{uuid.uuid4().hex[:8]}"
        
        # For compatibility with TPUsight
        self.profile_data = ProfileData(
            session_id=self.session_id,
            start_time=time.time(),
        )
        self._detect_device()
    
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
        except Exception:
            self.profile_data.device_type = "unknown"
            self.profile_data.device_count = 0
    
    def start(self):
        """Start live profiling."""
        self._start_time = time.time()
        self._stop_event.clear()
        
        # Start background update thread
        self._update_thread = Thread(target=self._update_loop, daemon=True)
        self._update_thread.start()
        
        print(f"🔴 Live profiling started (session: {self.session_id})")
    
    def stop(self):
        """Stop live profiling."""
        self._stop_event.set()
        if self._update_thread:
            self._update_thread.join(timeout=2.0)
        
        self.profile_data.finalize()
        
        print(f"⏹️  Live profiling stopped. Captured {self._total_ops} operations, {len(self._alerts)} alerts")
    
    def _update_loop(self):
        """Background thread for computing rolling statistics."""
        while not self._stop_event.is_set():
            self._compute_rolling_stats()
            self._stop_event.wait(self.update_interval)
    
    def _compute_rolling_stats(self):
        """Compute rolling statistics from recent operations."""
        with self._lock:
            now = time.time()
            
            if not self._recent_ops:
                return
            
            # Calculate rolling averages
            recent_list = list(self._recent_ops)
            
            # MXU utilization
            mxu_values = [op.mxu_utilization for op in recent_list if op.mxu_utilization is not None]
            avg_mxu = sum(mxu_values) / len(mxu_values) if mxu_values else 0
            
            # Time breakdown estimate
            total_duration = sum(op.duration_ns for op in recent_list)
            compute_ops = [op for op in recent_list if op.op_type in [OperationType.MATMUL, OperationType.CONV]]
            compute_duration = sum(op.duration_ns for op in compute_ops)
            compute_pct = (compute_duration / total_duration * 100) if total_duration > 0 else 0
            
            # Ops per second
            elapsed = now - self._start_time if self._start_time else 1
            ops_per_second = self._total_ops / elapsed if elapsed > 0 else 0
            
            # Latest operation
            latest = recent_list[-1] if recent_list else None
            
            # Update metrics
            self._current_metrics = LiveMetrics(
                timestamp=now,
                total_ops=self._total_ops,
                ops_per_second=ops_per_second,
                mxu_utilization=avg_mxu,
                compute_pct=compute_pct,
                memory_pct=100 - compute_pct,
                last_op_name=latest.name if latest else "",
                last_op_duration_ms=latest.duration_ns / 1e6 if latest else 0,
                active_alerts=sum(1 for a in self._alerts if now - a.timestamp < 60),
            )
            
            self._metrics_history.append(self._current_metrics)
            
            # Fire callbacks
            for callback in self._on_metrics_callbacks:
                try:
                    callback(self._current_metrics)
                except Exception:
                    pass
    
    def trace(self, fn: Callable) -> Callable:
        """Decorator to trace a function with live profiling."""
        if not HAS_JAX:
            return fn
        
        fn_name = fn.__name__ if hasattr(fn, '__name__') else str(fn)
        jitted = jax.jit(fn)
        
        @wraps(fn)
        def wrapper(*args, **kwargs):
            return self._trace_call(jitted, fn_name, args, kwargs)
        
        return wrapper
    
    def _trace_call(
        self,
        jitted_fn: Callable,
        fn_name: str,
        args: tuple,
        kwargs: dict,
    ) -> Any:
        """Trace a function call and record metrics."""
        # Get input info
        input_shapes = []
        input_dtypes = []
        
        for arg in args:
            if hasattr(arg, 'shape'):
                input_shapes.append(tuple(arg.shape))
                input_dtypes.append(str(arg.dtype))
        
        # Time execution
        start_time = time.perf_counter_ns()
        result = jitted_fn(*args, **kwargs)
        
        if HAS_JAX and hasattr(result, 'block_until_ready'):
            result.block_until_ready()
        
        end_time = time.perf_counter_ns()
        duration_ns = end_time - start_time
        
        # Get output info
        output_shapes = []
        output_dtypes = []
        
        if hasattr(result, 'shape'):
            output_shapes.append(tuple(result.shape))
            output_dtypes.append(str(result.dtype))
        elif isinstance(result, (tuple, list)):
            for r in result:
                if hasattr(r, 'shape'):
                    output_shapes.append(tuple(r.shape))
                    output_dtypes.append(str(r.dtype))
        
        # Create operation record
        op_record = self._create_operation_record(
            fn_name, input_shapes, output_shapes,
            input_dtypes, output_dtypes, duration_ns
        )
        
        # Record and check for alerts
        self._record_operation(op_record)
        
        return result
    
    def _create_operation_record(
        self,
        name: str,
        input_shapes: List,
        output_shapes: List,
        input_dtypes: List,
        output_dtypes: List,
        duration_ns: float,
    ) -> OperationRecord:
        """Create an operation record with computed metrics."""
        op_type = get_op_type(name, input_shapes)
        
        flops = None
        mxu_util = None
        padding_waste = None
        
        if op_type == OperationType.MATMUL and len(input_shapes) >= 2:
            flops = calculate_matmul_flops(input_shapes[0], input_shapes[1])
            
            if len(input_shapes[0]) >= 2 and len(input_shapes[1]) >= 2:
                m = input_shapes[0][-2]
                k = input_shapes[0][-1]
                n = input_shapes[1][-1]
                
                mxu_metrics = estimate_mxu_utilization(m, n, k)
                mxu_util = mxu_metrics["mxu_utilization_pct"]
                
                if output_shapes:
                    padding_info = calculate_padding_waste(output_shapes[0])
                    padding_waste = padding_info["wasted_compute_pct"]
        
        bytes_accessed = estimate_bytes_accessed(
            input_shapes, output_shapes,
            input_dtypes[0] if input_dtypes else "float32"
        )
        
        return OperationRecord(
            name=name,
            op_type=op_type,
            timestamp=time.time(),
            duration_ns=duration_ns,
            input_shapes=input_shapes,
            output_shapes=output_shapes,
            input_dtypes=input_dtypes,
            output_dtypes=output_dtypes,
            flops=flops,
            bytes_accessed=bytes_accessed,
            mxu_utilization=mxu_util,
            padding_waste_pct=padding_waste,
        )
    
    def _record_operation(self, op: OperationRecord):
        """Record an operation and check for alerts."""
        with self._lock:
            self._recent_ops.append(op)
            self._total_ops += 1
            self.profile_data.add_operation(op)
        
        # Check for alerts
        self._check_alerts(op)
        
        # Fire callbacks
        for callback in self._on_operation_callbacks:
            try:
                callback(op)
            except Exception:
                pass
    
    def _check_alerts(self, op: OperationRecord):
        """Check if operation triggers any alerts."""
        alerts = []
        
        # Low MXU utilization
        if op.mxu_utilization is not None:
            if op.mxu_utilization < self.thresholds["mxu_utilization_low"]:
                alerts.append(LiveAlert(
                    id=f"mxu_{uuid.uuid4().hex[:6]}",
                    timestamp=time.time(),
                    severity="critical",
                    category="mxu_utilization",
                    message=f"Very low MXU utilization: {op.mxu_utilization:.1f}%",
                    operation=op.name,
                    value=op.mxu_utilization,
                    threshold=self.thresholds["mxu_utilization_low"],
                ))
            elif op.mxu_utilization < self.thresholds["mxu_utilization_warning"]:
                alerts.append(LiveAlert(
                    id=f"mxu_{uuid.uuid4().hex[:6]}",
                    timestamp=time.time(),
                    severity="warning",
                    category="mxu_utilization",
                    message=f"Low MXU utilization: {op.mxu_utilization:.1f}%",
                    operation=op.name,
                    value=op.mxu_utilization,
                    threshold=self.thresholds["mxu_utilization_warning"],
                ))
        
        # High padding waste
        if op.padding_waste_pct is not None:
            if op.padding_waste_pct > self.thresholds["padding_waste_high"]:
                alerts.append(LiveAlert(
                    id=f"pad_{uuid.uuid4().hex[:6]}",
                    timestamp=time.time(),
                    severity="warning",
                    category="padding",
                    message=f"High padding waste: {op.padding_waste_pct:.1f}%",
                    operation=op.name,
                    value=op.padding_waste_pct,
                    threshold=self.thresholds["padding_waste_high"],
                ))
        
        # Record alerts and fire callbacks
        for alert in alerts:
            with self._lock:
                self._alerts.append(alert)
            
            for callback in self._on_alert_callbacks:
                try:
                    callback(alert)
                except Exception:
                    pass
    
    def record_compilation(self, fn_name: str, compilation_time_ms: float, cache_hit: bool):
        """Manually record a compilation event."""
        comp = CompilationRecord(
            function_name=fn_name,
            timestamp=time.time(),
            compilation_time_ms=compilation_time_ms,
            input_shapes=[],
            input_dtypes=[],
            cache_hit=cache_hit,
        )
        
        with self._lock:
            self._recent_compilations.append(comp)
            self.profile_data.add_compilation(comp)
        
        # Alert on slow compilation
        if not cache_hit and compilation_time_ms > self.thresholds["compilation_time_high"]:
            alert = LiveAlert(
                id=f"comp_{uuid.uuid4().hex[:6]}",
                timestamp=time.time(),
                severity="warning",
                category="compilation",
                message=f"Slow compilation: {compilation_time_ms:.0f}ms for {fn_name}",
                operation=fn_name,
                value=compilation_time_ms,
                threshold=self.thresholds["compilation_time_high"],
            )
            
            with self._lock:
                self._alerts.append(alert)
            
            for callback in self._on_alert_callbacks:
                try:
                    callback(alert)
                except Exception:
                    pass
    
    # Callback registration
    def on_operation(self, callback: Callable[[OperationRecord], None]):
        """Register callback for each operation."""
        self._on_operation_callbacks.append(callback)
        return callback
    
    def on_alert(self, callback: Callable[[LiveAlert], None]):
        """Register callback for alerts."""
        self._on_alert_callbacks.append(callback)
        return callback
    
    def on_metrics(self, callback: Callable[[LiveMetrics], None]):
        """Register callback for metrics updates."""
        self._on_metrics_callbacks.append(callback)
        return callback
    
    # Accessors
    def get_current_metrics(self) -> LiveMetrics:
        """Get current live metrics."""
        return self._current_metrics
    
    def get_recent_alerts(self, n: int = 10) -> List[LiveAlert]:
        """Get N most recent alerts."""
        with self._lock:
            return list(self._alerts)[-n:]
    
    def get_metrics_history(self) -> List[LiveMetrics]:
        """Get metrics history for plotting."""
        with self._lock:
            return list(self._metrics_history)
    
    def get_alert_counts(self) -> Dict[str, int]:
        """Get alert counts by category."""
        with self._lock:
            counts: Dict[str, int] = {}
            for alert in self._alerts:
                counts[alert.category] = counts.get(alert.category, 0) + 1
            return counts
    
    def clear_alerts(self):
        """Clear all alerts."""
        with self._lock:
            self._alerts.clear()
    
    def __repr__(self) -> str:
        status = "running" if self._update_thread and self._update_thread.is_alive() else "stopped"
        return f"LiveProfiler(status={status}, ops={self._total_ops}, alerts={len(self._alerts)})"

