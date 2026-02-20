# Context Window Pattern

## Intent
Limit analysis to recent history to adapt to non-stationary environments.

## Motivation
In changing environments, old data can mislead. By restricting attention to a recent window, we ensure estimates reflect current conditions rather than outdated patterns. This is particularly valuable when concept drift occurs — when the underlying data distribution shifts over time.

## Applicability
- Systems with concept drift (user preferences change, markets evolve)
- When storage/computation must be bounded
- Real-time analytics on streams
- When recent data is more relevant than old data

## Structure
```python
from collections import deque

class WindowedStats:
    def __init__(self, window_size=1000):
        self.window = deque(maxlen=window_size)

    def add(self, value):
        self.window.append(value)

    def mean(self):
        return sum(self.window) / len(self.window) if self.window else 0

    def quantile(self, q):
        import numpy as np
        return np.quantile(self.window, q) if self.window else 0
```

## Implementation Notes
- **Window size**: Trade-off between stability (larger) and adaptiveness (smaller)
- **Data structure**: Use `deque` for O(1) append and automatic eviction
- **Statistics**: Can compute any statistic over window (mean, median, percentiles)

## Consequences

Benefits:
- Adapts to changes automatically
- Bounded memory usage
- Simple to implement and understand

Drawbacks:
- Abrupt changes at window boundary (new data in, old data out)
- Requires storing full window
- Window size is a hyperparameter to tune

## Related Patterns
- Alternative to **Exponential Smoothing** (window vs weighted)
- Complements **Reservoir Sampling** when window doesn't fit in memory
- Often used with anomaly detection for adaptive thresholds
