# Exponential Smoothing Pattern

## Intent
Provide adaptive estimates that give more weight to recent observations while maintaining historical context.

## Motivation
When tracking changing values over time (user behavior, system metrics, model performance), we need estimates that respond to genuine shifts without overreacting to noise. Exponential smoothing achieves this by blending each new observation with the existing estimate using a configurable decay factor.

## Applicability
- Tracking metrics that evolve over time (click-through rates, error rates)
- Smoothing noisy signals while preserving trends
- Adaptive baseline estimation for anomaly detection
- When recent data is more indicative of current state than older data

## Structure
```python
class ExponentialSmoother:
    def __init__(self, alpha=0.1):
        self.alpha = alpha  # Smoothing factor (0 < alpha <= 1)
        self.estimate = None

    def update(self, observation):
        if self.estimate is None:
            self.estimate = observation
        else:
            self.estimate = self.alpha * observation + (1 - self.alpha) * self.estimate
        return self.estimate
```

## Implementation Notes
- **Alpha selection**: Lower alpha (e.g., 0.1) = more smoothing, slower adaptation. Higher alpha (e.g., 0.5) = faster response to changes
- **Initialization**: First observation sets initial estimate
- **Variants**: Double exponential smoothing for trends, triple for seasonality

## Consequences

Benefits:
- Simple, computationally efficient
- Automatically adapts to changing conditions
- Requires minimal state (just current estimate)

Drawbacks:
- Choice of alpha requires tuning
- No explicit uncertainty quantification
- Can lag during rapid changes

## Related Patterns
- Combines with **Beta Distribution** for uncertainty-aware smoothing
- Alternative to **Context Window** when memory is constrained
