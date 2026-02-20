# Reservoir Sampling Pattern

## Intent
Maintain a fixed-size random sample from a stream of unknown length.

## Motivation
When processing streams where you can't store everything, reservoir sampling guarantees each item has equal probability of being in the sample, regardless of stream length. This enables statistically valid analysis on arbitrarily large or unbounded streams with constant memory.

## Applicability
- Sampling from logs, event streams, large datasets
- When total size is unknown or unbounded
- Memory-constrained environments
- Debugging/monitoring large-scale systems

## Structure
```python
import random

class ReservoirSampler:
    def __init__(self, k):
        self.k = k
        self.reservoir = []
        self.n = 0  # Items seen so far

    def add(self, item):
        self.n += 1
        if len(self.reservoir) < self.k:
            self.reservoir.append(item)
        else:
            # Replace existing item with probability k/n
            j = random.randint(0, self.n - 1)
            if j < self.k:
                self.reservoir[j] = item

    def sample(self):
        return self.reservoir
```

## Implementation Notes
- **Correctness**: Each item has exactly k/n probability of being in reservoir
- **Single pass**: Only need to see each item once
- **Variants**: Weighted reservoir sampling for non-uniform sampling

## Consequences

Benefits:
- Fixed memory usage regardless of stream size
- Provably uniform random sample
- Single-pass algorithm

Drawbacks:
- Sample size k must be chosen upfront
- No guarantee of recent bias (use **Context Window** for that)
- Randomness may not be reproducible across runs

## Related Patterns
- Alternative to **Context Window** when uniformity matters more than recency
- Useful for debugging online learning systems
- Combines with stratified sampling for balanced samples
