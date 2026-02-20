# Beta Distribution Pattern

## Intent
Track proportions/probabilities with uncertainty when you have limited observations.

## Motivation
When estimating rates (conversion rates, success probabilities) from small samples, point estimates can be misleading. The Beta distribution provides a principled way to represent both the estimate and our confidence in it. As more data arrives, the distribution narrows, reflecting growing certainty.

## Applicability
- A/B testing with limited data
- Multi-armed bandit problems
- Estimating click-through rates, conversion rates, success probabilities
- Thompson sampling for exploration/exploitation
- When you need to quantify uncertainty in proportion estimates

## Structure
```python
class BetaEstimator:
    def __init__(self, alpha=1, beta=1):
        # alpha=1, beta=1 gives uniform prior (no initial bias)
        self.alpha = alpha  # Successes + prior
        self.beta = beta    # Failures + prior

    def update(self, success: bool):
        if success:
            self.alpha += 1
        else:
            self.beta += 1

    def mean(self):
        return self.alpha / (self.alpha + self.beta)

    def sample(self):
        import numpy as np
        return np.random.beta(self.alpha, self.beta)

    def credible_interval(self, confidence=0.95):
        from scipy.stats import beta
        lower = (1 - confidence) / 2
        upper = 1 - lower
        return beta.ppf([lower, upper], self.alpha, self.beta)
```

## Implementation Notes
- **Prior selection**: Start with (1, 1) for uniform prior, or use domain knowledge
- **Conjugacy**: Beta is conjugate prior for binomial likelihood — updates are simple
- **Sample size**: Width of credible interval shrinks as alpha + beta grows

## Consequences

Benefits:
- Natural representation of uncertainty
- Mathematically principled (Bayesian inference)
- Enables Thompson sampling and other sophisticated strategies

Drawbacks:
- Requires understanding of Bayesian concepts
- Less interpretable than simple point estimates
- Need scipy/numpy for advanced features

## Related Patterns
- Powers **Thompson Sampling** for bandit algorithms
- Combines with **Exponential Smoothing** for time-varying rates
- Alternative to frequentist confidence intervals
