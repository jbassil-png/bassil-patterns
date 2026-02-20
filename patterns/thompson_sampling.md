# Thompson Sampling Pattern

## Intent
Balance exploration and exploitation by sampling from posterior belief distributions.

## Motivation
In multi-armed bandit problems (A/B tests, recommendation systems, ad selection), we must choose between exploring uncertain options and exploiting known good ones. Thompson sampling provides an elegant, probabilistic solution: arms with higher uncertainty are naturally explored more, while arms with strong evidence are exploited.

## Applicability
- A/B/n testing with multiple variants
- Recommendation systems
- Ad selection and placement
- Resource allocation under uncertainty
- Any decision problem with explore/exploit tradeoff

## Structure
```python
class ThompsonSampler:
    def __init__(self, n_arms):
        # Use Beta distribution for each arm
        self.arms = [BetaEstimator() for _ in range(n_arms)]

    def select_arm(self):
        # Sample from each arm's posterior, pick highest
        samples = [arm.sample() for arm in self.arms]
        return samples.index(max(samples))

    def update(self, arm_id, reward):
        # Reward should be boolean (success/failure)
        self.arms[arm_id].update(reward)


class BetaEstimator:
    """Inline copy — see Beta Distribution pattern for full version."""
    def __init__(self, alpha=1, beta=1):
        self.alpha = alpha
        self.beta = beta

    def update(self, success: bool):
        if success:
            self.alpha += 1
        else:
            self.beta += 1

    def sample(self):
        import numpy as np
        return np.random.beta(self.alpha, self.beta)
```

## Implementation Notes
- **Beta-Bernoulli**: Most common implementation for binary rewards
- **Gaussian**: For continuous rewards, use Gaussian posterior
- **Particle filters**: For complex posteriors that aren't analytically tractable

## Consequences

Benefits:
- Automatically balances exploration/exploitation
- No hyperparameters to tune (unlike epsilon-greedy, UCB)
- Optimal regret bounds in many settings
- Naturally handles delayed feedback

Drawbacks:
- Requires maintaining posterior for each arm
- Assumes rewards are stationary (use with **Exponential Smoothing** if not)
- More complex than simple strategies like epsilon-greedy

## Related Patterns
- Depends on **Beta Distribution** for posterior tracking
- Combines with **Context Window** or **Exponential Smoothing** for non-stationary bandits
- Alternative to UCB, epsilon-greedy for exploration
