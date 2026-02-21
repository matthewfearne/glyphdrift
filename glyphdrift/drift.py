"""Chaotic drift dynamics for GlyphDrift (v4).

Mutation rate driven by logistic map instead of constant value.
x_{n+1} = r * x_n * (1 - x_n)

At r=3.9 (chaotic regime), the mutation rate oscillates unpredictably —
periods of stability punctuated by bursts of high mutation.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LogisticDrift:
    """Logistic map driving mutation rate dynamics.

    mutation_rate(t) = base_rate * x(t)
    where x(t) follows the logistic map.
    """

    r: float = 3.9          # Logistic parameter (3.9 = chaotic)
    x: float = 0.5          # Current state
    base_rate: float = 0.1  # Base mutation rate (scaled by x)
    storm_interval: int = 0  # Generations between entropy storms (0 = none)

    # Track history for analysis
    x_history: list[float] = field(default_factory=list)
    rate_history: list[float] = field(default_factory=list)

    def step(self, generation: int) -> float:
        """Advance one step and return the current mutation rate.

        Optionally applies entropy storms (forces x to extreme value).
        """
        # Check for entropy storm
        if self.storm_interval > 0 and generation > 0 and generation % self.storm_interval == 0:
            self.x = 0.99  # Force high mutation burst

        # Record current state
        self.x_history.append(self.x)
        rate = self.base_rate * self.x
        self.rate_history.append(rate)

        # Advance logistic map
        self.x = self.r * self.x * (1.0 - self.x)

        # Clamp to valid range (numerical safety)
        self.x = max(0.001, min(0.999, self.x))

        return rate

    @property
    def current_rate(self) -> float:
        """Current mutation rate without advancing."""
        return self.base_rate * self.x

    def mean_rate(self) -> float:
        """Mean mutation rate so far."""
        if not self.rate_history:
            return self.base_rate * self.x
        return sum(self.rate_history) / len(self.rate_history)

    def rate_variance(self) -> float:
        """Variance of mutation rate so far."""
        if len(self.rate_history) < 2:
            return 0.0
        mean = self.mean_rate()
        return sum((r - mean) ** 2 for r in self.rate_history) / (len(self.rate_history) - 1)
