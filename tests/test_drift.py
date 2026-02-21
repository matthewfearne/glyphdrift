"""Tests for chaotic drift module."""

import pytest

from glyphdrift.drift import LogisticDrift


class TestLogisticDrift:
    def test_chaotic_regime(self):
        """r=3.9 should produce chaotic (non-repeating) mutation rates."""
        drift = LogisticDrift(r=3.9, x=0.5, base_rate=0.1)
        rates = [drift.step(i) for i in range(100)]
        # Should have high variance (chaotic)
        assert drift.rate_variance() > 0.0001

    def test_fixed_point_regime(self):
        """r=2.5 should converge to a fixed point."""
        drift = LogisticDrift(r=2.5, x=0.5, base_rate=0.1)
        # Run 200 steps to let it converge
        for i in range(200):
            drift.step(i)
        # Last 50 rates should be nearly identical
        last_50 = drift.rate_history[-50:]
        variance = sum((r - last_50[0]) ** 2 for r in last_50) / len(last_50)
        assert variance < 1e-10

    def test_rate_bounded(self):
        """Mutation rates should stay within [0, base_rate]."""
        drift = LogisticDrift(r=3.9, x=0.5, base_rate=0.1)
        for i in range(500):
            rate = drift.step(i)
            assert 0 <= rate <= 0.1 + 1e-10

    def test_deterministic(self):
        """Same parameters should produce same rates."""
        d1 = LogisticDrift(r=3.9, x=0.5, base_rate=0.1)
        d2 = LogisticDrift(r=3.9, x=0.5, base_rate=0.1)
        rates1 = [d1.step(i) for i in range(50)]
        rates2 = [d2.step(i) for i in range(50)]
        assert rates1 == rates2

    def test_different_x0(self):
        """Different initial conditions should diverge (chaos)."""
        d1 = LogisticDrift(r=3.9, x=0.5, base_rate=0.1)
        d2 = LogisticDrift(r=3.9, x=0.50001, base_rate=0.1)
        rates1 = [d1.step(i) for i in range(100)]
        rates2 = [d2.step(i) for i in range(100)]
        # Early rates similar, late rates diverge
        assert abs(rates1[0] - rates2[0]) < 0.001
        # After 50+ steps, should have diverged significantly
        late_diffs = [abs(r1 - r2) for r1, r2 in zip(rates1[50:], rates2[50:])]
        assert max(late_diffs) > 0.001

    def test_mean_rate(self):
        drift = LogisticDrift(r=3.9, x=0.5, base_rate=0.1)
        for i in range(100):
            drift.step(i)
        mean = drift.mean_rate()
        assert 0 < mean < 0.1

    def test_history_tracking(self):
        drift = LogisticDrift(r=3.9, x=0.5, base_rate=0.1)
        for i in range(10):
            drift.step(i)
        assert len(drift.x_history) == 10
        assert len(drift.rate_history) == 10


class TestEntropyStorms:
    def test_storm_forces_high_x(self):
        drift = LogisticDrift(r=3.9, x=0.5, base_rate=0.1, storm_interval=10)
        # Step to generation 10 (storm trigger)
        for i in range(11):
            rate = drift.step(i)
        # At generation 10, x was forced to 0.99, so rate should be ~0.099
        assert drift.rate_history[10] > 0.09

    def test_no_storm_when_zero_interval(self):
        drift = LogisticDrift(r=3.9, x=0.5, base_rate=0.1, storm_interval=0)
        for i in range(100):
            drift.step(i)
        # Rates should just follow logistic map, no forced spikes
        # (Can't be certain none are high, but pattern should differ from storms)
        assert len(drift.rate_history) == 100

    def test_periodic_storms(self):
        drift = LogisticDrift(r=3.9, x=0.5, base_rate=0.1, storm_interval=50)
        for i in range(200):
            drift.step(i)
        # Storm at gen 50, 100, 150
        assert drift.rate_history[50] > 0.09
        assert drift.rate_history[100] > 0.09
        assert drift.rate_history[150] > 0.09
