"""Tests for metrics module."""

import numpy as np
import pytest

from glyphdrift.glyph import DEFAULT_ALPHABET, generate_population
from glyphdrift.metrics import compute_generation_metrics, RunMetrics, GenerationMetrics


class TestComputeGenerationMetrics:
    def test_basic_metrics(self):
        rng = np.random.default_rng(42)
        pop = generate_population(rng, DEFAULT_ALPHABET, 50, 10, 0.5)
        m = compute_generation_metrics(pop, 0)
        assert m.generation == 0
        assert m.shannon_entropy > 0
        assert 0 < m.diversity_ratio <= 1.0
        assert m.unique_glyphs > 0
        assert m.most_common_glyph != ""

    def test_empty_population(self):
        m = compute_generation_metrics([], 0)
        assert m.shannon_entropy == 0.0
        assert m.diversity_ratio == 0.0

    def test_role_fractions_sum(self):
        rng = np.random.default_rng(42)
        pop = generate_population(rng, DEFAULT_ALPHABET, 100, 10, 1.0)
        m = compute_generation_metrics(pop, 0)
        total = m.particle_frac + m.root_frac + m.suffix_frac + m.modifier_frac
        assert abs(total - 1.0) < 0.01

    def test_high_entropy_more_unique(self):
        rng_low = np.random.default_rng(42)
        rng_high = np.random.default_rng(42)
        pop_low = generate_population(rng_low, DEFAULT_ALPHABET, 100, 10, 0.1)
        pop_high = generate_population(rng_high, DEFAULT_ALPHABET, 100, 10, 0.9)
        m_low = compute_generation_metrics(pop_low, 0)
        m_high = compute_generation_metrics(pop_high, 0)
        assert m_high.shannon_entropy >= m_low.shannon_entropy

    def test_uniform_entropy_near_max(self):
        """Uniform random across 40 glyphs: max entropy = log2(40) ≈ 5.32."""
        rng = np.random.default_rng(42)
        pop = generate_population(rng, DEFAULT_ALPHABET, 1000, 20, 1.0)
        m = compute_generation_metrics(pop, 0)
        max_entropy = np.log2(40)
        # Should be close to max with 20000 draws
        assert m.shannon_entropy > max_entropy * 0.9


class TestRunMetrics:
    def test_from_generation_metrics(self):
        gen_list = [
            GenerationMetrics(generation=0, shannon_entropy=3.0, diversity_ratio=0.8),
            GenerationMetrics(generation=1, shannon_entropy=3.5, diversity_ratio=0.9),
        ]
        rm = RunMetrics.from_generation_metrics(gen_list, scenario="test", seed=42)
        assert rm.generations_completed == 2
        assert rm.final_shannon_entropy == 3.5
        assert rm.mean_shannon_entropy == 3.25

    def test_summary_dict(self):
        rm = RunMetrics(scenario="test", seed=42, final_shannon_entropy=3.14)
        d = rm.summary_dict()
        assert d["scenario"] == "test"
        assert "3.14" in d["final_shannon"]


class TestConfig:
    def test_default_config(self):
        from glyphdrift.config import GlyphDriftConfig
        c = GlyphDriftConfig()
        assert c.population_size == 100
        assert c.entropy == 0.5
        assert c.generations == 1000
        assert not c.use_evolution
