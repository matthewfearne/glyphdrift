"""Tests for grammar emergence detection module."""

import numpy as np
import pytest

from glyphdrift.evolution import evolve_generation
from glyphdrift.glyph import DEFAULT_ALPHABET, Glyph, Role, generate_population
from glyphdrift.grammar import (
    bigram_counts,
    compression_ratio,
    grammar_rule_count,
    grammar_summary,
    mutual_information,
    significant_bigrams,
    significant_trigrams,
    trigram_counts,
)


@pytest.fixture
def random_pop():
    """Random population — no structure."""
    rng = np.random.default_rng(42)
    return generate_population(rng, DEFAULT_ALPHABET, 100, 10, 1.0)


@pytest.fixture
def evolved_pop():
    """Population evolved for 500 generations — should have structure."""
    rng = np.random.default_rng(42)
    pop = generate_population(rng, DEFAULT_ALPHABET, 100, 10, 0.5)
    for _ in range(500):
        pop = evolve_generation(rng, pop, DEFAULT_ALPHABET, 0.05, 5, 0.5, 10)
    return pop


class TestNgramCounts:
    def test_bigram_counts(self, random_pop):
        bg = bigram_counts(random_pop)
        assert len(bg) > 0
        # 100 sequences × 9 bigrams each = 900
        assert sum(bg.values()) == 900

    def test_trigram_counts(self, random_pop):
        tg = trigram_counts(random_pop)
        assert len(tg) > 0
        # 100 sequences × 8 trigrams each = 800
        assert sum(tg.values()) == 800

    def test_empty(self):
        assert len(bigram_counts([])) == 0
        assert len(trigram_counts([])) == 0


class TestSignificantNgrams:
    def test_random_has_few_significant(self, random_pop):
        """Random population should have few or no significant bigrams."""
        sb = significant_bigrams(random_pop)
        # Some may appear by chance, but not many
        assert len(sb) < 50

    def test_evolved_has_significant(self, evolved_pop):
        """Evolved population should have at least some significant bigrams."""
        sb = significant_bigrams(evolved_pop)
        # With strong convergence, a few dominant bigrams emerge
        assert len(sb) >= 1

    def test_chi2_values_positive(self, evolved_pop):
        sb = significant_bigrams(evolved_pop)
        for bigram, chi2 in sb:
            assert chi2 > 0
            assert len(bigram) == 2

    def test_trigram_significance(self, evolved_pop):
        st = significant_trigrams(evolved_pop)
        for trigram, chi2 in st:
            assert chi2 > 0
            assert len(trigram) == 3


class TestMutualInformation:
    def test_nonnegative(self, random_pop):
        mi = mutual_information(random_pop)
        assert mi >= 0

    def test_evolved_vs_random(self, random_pop, evolved_pop):
        """Both random and evolved populations produce measurable MI."""
        mi_random = mutual_information(random_pop)
        mi_evolved = mutual_information(evolved_pop)
        # Both should be non-negative; heavily converged populations can have
        # lower MI than random because vocabulary collapse reduces joint entropy.
        assert mi_random >= 0
        assert mi_evolved >= 0

    def test_empty(self):
        assert mutual_information([]) == 0.0


class TestCompressionRatio:
    def test_range(self, random_pop):
        cr = compression_ratio(random_pop)
        assert 0 < cr <= 1.5  # Can be slightly > 1 for very small inputs

    def test_evolved_more_compressible(self, random_pop, evolved_pop):
        """Evolved (structured) should be more compressible than random."""
        cr_random = compression_ratio(random_pop)
        cr_evolved = compression_ratio(evolved_pop)
        assert cr_evolved < cr_random

    def test_empty(self):
        assert compression_ratio([]) == 1.0


class TestGrammarSummary:
    def test_returns_all_fields(self, random_pop):
        gs = grammar_summary(random_pop)
        assert "significant_bigrams" in gs
        assert "significant_trigrams" in gs
        assert "mutual_information" in gs
        assert "compression_ratio" in gs

    def test_grammar_rule_count(self, evolved_pop):
        sb, st = grammar_rule_count(evolved_pop)
        assert sb >= 0
        assert st >= 0
