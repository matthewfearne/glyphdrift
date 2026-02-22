"""Tests for evolution module."""

import numpy as np
import pytest

from glyphdrift.evolution import (
    bigram_frequencies,
    compute_population_fitness,
    evolve_generation,
    fitness_bigram_coherence,
    fitness_sliding_window,
    mutate,
    mutate_deletion,
    mutate_insertion,
    mutate_substitution,
    tournament_select,
    uniform_bigram_prior,
    uniform_crossover,
)
from glyphdrift.glyph import DEFAULT_ALPHABET, Glyph, Role, generate_population


@pytest.fixture
def small_pop():
    rng = np.random.default_rng(42)
    return generate_population(rng, DEFAULT_ALPHABET, 20, 10, 0.5)


@pytest.fixture
def rng():
    return np.random.default_rng(42)


class TestBigramFrequencies:
    def test_counts_bigrams(self, small_pop):
        bg = bigram_frequencies(small_pop)
        assert len(bg) > 0
        # Total bigrams = 20 sequences × 9 bigrams each = 180
        assert sum(bg.values()) == 20 * 9

    def test_empty_population(self):
        bg = bigram_frequencies([])
        assert len(bg) == 0


class TestFitness:
    def test_fitness_nonnegative(self, small_pop):
        bg = bigram_frequencies(small_pop)
        for seq in small_pop:
            f = fitness_bigram_coherence(seq, bg)
            assert f >= 0

    def test_repeated_bigrams_score_higher(self):
        """A sequence with common bigrams should score higher."""
        g1, g2, g3 = DEFAULT_ALPHABET[0], DEFAULT_ALPHABET[1], DEFAULT_ALPHABET[2]
        # Population where (g1, g2) appears many times
        pop = [[g1, g2, g3] for _ in range(10)]
        bg = bigram_frequencies(pop)

        # Sequence with the common bigram
        common_seq = [g1, g2, g3]
        # Sequence with a rare bigram
        rare_seq = [g3, g3, g3]

        f_common = fitness_bigram_coherence(common_seq, bg)
        f_rare = fitness_bigram_coherence(rare_seq, bg)
        assert f_common > f_rare

    def test_compute_population_fitness(self, small_pop):
        fitnesses = compute_population_fitness(small_pop)
        assert len(fitnesses) == len(small_pop)
        assert all(f >= 0 for f in fitnesses)


class TestSelection:
    def test_returns_copy(self, small_pop, rng):
        fitnesses = compute_population_fitness(small_pop)
        selected = tournament_select(rng, small_pop, fitnesses, 3)
        # Should be a list of Glyph, same length as a sequence
        assert len(selected) == 10
        assert all(isinstance(g, Glyph) for g in selected)

    def test_tournament_size_1_is_random(self, small_pop, rng):
        """Tournament size 1 = pick one random individual (no selection pressure)."""
        fitnesses = compute_population_fitness(small_pop)
        selected = tournament_select(rng, small_pop, fitnesses, 1)
        assert len(selected) == 10


class TestMutation:
    def test_substitution_preserves_length(self, rng):
        seq = list(DEFAULT_ALPHABET[:10])
        mutated = mutate_substitution(rng, seq, DEFAULT_ALPHABET, 0.5)
        assert len(mutated) == len(seq)

    def test_substitution_zero_rate(self, rng):
        seq = list(DEFAULT_ALPHABET[:10])
        mutated = mutate_substitution(rng, seq, DEFAULT_ALPHABET, 0.0)
        assert [g.symbol for g in mutated] == [g.symbol for g in seq]

    def test_insertion_adds_one(self, rng):
        seq = list(DEFAULT_ALPHABET[:5])
        result = mutate_insertion(rng, seq, DEFAULT_ALPHABET)
        assert len(result) == 6

    def test_deletion_removes_one(self, rng):
        seq = list(DEFAULT_ALPHABET[:5])
        result = mutate_deletion(rng, seq)
        assert len(result) == 4

    def test_deletion_min_length(self, rng):
        seq = [DEFAULT_ALPHABET[0]]
        result = mutate_deletion(rng, seq)
        assert len(result) == 1

    def test_mutate_enforces_target_length(self, rng):
        seq = list(DEFAULT_ALPHABET[:10])
        mutated = mutate(rng, seq, DEFAULT_ALPHABET, 0.5, 10)
        assert len(mutated) == 10


class TestCrossover:
    def test_uniform_crossover_length(self, rng):
        p1 = list(DEFAULT_ALPHABET[:10])
        p2 = list(DEFAULT_ALPHABET[10:20])
        child = uniform_crossover(rng, p1, p2)
        assert len(child) == 10

    def test_child_from_parents(self, rng):
        p1 = list(DEFAULT_ALPHABET[:10])
        p2 = list(DEFAULT_ALPHABET[10:20])
        child = uniform_crossover(rng, p1, p2)
        p1_symbols = {g.symbol for g in p1}
        p2_symbols = {g.symbol for g in p2}
        for g in child:
            assert g.symbol in p1_symbols or g.symbol in p2_symbols


class TestEvolveGeneration:
    def test_preserves_pop_size(self, small_pop, rng):
        new_pop = evolve_generation(
            rng, small_pop, DEFAULT_ALPHABET, 0.1, 3, 0.5, 10,
        )
        assert len(new_pop) == len(small_pop)

    def test_preserves_seq_length(self, small_pop, rng):
        new_pop = evolve_generation(
            rng, small_pop, DEFAULT_ALPHABET, 0.1, 3, 0.5, 10,
        )
        for seq in new_pop:
            assert len(seq) == 10

    def test_deterministic(self, small_pop):
        pop1 = evolve_generation(
            np.random.default_rng(42), small_pop, DEFAULT_ALPHABET, 0.1, 3, 0.5, 10,
        )
        pop2 = evolve_generation(
            np.random.default_rng(42), small_pop, DEFAULT_ALPHABET, 0.1, 3, 0.5, 10,
        )
        for s1, s2 in zip(pop1, pop2):
            assert [g.symbol for g in s1] == [g.symbol for g in s2]

    def test_evolution_changes_population(self, small_pop, rng):
        """After many generations, population should differ from initial."""
        pop = small_pop
        for _ in range(50):
            pop = evolve_generation(
                rng, pop, DEFAULT_ALPHABET, 0.1, 3, 0.5, 10,
            )
        # Population should have changed
        initial_seqs = set(tuple(g.symbol for g in s) for s in small_pop)
        final_seqs = set(tuple(g.symbol for g in s) for s in pop)
        assert initial_seqs != final_seqs

    def test_selection_reduces_diversity(self):
        """Strong selection (large tournament) should reduce diversity over time."""
        rng = np.random.default_rng(42)
        pop = generate_population(rng, DEFAULT_ALPHABET, 100, 10, 0.5)

        # Evolve with strong selection
        for _ in range(200):
            pop = evolve_generation(
                rng, pop, DEFAULT_ALPHABET, 0.02, 7, 0.5, 10,
            )

        # Count unique sequences
        unique = len(set(tuple(g.symbol for g in s) for s in pop))
        # With strong selection and low mutation, diversity should drop below 100%
        assert unique < 100


# ── v6: Sliding Window Fitness ──────────────────────────────────────


class TestUniformBigramPrior:
    def test_has_all_bigram_pairs(self):
        prior = uniform_bigram_prior(DEFAULT_ALPHABET, 100, 10)
        n = len(DEFAULT_ALPHABET)
        assert len(prior) == n * n

    def test_all_counts_equal_one(self):
        prior = uniform_bigram_prior(DEFAULT_ALPHABET, 100, 10)
        assert all(v == 1 for v in prior.values())

    def test_uses_symbols(self):
        prior = uniform_bigram_prior(DEFAULT_ALPHABET, 100, 10)
        for (a, b) in prior:
            assert isinstance(a, str)
            assert isinstance(b, str)


class TestSlidingWindowFitness:
    def test_nonnegative(self, small_pop):
        prior = uniform_bigram_prior(DEFAULT_ALPHABET, 20, 10)
        for seq in small_pop:
            f = fitness_sliding_window(seq, prior)
            assert f >= 0

    def test_short_sequence(self):
        g = DEFAULT_ALPHABET[0]
        prior = uniform_bigram_prior(DEFAULT_ALPHABET, 100, 10)
        assert fitness_sliding_window([g], prior) == 0.0
        assert fitness_sliding_window([], prior) == 0.0

    def test_scores_against_prev_bigrams(self):
        """Sequences matching prev_bigrams should score higher."""
        g1, g2, g3 = DEFAULT_ALPHABET[0], DEFAULT_ALPHABET[1], DEFAULT_ALPHABET[2]
        from collections import Counter
        prev = Counter({(g1.symbol, g2.symbol): 100, (g2.symbol, g3.symbol): 50})
        # Sequence with matching bigrams
        match_seq = [g1, g2, g3]
        f_match = fitness_sliding_window(match_seq, prev)
        # Sequence with non-matching bigrams
        nomatch_seq = [g3, g3, g3]
        f_nomatch = fitness_sliding_window(nomatch_seq, prev)
        assert f_match > f_nomatch

    def test_compute_population_fitness_with_prev(self, small_pop):
        prior = uniform_bigram_prior(DEFAULT_ALPHABET, 20, 10)
        fitnesses = compute_population_fitness(small_pop, prev_bigrams=prior)
        assert len(fitnesses) == len(small_pop)
        assert all(f >= 0 for f in fitnesses)

    def test_sliding_vs_circular_different(self, small_pop):
        """Sliding window and circular fitness should produce different values."""
        prior = uniform_bigram_prior(DEFAULT_ALPHABET, 20, 10)
        sliding = compute_population_fitness(small_pop, prev_bigrams=prior)
        circular = compute_population_fitness(small_pop, prev_bigrams=None)
        # They use different reference bigrams, so values should differ
        assert sliding != circular


class TestEvolveWithSlidingWindow:
    def test_preserves_pop_size(self, small_pop, rng):
        prior = uniform_bigram_prior(DEFAULT_ALPHABET, 20, 10)
        new_pop = evolve_generation(
            rng, small_pop, DEFAULT_ALPHABET, 0.1, 3, 0.5, 10,
            prev_bigrams=prior,
        )
        assert len(new_pop) == len(small_pop)

    def test_preserves_seq_length(self, small_pop, rng):
        prior = uniform_bigram_prior(DEFAULT_ALPHABET, 20, 10)
        new_pop = evolve_generation(
            rng, small_pop, DEFAULT_ALPHABET, 0.1, 3, 0.5, 10,
            prev_bigrams=prior,
        )
        for seq in new_pop:
            assert len(seq) == 10

    def test_multi_generation_sliding(self):
        """Run multiple generations with sliding window — smoke test."""
        rng = np.random.default_rng(42)
        pop = generate_population(rng, DEFAULT_ALPHABET, 50, 10, 0.5)
        prev_bigrams = uniform_bigram_prior(DEFAULT_ALPHABET, 50, 10)

        for _ in range(20):
            prev_bigrams = bigram_frequencies(pop)
            pop = evolve_generation(
                rng, pop, DEFAULT_ALPHABET, 0.1, 3, 0.5, 10,
                prev_bigrams=prev_bigrams,
            )
        assert len(pop) == 50
        assert all(len(seq) == 10 for seq in pop)
