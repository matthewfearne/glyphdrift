"""Tests for glyph module."""

import numpy as np
import pytest

from glyphdrift.glyph import (
    DEFAULT_ALPHABET,
    MODIFIERS,
    PARTICLES,
    ROOTS,
    SUFFIXES,
    Glyph,
    Role,
    generate_population,
    generate_sequence,
    sequence_to_str,
)


class TestGlyphAlphabet:
    def test_alphabet_size(self):
        assert len(DEFAULT_ALPHABET) == 40

    def test_role_counts(self):
        assert len(PARTICLES) == 10
        assert len(ROOTS) == 10
        assert len(SUFFIXES) == 10
        assert len(MODIFIERS) == 10

    def test_unique_symbols(self):
        symbols = [g.symbol for g in DEFAULT_ALPHABET]
        assert len(symbols) == len(set(symbols))

    def test_unique_names(self):
        names = [g.name for g in DEFAULT_ALPHABET]
        assert len(names) == len(set(names))

    def test_all_weights_positive(self):
        for g in DEFAULT_ALPHABET:
            assert g.weight > 0


class TestGenerateSequence:
    def test_correct_length(self):
        rng = np.random.default_rng(42)
        seq = generate_sequence(rng, DEFAULT_ALPHABET, 10, 0.5)
        assert len(seq) == 10

    def test_deterministic_with_seed(self):
        seq1 = generate_sequence(np.random.default_rng(42), DEFAULT_ALPHABET, 10, 0.5)
        seq2 = generate_sequence(np.random.default_rng(42), DEFAULT_ALPHABET, 10, 0.5)
        assert [g.symbol for g in seq1] == [g.symbol for g in seq2]

    def test_zero_entropy_deterministic(self):
        rng = np.random.default_rng(42)
        seq = generate_sequence(rng, DEFAULT_ALPHABET, 20, 0.0)
        # All glyphs should be the same (highest weight, all equal so first)
        assert len(set(g.symbol for g in seq)) == 1

    def test_high_entropy_diverse(self):
        rng = np.random.default_rng(42)
        seq = generate_sequence(rng, DEFAULT_ALPHABET, 100, 1.0)
        unique = len(set(g.symbol for g in seq))
        # With 100 draws from 40 glyphs at uniform, expect high diversity
        assert unique > 20

    def test_entropy_gradient(self):
        """Higher entropy should produce more diverse sequences."""
        rng_low = np.random.default_rng(42)
        rng_high = np.random.default_rng(42)
        low = generate_sequence(rng_low, DEFAULT_ALPHABET, 200, 0.1)
        high = generate_sequence(rng_high, DEFAULT_ALPHABET, 200, 0.9)
        unique_low = len(set(g.symbol for g in low))
        unique_high = len(set(g.symbol for g in high))
        assert unique_high >= unique_low

    def test_all_glyphs_from_alphabet(self):
        rng = np.random.default_rng(42)
        seq = generate_sequence(rng, DEFAULT_ALPHABET, 10, 0.5)
        for g in seq:
            assert g in DEFAULT_ALPHABET


class TestGeneratePopulation:
    def test_population_size(self):
        rng = np.random.default_rng(42)
        pop = generate_population(rng, DEFAULT_ALPHABET, 50, 10, 0.5)
        assert len(pop) == 50

    def test_sequence_lengths(self):
        rng = np.random.default_rng(42)
        pop = generate_population(rng, DEFAULT_ALPHABET, 20, 8, 0.5)
        for seq in pop:
            assert len(seq) == 8


class TestSequenceToStr:
    def test_renders_symbols(self):
        rng = np.random.default_rng(42)
        seq = generate_sequence(rng, DEFAULT_ALPHABET, 3, 0.5)
        s = sequence_to_str(seq)
        assert len(s.split()) == 3
