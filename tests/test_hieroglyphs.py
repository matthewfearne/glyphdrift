"""Tests for hieroglyphic alphabet module."""

import numpy as np
import pytest

from glyphdrift.glyph import Glyph, Role, generate_population
from glyphdrift.hieroglyphs import (
    HIEROGLYPH_ALPHABET,
    HIERO_MODIFIERS,
    HIERO_PARTICLES,
    HIERO_ROOTS,
    HIERO_SUFFIXES,
    get_alphabet,
)


class TestHieroglyphAlphabet:
    def test_total_size(self):
        assert len(HIEROGLYPH_ALPHABET) == 80

    def test_role_counts(self):
        assert len(HIERO_PARTICLES) == 20
        assert len(HIERO_ROOTS) == 20
        assert len(HIERO_SUFFIXES) == 20
        assert len(HIERO_MODIFIERS) == 20

    def test_correct_roles(self):
        for g in HIERO_PARTICLES:
            assert g.role == Role.PARTICLE
        for g in HIERO_ROOTS:
            assert g.role == Role.ROOT
        for g in HIERO_SUFFIXES:
            assert g.role == Role.SUFFIX
        for g in HIERO_MODIFIERS:
            assert g.role == Role.MODIFIER

    def test_unique_symbols(self):
        symbols = [g.symbol for g in HIEROGLYPH_ALPHABET]
        assert len(symbols) == len(set(symbols))

    def test_unique_names(self):
        names = [g.name for g in HIEROGLYPH_ALPHABET]
        assert len(names) == len(set(names))

    def test_all_glyphs_are_glyphs(self):
        for g in HIEROGLYPH_ALPHABET:
            assert isinstance(g, Glyph)

    def test_symbols_are_unicode(self):
        for g in HIEROGLYPH_ALPHABET:
            assert len(g.symbol) >= 1
            # Egyptian hieroglyphs are in supplementary plane (>= U+10000)
            assert ord(g.symbol[0]) >= 0x10000 or len(g.symbol) > 1


class TestGetAlphabet:
    def test_alchemical(self):
        alpha = get_alphabet("alchemical")
        assert len(alpha) == 40

    def test_hieroglyphic(self):
        alpha = get_alphabet("hieroglyphic")
        assert len(alpha) == 80

    def test_hieroglyphic_40(self):
        alpha = get_alphabet("hieroglyphic_40")
        assert len(alpha) == 40
        # 10 per role
        roles = [g.role for g in alpha]
        assert roles.count(Role.PARTICLE) == 10
        assert roles.count(Role.ROOT) == 10
        assert roles.count(Role.SUFFIX) == 10
        assert roles.count(Role.MODIFIER) == 10

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown alphabet"):
            get_alphabet("nonexistent")

    def test_returns_copy(self):
        a1 = get_alphabet("alchemical")
        a2 = get_alphabet("alchemical")
        assert a1 is not a2


class TestHieroglyphPopulation:
    def test_generate_with_80(self):
        rng = np.random.default_rng(42)
        alpha = get_alphabet("hieroglyphic")
        pop = generate_population(rng, alpha, 50, 10, 0.5)
        assert len(pop) == 50
        for seq in pop:
            assert len(seq) == 10
            for g in seq:
                assert g in alpha

    def test_generate_with_40(self):
        rng = np.random.default_rng(42)
        alpha = get_alphabet("hieroglyphic_40")
        pop = generate_population(rng, alpha, 50, 10, 0.5)
        assert len(pop) == 50
        for seq in pop:
            for g in seq:
                assert g in alpha

    def test_no_overlap_with_alchemical(self):
        """Hieroglyphic and alchemical alphabets should have zero symbol overlap."""
        alch_symbols = {g.symbol for g in get_alphabet("alchemical")}
        hiero_symbols = {g.symbol for g in get_alphabet("hieroglyphic")}
        assert len(alch_symbols & hiero_symbols) == 0
