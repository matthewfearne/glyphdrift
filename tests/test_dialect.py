"""Tests for dialect module."""

import numpy as np
import pytest

from glyphdrift.dialect import (
    DEFAULT_DIALECTS,
    Dialect,
    generate_dialect_phrase,
    generate_dialect_population,
    jaccard_distance,
    mean_pairwise_jaccard,
)
from glyphdrift.glyph import Role


class TestDialectConstruction:
    def test_four_dialects(self):
        assert len(DEFAULT_DIALECTS) == 4

    def test_dialect_names(self):
        names = [d.name for d in DEFAULT_DIALECTS]
        assert names == ["Council", "Veydris", "Xael", "Masks"]

    def test_dialect_size(self):
        for d in DEFAULT_DIALECTS:
            # 7 glyphs per role × 4 roles = 28
            assert len(d.glyphs) == 28

    def test_dialects_have_all_roles(self):
        for d in DEFAULT_DIALECTS:
            roles = {g.role for g in d.glyphs}
            assert roles == {Role.PARTICLE, Role.ROOT, Role.SUFFIX, Role.MODIFIER}

    def test_by_role(self):
        d = DEFAULT_DIALECTS[0]
        for role in Role:
            role_glyphs = d.by_role(role)
            assert len(role_glyphs) == 7
            assert all(g.role == role for g in role_glyphs)

    def test_dialects_overlap(self):
        """Adjacent dialects should share glyphs, but not be identical."""
        d0, d1 = DEFAULT_DIALECTS[0], DEFAULT_DIALECTS[1]
        shared = d0.symbols & d1.symbols
        assert len(shared) > 0  # Some overlap
        assert len(shared) < len(d0.symbols)  # Not identical


class TestJaccardDistance:
    def test_identical_dialects(self):
        d = DEFAULT_DIALECTS[0]
        assert jaccard_distance(d, d) == 0.0

    def test_symmetric(self):
        d0, d1 = DEFAULT_DIALECTS[0], DEFAULT_DIALECTS[1]
        assert jaccard_distance(d0, d1) == jaccard_distance(d1, d0)

    def test_range(self):
        for i, d1 in enumerate(DEFAULT_DIALECTS):
            for d2 in DEFAULT_DIALECTS[i + 1:]:
                jd = jaccard_distance(d1, d2)
                assert 0.0 <= jd <= 1.0

    def test_empty_dialects(self):
        d1 = Dialect(name="empty1", glyphs=[])
        d2 = Dialect(name="empty2", glyphs=[])
        assert jaccard_distance(d1, d2) == 0.0

    def test_mean_pairwise(self):
        mpj = mean_pairwise_jaccard(DEFAULT_DIALECTS)
        # Should be between 0 and 1
        assert 0.0 < mpj < 1.0


class TestDialectPhraseGeneration:
    def test_phrase_length(self):
        rng = np.random.default_rng(42)
        d = DEFAULT_DIALECTS[0]
        phrase, _ = generate_dialect_phrase(rng, d, DEFAULT_DIALECTS, 0.5, 0.1, 10)
        assert len(phrase) == 10

    def test_deterministic(self):
        d = DEFAULT_DIALECTS[0]
        p1, f1 = generate_dialect_phrase(
            np.random.default_rng(42), d, DEFAULT_DIALECTS, 0.5, 0.1, 10,
        )
        p2, f2 = generate_dialect_phrase(
            np.random.default_rng(42), d, DEFAULT_DIALECTS, 0.5, 0.1, 10,
        )
        assert [g.symbol for g in p1] == [g.symbol for g in p2]
        assert f1 == f2

    def test_no_mixing_no_foreign(self):
        rng = np.random.default_rng(42)
        d = DEFAULT_DIALECTS[0]
        phrase, foreign = generate_dialect_phrase(
            rng, d, DEFAULT_DIALECTS, 0.5, 0.0, 100,
        )
        assert foreign == 0

    def test_high_mixing_has_foreign(self):
        rng = np.random.default_rng(42)
        d = DEFAULT_DIALECTS[0]
        phrase, foreign = generate_dialect_phrase(
            rng, d, DEFAULT_DIALECTS, 0.5, 0.9, 100,
        )
        # With 90% mixing and partial overlap, expect some foreign glyphs
        assert foreign > 0


class TestDialectPopulation:
    def test_population_size(self):
        rng = np.random.default_rng(42)
        pop, assignments, _ = generate_dialect_population(
            rng, DEFAULT_DIALECTS, 50, 10, 0.5, 0.1,
        )
        assert len(pop) == 50
        assert len(assignments) == 50

    def test_sequence_length(self):
        rng = np.random.default_rng(42)
        pop, _, _ = generate_dialect_population(
            rng, DEFAULT_DIALECTS, 20, 8, 0.5, 0.1,
        )
        for seq in pop:
            assert len(seq) == 8

    def test_all_dialects_assigned(self):
        rng = np.random.default_rng(42)
        _, assignments, _ = generate_dialect_population(
            rng, DEFAULT_DIALECTS, 200, 10, 0.5, 0.1,
        )
        # With 200 sequences across 4 dialects, all should appear
        assert set(assignments) == {"Council", "Veydris", "Xael", "Masks"}

    def test_foreign_frac_with_mixing(self):
        rng = np.random.default_rng(42)
        _, _, foreign_frac = generate_dialect_population(
            rng, DEFAULT_DIALECTS, 100, 10, 0.5, 0.5,
        )
        assert foreign_frac > 0

    def test_no_foreign_without_mixing(self):
        rng = np.random.default_rng(42)
        _, _, foreign_frac = generate_dialect_population(
            rng, DEFAULT_DIALECTS, 100, 10, 0.5, 0.0,
        )
        assert foreign_frac == 0.0
