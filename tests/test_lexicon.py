"""Tests for lexicon module."""

import numpy as np
import pytest

from glyphdrift.glyph import DEFAULT_ALPHABET, Role
from glyphdrift.lexicon import (
    Lexicon,
    check_template_adherence,
    generate_phrase,
    generate_structured_population,
    template_adherence_rate,
)


class TestLexicon:
    def test_default_lexicon(self):
        lex = Lexicon()
        assert len(lex.particles) == 10
        assert len(lex.roots) == 10
        assert len(lex.suffixes) == 10
        assert len(lex.modifiers) == 10

    def test_role_assignment(self):
        lex = Lexicon()
        for g in lex.particles:
            assert g.role == Role.PARTICLE
        for g in lex.roots:
            assert g.role == Role.ROOT


class TestGeneratePhrase:
    def test_returns_glyphs(self):
        rng = np.random.default_rng(42)
        lex = Lexicon()
        phrase = generate_phrase(rng, lex, 0.5)
        assert len(phrase) >= 3
        assert all(g in DEFAULT_ALPHABET for g in phrase)

    def test_target_length(self):
        rng = np.random.default_rng(42)
        lex = Lexicon()
        phrase = generate_phrase(rng, lex, 0.5, target_length=10)
        assert len(phrase) == 10

    def test_deterministic_with_seed(self):
        lex = Lexicon()
        p1 = generate_phrase(np.random.default_rng(42), lex, 0.5, target_length=10)
        p2 = generate_phrase(np.random.default_rng(42), lex, 0.5, target_length=10)
        assert [g.symbol for g in p1] == [g.symbol for g in p2]

    def test_follows_template(self):
        rng = np.random.default_rng(42)
        lex = Lexicon()
        # Generate many phrases, most should follow a template
        adherent = 0
        total = 100
        for _ in range(total):
            phrase = generate_phrase(rng, lex, 0.5, target_length=10)
            if check_template_adherence(phrase):
                adherent += 1
        # With 4 templates of 3-4 roles at start, most should adhere
        assert adherent > 50


class TestStructuredPopulation:
    def test_size(self):
        rng = np.random.default_rng(42)
        lex = Lexicon()
        pop = generate_structured_population(rng, lex, 50, 10, 0.5)
        assert len(pop) == 50
        for seq in pop:
            assert len(seq) == 10


class TestTemplateAdherence:
    def test_structured_has_adherence(self):
        rng = np.random.default_rng(42)
        lex = Lexicon()
        pop = generate_structured_population(rng, lex, 100, 10, 0.5)
        rate = template_adherence_rate(pop)
        assert rate > 0.5  # Most structured phrases should match

    def test_random_has_low_adherence(self):
        from glyphdrift.glyph import generate_population
        rng = np.random.default_rng(42)
        pop = generate_population(rng, DEFAULT_ALPHABET, 100, 10, 1.0)
        rate = template_adherence_rate(pop)
        # Random sequences have low chance of matching templates
        assert rate < 0.5
