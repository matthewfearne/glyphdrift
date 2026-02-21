"""Role-based lexicon and phrase template generation (v1)."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .glyph import (
    DEFAULT_ALPHABET,
    MODIFIERS,
    PARTICLES,
    ROOTS,
    SUFFIXES,
    Glyph,
    Role,
    _zipf_base_probs,
)


class Lexicon:
    """Organizes glyphs by role for structured phrase generation."""

    def __init__(self, glyphs: Sequence[Glyph] | None = None) -> None:
        if glyphs is None:
            glyphs = DEFAULT_ALPHABET
        self.glyphs = list(glyphs)
        self._by_role: dict[Role, list[Glyph]] = {r: [] for r in Role}
        for g in self.glyphs:
            self._by_role[g.role].append(g)

    def get_role(self, role: Role) -> list[Glyph]:
        return self._by_role[role]

    @property
    def particles(self) -> list[Glyph]:
        return self._by_role[Role.PARTICLE]

    @property
    def roots(self) -> list[Glyph]:
        return self._by_role[Role.ROOT]

    @property
    def suffixes(self) -> list[Glyph]:
        return self._by_role[Role.SUFFIX]

    @property
    def modifiers(self) -> list[Glyph]:
        return self._by_role[Role.MODIFIER]


# ── Phrase templates ──────────────────────────────────────────────────
# A phrase is a sequence of role slots. The template defines the order.
# Templates represent common "grammatical" structures that evolution
# might converge toward — but in v1, we impose them to measure effect.

TEMPLATES = [
    # particle-root-suffix (basic "sentence")
    (Role.PARTICLE, Role.ROOT, Role.SUFFIX),
    # modifier-root-suffix (modified action)
    (Role.MODIFIER, Role.ROOT, Role.SUFFIX),
    # particle-root-modifier-suffix (extended)
    (Role.PARTICLE, Role.ROOT, Role.MODIFIER, Role.SUFFIX),
    # particle-particle-root-suffix (emphasis)
    (Role.PARTICLE, Role.PARTICLE, Role.ROOT, Role.SUFFIX),
]


def _sample_from_role(
    rng: np.random.Generator,
    glyphs: list[Glyph],
    entropy: float,
) -> Glyph:
    """Sample a single glyph from a role category using entropy tempering."""
    n = len(glyphs)
    if n == 0:
        raise ValueError("Empty glyph list")
    if n == 1 or entropy <= 0.0:
        return glyphs[0]
    if entropy >= 1.0:
        return glyphs[rng.integers(0, n)]

    zipf = _zipf_base_probs(n)
    uniform = np.ones(n, dtype=np.float64) / n
    probs = (1.0 - entropy) * zipf + entropy * uniform
    probs /= probs.sum()
    return glyphs[rng.choice(n, p=probs)]


def generate_phrase(
    rng: np.random.Generator,
    lexicon: Lexicon,
    entropy: float,
    target_length: int = 0,
) -> list[Glyph]:
    """Generate a structured phrase using role templates.

    Picks a random template, fills each slot from the corresponding role
    category. If target_length > template length, pads with random-role
    glyphs. If target_length == 0, uses template length.
    """
    template = TEMPLATES[rng.integers(0, len(TEMPLATES))]
    phrase: list[Glyph] = []

    for role in template:
        candidates = lexicon.get_role(role)
        if candidates:
            phrase.append(_sample_from_role(rng, candidates, entropy))

    # Pad to target_length if needed
    if target_length > 0:
        while len(phrase) < target_length:
            # Pad with glyphs from random roles
            role = list(Role)[rng.integers(0, len(Role))]
            candidates = lexicon.get_role(role)
            if candidates:
                phrase.append(_sample_from_role(rng, candidates, entropy))

    return phrase[:target_length] if target_length > 0 else phrase


def generate_structured_population(
    rng: np.random.Generator,
    lexicon: Lexicon,
    pop_size: int,
    seq_length: int,
    entropy: float,
) -> list[list[Glyph]]:
    """Generate a population of structured phrases."""
    return [
        generate_phrase(rng, lexicon, entropy, target_length=seq_length)
        for _ in range(pop_size)
    ]


def check_template_adherence(sequence: Sequence[Glyph]) -> bool:
    """Check if a sequence matches any known template's role pattern."""
    roles = tuple(g.role for g in sequence)
    for template in TEMPLATES:
        if len(template) <= len(roles):
            if roles[:len(template)] == template:
                return True
    return False


def template_adherence_rate(population: list[list[Glyph]]) -> float:
    """Fraction of population that matches a template."""
    if not population:
        return 0.0
    matches = sum(1 for seq in population if check_template_adherence(seq))
    return matches / len(population)
