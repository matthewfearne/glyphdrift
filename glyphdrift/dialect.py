"""Multi-dialect glyph system with cross-dialect mixing (v2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .glyph import (
    MODIFIERS,
    PARTICLES,
    ROOTS,
    SUFFIXES,
    Glyph,
    Role,
    _zipf_base_probs,
)


@dataclass
class Dialect:
    """A named subset of glyphs with its own frequency profile."""
    name: str
    glyphs: list[Glyph]

    @property
    def symbols(self) -> set[str]:
        return {g.symbol for g in self.glyphs}

    def by_role(self, role: Role) -> list[Glyph]:
        return [g for g in self.glyphs if g.role == role]


# ── The Four Dialects ──────────────────────────────────────────────
# Each dialect gets 6-7 glyphs per role (out of 10), with ~30% overlap.
# Named from the original Entropy Drift spec: Council, Veydris, Xael, Masks.

def _build_dialects() -> list[Dialect]:
    """Build 4 dialects with overlapping glyph sets."""
    dialects = []

    # Each dialect takes indices [0..6] from each role, shifted by dialect_id * 2
    # This gives ~4 shared glyphs between adjacent dialects, ~2 between distant ones
    for did, name in enumerate(["Council", "Veydris", "Xael", "Masks"]):
        glyphs: list[Glyph] = []
        for role_list in [PARTICLES, ROOTS, SUFFIXES, MODIFIERS]:
            n = len(role_list)
            # Take 7 glyphs starting at offset did*2, wrapping around
            indices = [(did * 2 + i) % n for i in range(7)]
            glyphs.extend(role_list[i] for i in indices)
        dialects.append(Dialect(name=name, glyphs=glyphs))

    return dialects


DEFAULT_DIALECTS: list[Dialect] = _build_dialects()


def jaccard_distance(d1: Dialect, d2: Dialect) -> float:
    """Jaccard distance (1 - intersection/union) between two dialects."""
    s1, s2 = d1.symbols, d2.symbols
    if not s1 and not s2:
        return 0.0
    intersection = len(s1 & s2)
    union = len(s1 | s2)
    return 1.0 - intersection / union


def mean_pairwise_jaccard(dialects: Sequence[Dialect]) -> float:
    """Mean Jaccard distance across all dialect pairs."""
    n = len(dialects)
    if n < 2:
        return 0.0
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += jaccard_distance(dialects[i], dialects[j])
            count += 1
    return total / count


def _sample_glyph(
    rng: np.random.Generator,
    glyphs: list[Glyph],
    entropy: float,
) -> Glyph:
    """Sample a single glyph with entropy tempering."""
    n = len(glyphs)
    if n <= 1:
        return glyphs[0]
    if entropy >= 1.0:
        return glyphs[rng.integers(0, n)]
    if entropy <= 0.0:
        return glyphs[0]

    zipf = _zipf_base_probs(n)
    uniform = np.ones(n, dtype=np.float64) / n
    probs = (1.0 - entropy) * zipf + entropy * uniform
    probs /= probs.sum()
    return glyphs[rng.choice(n, p=probs)]


def generate_dialect_phrase(
    rng: np.random.Generator,
    dialect: Dialect,
    all_dialects: Sequence[Dialect],
    entropy: float,
    mixing_rate: float,
    target_length: int = 10,
) -> tuple[list[Glyph], int]:
    """Generate a phrase from a dialect with cross-dialect mixing.

    Returns (phrase, foreign_count) where foreign_count is the number
    of glyphs borrowed from other dialects.
    """
    phrase: list[Glyph] = []
    foreign_count = 0

    for _ in range(target_length):
        # Cross-dialect mixing: with probability mixing_rate, sample from a foreign dialect
        if mixing_rate > 0 and rng.random() < mixing_rate:
            # Pick a random OTHER dialect
            others = [d for d in all_dialects if d.name != dialect.name]
            if others:
                foreign = others[rng.integers(0, len(others))]
                g = _sample_glyph(rng, foreign.glyphs, entropy)
                phrase.append(g)
                if g.symbol not in dialect.symbols:
                    foreign_count += 1
                continue

        # Normal: sample from own dialect
        g = _sample_glyph(rng, dialect.glyphs, entropy)
        phrase.append(g)

    return phrase, foreign_count


def generate_dialect_population(
    rng: np.random.Generator,
    dialects: Sequence[Dialect],
    pop_size: int,
    seq_length: int,
    entropy: float,
    mixing_rate: float,
) -> tuple[list[list[Glyph]], list[str], float]:
    """Generate a population distributed across dialects.

    Each sequence is assigned to a random dialect. Returns:
    (population, dialect_assignments, mean_foreign_frac)
    """
    population: list[list[Glyph]] = []
    assignments: list[str] = []
    total_foreign = 0
    total_glyphs = 0

    for _ in range(pop_size):
        # Assign to a random dialect
        did = rng.integers(0, len(dialects))
        dialect = dialects[did]
        phrase, foreign = generate_dialect_phrase(
            rng, dialect, dialects, entropy, mixing_rate, seq_length,
        )
        population.append(phrase)
        assignments.append(dialect.name)
        total_foreign += foreign
        total_glyphs += len(phrase)

    mean_foreign = total_foreign / total_glyphs if total_glyphs > 0 else 0.0
    return population, assignments, mean_foreign
