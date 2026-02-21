"""Glyph atoms, alphabet, and sequence generation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import numpy as np


class Role(Enum):
    """Glyph functional roles."""
    PARTICLE = "particle"
    ROOT = "root"
    SUFFIX = "suffix"
    MODIFIER = "modifier"


@dataclass(frozen=True)
class Glyph:
    """A single symbolic atom."""
    symbol: str
    name: str
    role: Role
    weight: float = 1.0

    def __repr__(self) -> str:
        return self.symbol


# ── The Glyph Alphabet ──────────────────────────────────────────────
# ~40 glyphs from Unicode alchemical, astronomical, and misc symbols.
# Organized by role for later lexicon use (v1+).

PARTICLES = [
    Glyph("🜍", "abyss_pulse", Role.PARTICLE),
    Glyph("🜔", "fire_seed", Role.PARTICLE),
    Glyph("🜁", "air_drift", Role.PARTICLE),
    Glyph("🜄", "water_drop", Role.PARTICLE),
    Glyph("🜃", "earth_core", Role.PARTICLE),
    Glyph("🝆", "void_spark", Role.PARTICLE),
    Glyph("🝊", "flux_origin", Role.PARTICLE),
    Glyph("☉", "sol", Role.PARTICLE),
    Glyph("☽", "luna", Role.PARTICLE),
    Glyph("♃", "jupiter", Role.PARTICLE),
]

ROOTS = [
    Glyph("☍", "fractal_veil", Role.ROOT),
    Glyph("🜂", "flame_root", Role.ROOT),
    Glyph("🜊", "tide_root", Role.ROOT),
    Glyph("🜎", "void_pulse", Role.ROOT),
    Glyph("🝉", "chaos_stem", Role.ROOT),
    Glyph("♄", "saturn_bind", Role.ROOT),
    Glyph("♂", "mars_strike", Role.ROOT),
    Glyph("♀", "venus_weave", Role.ROOT),
    Glyph("☿", "mercury_flow", Role.ROOT),
    Glyph("⚶", "node_cross", Role.ROOT),
]

SUFFIXES = [
    Glyph("🜇", "flux_end", Role.SUFFIX),
    Glyph("✶", "star_seal", Role.SUFFIX),
    Glyph("🜏", "ash_fall", Role.SUFFIX),
    Glyph("🜑", "salt_close", Role.SUFFIX),
    Glyph("🝋", "drift_end", Role.SUFFIX),
    Glyph("☊", "ascend", Role.SUFFIX),
    Glyph("☋", "descend", Role.SUFFIX),
    Glyph("⚷", "key_lock", Role.SUFFIX),
    Glyph("⚹", "sextile", Role.SUFFIX),
    Glyph("☌", "conjunct", Role.SUFFIX),
]

MODIFIERS = [
    Glyph("⊕", "amplify", Role.MODIFIER),
    Glyph("⊗", "nullify", Role.MODIFIER),
    Glyph("⊘", "invert", Role.MODIFIER),
    Glyph("⊙", "focus", Role.MODIFIER),
    Glyph("⊛", "scatter", Role.MODIFIER),
    Glyph("⊜", "balance", Role.MODIFIER),
    Glyph("⊝", "diminish", Role.MODIFIER),
    Glyph("⊞", "expand", Role.MODIFIER),
    Glyph("⊟", "compress", Role.MODIFIER),
    Glyph("⊠", "bind", Role.MODIFIER),
]

# Full default alphabet (all 40 glyphs)
DEFAULT_ALPHABET: list[Glyph] = PARTICLES + ROOTS + SUFFIXES + MODIFIERS


def _zipf_base_probs(n: int) -> np.ndarray:
    """Zipf-like base distribution: rank-based frequency falloff.

    This gives glyphs a natural frequency hierarchy (like real languages).
    Glyph 0 is ~4x more likely than glyph n-1 at entropy=0.
    """
    ranks = np.arange(1, n + 1, dtype=np.float64)
    probs = 1.0 / ranks
    return probs / probs.sum()


def generate_sequence(
    rng: np.random.Generator,
    alphabet: Sequence[Glyph],
    length: int,
    entropy: float,
) -> list[Glyph]:
    """Generate a glyph sequence using entropy-tempered sampling.

    entropy=0.0: Zipf distribution (peaked, few glyphs dominate)
    entropy=1.0: uniform random (all glyphs equally likely)
    Intermediate values interpolate between Zipf and uniform.

    The Zipf base gives glyphs a natural frequency hierarchy,
    so the entropy parameter has real effect on output diversity.
    """
    n = len(alphabet)

    if entropy >= 1.0:
        # Uniform random
        indices = rng.integers(0, n, size=length)
        return [alphabet[i] for i in indices]

    if entropy <= 0.0:
        # Pure Zipf — always pick the highest-rank glyph
        return [alphabet[0]] * length

    # Interpolate: probs = (1 - entropy) * zipf + entropy * uniform
    zipf = _zipf_base_probs(n)
    uniform = np.ones(n, dtype=np.float64) / n
    probs = (1.0 - entropy) * zipf + entropy * uniform
    probs /= probs.sum()  # renormalize (should already sum to 1)

    indices = rng.choice(n, size=length, p=probs)
    return [alphabet[i] for i in indices]


def generate_population(
    rng: np.random.Generator,
    alphabet: Sequence[Glyph],
    pop_size: int,
    seq_length: int,
    entropy: float,
) -> list[list[Glyph]]:
    """Generate a population of glyph sequences."""
    return [
        generate_sequence(rng, alphabet, seq_length, entropy)
        for _ in range(pop_size)
    ]


def sequence_to_str(seq: Sequence[Glyph]) -> str:
    """Render a glyph sequence as a Unicode string."""
    return " ".join(g.symbol for g in seq)
