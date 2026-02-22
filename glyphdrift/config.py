"""Configuration dataclasses for GlyphDrift."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GlyphDriftConfig:
    """Core configuration for a GlyphDrift run."""

    # Population
    population_size: int = 100
    sequence_length: int = 10

    # Generation control
    generations: int = 1000

    # Entropy: 0.0 = deterministic (highest weight wins), 1.0 = uniform random
    entropy: float = 0.5

    # Random seed (None = random)
    seed: int | None = None

    # Lexicon (v1+)
    use_lexicon: bool = False

    # Dialects (v2+)
    use_dialects: bool = False
    dialect_mixing: float = 0.1

    # Evolution (v3+)
    use_evolution: bool = False
    mutation_rate: float = 0.1
    tournament_size: int = 3
    crossover_rate: float = 0.5

    # Chaotic drift (v4+)
    use_chaotic_drift: bool = False
    logistic_r: float = 3.9
    logistic_x0: float = 0.5
    storm_interval: int = 0  # 0 = no storms

    # Grammar detection (v5+)
    track_grammar: bool = False

    # Non-circular fitness (v6+)
    use_sliding_window: bool = False
