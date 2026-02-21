"""Metrics collection for GlyphDrift runs."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from .glyph import Glyph


@dataclass
class GenerationMetrics:
    """Metrics for a single generation."""

    generation: int = 0
    shannon_entropy: float = 0.0
    diversity_ratio: float = 0.0  # unique sequences / total
    unique_sequences: int = 0
    unique_glyphs: int = 0
    most_common_glyph: str = ""
    least_common_glyph: str = ""
    glyph_counts: dict[str, int] = field(default_factory=dict)

    # Role distribution
    particle_frac: float = 0.0
    root_frac: float = 0.0
    suffix_frac: float = 0.0
    modifier_frac: float = 0.0


@dataclass
class RunMetrics:
    """Aggregated metrics for a complete run."""

    scenario: str = ""
    seed: int = 0
    generations_completed: int = 0
    population_size: int = 0
    sequence_length: int = 0
    entropy_param: float = 0.0

    # Final generation metrics
    final_shannon_entropy: float = 0.0
    final_diversity_ratio: float = 0.0
    final_unique_sequences: int = 0
    final_unique_glyphs: int = 0
    final_most_common: str = ""

    # Trajectory summaries
    mean_shannon_entropy: float = 0.0
    mean_diversity_ratio: float = 0.0

    # Lexicon metrics (v1+)
    template_adherence: float = 0.0

    # Dialect metrics (v2+)
    mean_jaccard: float = 0.0
    mean_foreign_frac: float = 0.0

    # Evolution metrics (v3+)
    mean_fitness: float = 0.0
    final_fitness: float = 0.0

    # Drift metrics (v4+)
    mean_mutation_rate: float = 0.0
    mutation_rate_variance: float = 0.0

    # Grammar metrics (v5+)
    significant_bigrams: int = 0
    significant_trigrams: int = 0
    mutual_information: float = 0.0
    compression_ratio: float = 1.0

    # Time series (for plotting)
    entropy_series: list[float] = field(default_factory=list)
    diversity_series: list[float] = field(default_factory=list)
    fitness_series: list[float] = field(default_factory=list)
    mutation_rate_series: list[float] = field(default_factory=list)

    @classmethod
    def from_generation_metrics(
        cls,
        gen_metrics: list[GenerationMetrics],
        scenario: str = "",
        seed: int = 0,
        config_entropy: float = 0.0,
    ) -> RunMetrics:
        if not gen_metrics:
            return cls(scenario=scenario, seed=seed)

        entropies = [m.shannon_entropy for m in gen_metrics]
        diversities = [m.diversity_ratio for m in gen_metrics]
        final = gen_metrics[-1]

        return cls(
            scenario=scenario,
            seed=seed,
            generations_completed=len(gen_metrics),
            population_size=0,  # set by caller
            sequence_length=0,  # set by caller
            entropy_param=config_entropy,
            final_shannon_entropy=final.shannon_entropy,
            final_diversity_ratio=final.diversity_ratio,
            final_unique_sequences=final.unique_sequences,
            final_unique_glyphs=final.unique_glyphs,
            final_most_common=final.most_common_glyph,
            mean_shannon_entropy=float(np.mean(entropies)),
            mean_diversity_ratio=float(np.mean(diversities)),
            entropy_series=entropies,
            diversity_series=diversities,
        )

    def summary_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "scenario": self.scenario,
            "seed": self.seed,
            "generations": self.generations_completed,
            "entropy_param": self.entropy_param,
            "final_shannon": f"{self.final_shannon_entropy:.4f}",
            "final_diversity": f"{self.final_diversity_ratio:.4f}",
            "mean_shannon": f"{self.mean_shannon_entropy:.4f}",
            "unique_glyphs": self.final_unique_glyphs,
            "most_common": self.final_most_common,
        }
        if self.template_adherence > 0:
            d["template_adherence"] = f"{self.template_adherence:.3f}"
        return d


def compute_generation_metrics(
    population: list[list[Glyph]],
    generation: int,
) -> GenerationMetrics:
    """Compute metrics for one generation's population."""
    if not population:
        return GenerationMetrics(generation=generation)

    # Flatten all glyphs
    all_glyphs = [g for seq in population for g in seq]
    total = len(all_glyphs)

    if total == 0:
        return GenerationMetrics(generation=generation)

    # Glyph frequency counts
    counts = Counter(g.symbol for g in all_glyphs)
    freqs = np.array(list(counts.values()), dtype=np.float64)
    probs = freqs / freqs.sum()

    # Shannon entropy
    shannon = -float(np.sum(probs * np.log2(probs + 1e-10)))

    # Unique sequences
    seq_strs = set(tuple(g.symbol for g in seq) for seq in population)

    # Role distribution
    role_counts = Counter(g.role.value for g in all_glyphs)

    most_common = counts.most_common(1)[0][0]
    least_common = counts.most_common()[-1][0]

    return GenerationMetrics(
        generation=generation,
        shannon_entropy=shannon,
        diversity_ratio=len(seq_strs) / len(population),
        unique_sequences=len(seq_strs),
        unique_glyphs=len(counts),
        most_common_glyph=most_common,
        least_common_glyph=least_common,
        glyph_counts=dict(counts),
        particle_frac=role_counts.get("particle", 0) / total,
        root_frac=role_counts.get("root", 0) / total,
        suffix_frac=role_counts.get("suffix", 0) / total,
        modifier_frac=role_counts.get("modifier", 0) / total,
    )
