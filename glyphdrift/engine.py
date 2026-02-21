"""Core simulation engine for GlyphDrift."""

from __future__ import annotations

import numpy as np

from .config import GlyphDriftConfig
from .dialect import (
    DEFAULT_DIALECTS,
    generate_dialect_population,
    mean_pairwise_jaccard,
)
from .drift import LogisticDrift
from .evolution import compute_population_fitness, evolve_generation
from .glyph import DEFAULT_ALPHABET, Glyph, generate_population
from .grammar import grammar_summary
from .lexicon import Lexicon, generate_structured_population, template_adherence_rate
from .metrics import GenerationMetrics, RunMetrics, compute_generation_metrics


def _generate_pop(
    rng: np.random.Generator,
    config: GlyphDriftConfig,
    lexicon: Lexicon | None,
) -> tuple[list[list[Glyph]], list[str] | None, float]:
    """Generate a population using the appropriate method.

    Returns (population, dialect_assignments, mean_foreign_frac).
    dialect_assignments and mean_foreign_frac are None/0.0 when not using dialects.
    """
    if config.use_dialects:
        pop, assignments, foreign_frac = generate_dialect_population(
            rng, DEFAULT_DIALECTS, config.population_size,
            config.sequence_length, config.entropy, config.dialect_mixing,
        )
        return pop, assignments, foreign_frac

    if config.use_lexicon and lexicon is not None:
        pop = generate_structured_population(
            rng, lexicon, config.population_size,
            config.sequence_length, config.entropy,
        )
        return pop, None, 0.0

    pop = generate_population(
        rng, DEFAULT_ALPHABET, config.population_size,
        config.sequence_length, config.entropy,
    )
    return pop, None, 0.0


def run_simulation(
    config: GlyphDriftConfig,
    scenario: str = "",
) -> RunMetrics:
    """Run a complete GlyphDrift simulation.

    Returns RunMetrics with per-generation time series.
    """
    rng = np.random.default_rng(config.seed)
    lexicon = Lexicon() if config.use_lexicon else None

    # Generate initial population
    population, dialect_assignments, foreign_frac = _generate_pop(rng, config, lexicon)

    gen_metrics: list[GenerationMetrics] = []
    foreign_fracs: list[float] = []

    # Metric sampling: collect every sample_interval generations.
    # For evolution (v3+), sample every generation to track trajectories.
    # For no-evolution, each generation is i.i.d. — 100 samples is sufficient.
    if config.use_evolution:
        n_gens = config.generations
    else:
        n_gens = min(config.generations, 100)

    fitness_series: list[float] = []

    # v4: Set up logistic drift if enabled
    drift: LogisticDrift | None = None
    if config.use_chaotic_drift and config.use_evolution:
        drift = LogisticDrift(
            r=config.logistic_r,
            x=config.logistic_x0,
            base_rate=config.mutation_rate,
            storm_interval=config.storm_interval,
        )

    for gen in range(n_gens):
        metrics = compute_generation_metrics(population, gen)
        gen_metrics.append(metrics)
        if config.use_dialects:
            foreign_fracs.append(foreign_frac)

        if config.use_evolution:
            # Track mean fitness
            fitnesses = compute_population_fitness(population)
            fitness_series.append(float(np.mean(fitnesses)))

            # Determine mutation rate (constant or chaotic)
            mut_rate = drift.step(gen) if drift else config.mutation_rate

            # Evolve to next generation
            population = evolve_generation(
                rng, population, DEFAULT_ALPHABET,
                mut_rate, config.tournament_size,
                config.crossover_rate, config.sequence_length,
            )
        else:
            # No evolution: regenerate each generation (i.i.d. sampling)
            population, dialect_assignments, foreign_frac = _generate_pop(
                rng, config, lexicon,
            )

    # Final generation metrics
    final_metrics = compute_generation_metrics(population, n_gens)
    gen_metrics.append(final_metrics)
    if config.use_dialects:
        foreign_fracs.append(foreign_frac)

    result = RunMetrics.from_generation_metrics(
        gen_metrics,
        scenario=scenario,
        seed=config.seed or 0,
        config_entropy=config.entropy,
    )
    result.population_size = config.population_size
    result.sequence_length = config.sequence_length

    # v1+ metrics
    if config.use_lexicon:
        result.template_adherence = template_adherence_rate(population)

    # v2+ metrics
    if config.use_dialects:
        result.mean_jaccard = mean_pairwise_jaccard(DEFAULT_DIALECTS)
        result.mean_foreign_frac = float(np.mean(foreign_fracs)) if foreign_fracs else 0.0

    # v3+ metrics
    if config.use_evolution and fitness_series:
        result.mean_fitness = float(np.mean(fitness_series))
        result.final_fitness = fitness_series[-1]
        result.fitness_series = fitness_series

    # v4+ metrics
    if drift:
        result.mean_mutation_rate = drift.mean_rate()
        result.mutation_rate_variance = drift.rate_variance()
        result.mutation_rate_series = drift.rate_history

    # v5+ metrics: grammar detection on final population
    if config.track_grammar:
        gs = grammar_summary(population)
        result.significant_bigrams = int(gs["significant_bigrams"])
        result.significant_trigrams = int(gs["significant_trigrams"])
        result.mutual_information = gs["mutual_information"]
        result.compression_ratio = gs["compression_ratio"]

    return result
