"""Batch evaluation runner with statistical analysis.

Runs N episodes per scenario, computes mean +/- 95% CI for all metrics,
saves structured results to JSON. Same pattern as ChaosPot.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import GlyphDriftConfig
from .engine import run_simulation
from .metrics import RunMetrics
from .scenarios import SCENARIOS


@dataclass
class MetricSummary:
    """Mean +/- 95% CI for a single metric."""

    mean: float
    std: float
    ci_95: float
    min: float
    max: float
    n: int

    def __str__(self) -> str:
        return f"{self.mean:.4f} +/- {self.ci_95:.4f}"


@dataclass
class EvaluationResult:
    """Aggregated results from a batch evaluation."""

    version: str
    scenario: str
    episodes: int
    generations: int
    timestamp: str

    # Core metrics
    final_shannon: MetricSummary | None = None
    mean_shannon: MetricSummary | None = None
    final_diversity: MetricSummary | None = None
    mean_diversity: MetricSummary | None = None
    final_unique_glyphs: MetricSummary | None = None
    final_unique_sequences: MetricSummary | None = None

    # Lexicon metrics (v1+)
    template_adherence: MetricSummary | None = None

    # Dialect metrics (v2+)
    mean_jaccard: MetricSummary | None = None
    mean_foreign_frac: MetricSummary | None = None

    # Evolution metrics (v3+)
    mean_fitness: MetricSummary | None = None
    final_fitness: MetricSummary | None = None

    # Drift metrics (v4+)
    mean_mutation_rate: MetricSummary | None = None
    mutation_rate_variance: MetricSummary | None = None

    # Grammar metrics (v5+)
    significant_bigrams: MetricSummary | None = None
    significant_trigrams: MetricSummary | None = None
    mutual_information_metric: MetricSummary | None = None
    compression_ratio: MetricSummary | None = None

    # v6 grammar metrics
    permutation_bigrams: MetricSummary | None = None
    position_entropy: MetricSummary | None = None
    ncd_vs_shuffled: MetricSummary | None = None


def _summarize(values: list[float]) -> MetricSummary:
    """Compute mean +/- 95% CI."""
    arr = np.array(values, dtype=np.float64)
    n = len(arr)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    ci = 1.96 * std / math.sqrt(n) if n > 1 else 0.0
    return MetricSummary(
        mean=mean, std=std, ci_95=ci,
        min=float(arr.min()), max=float(arr.max()), n=n,
    )


def evaluate(
    scenario_name: str,
    version: str = "v0",
    episodes: int = 20,
    seeds: list[int] | None = None,
    save_dir: str = "results",
) -> EvaluationResult:
    """Run batch evaluation: N episodes with fixed seeds."""
    if seeds is None:
        seeds = list(range(42, 42 + episodes))

    config = SCENARIOS[scenario_name]
    print(f"Evaluating {version}/{scenario_name}: {episodes} episodes x "
          f"{config.generations} generations")

    all_runs: list[RunMetrics] = []
    t0 = time.time()

    for i, seed in enumerate(seeds):
        cfg = GlyphDriftConfig(
            population_size=config.population_size,
            sequence_length=config.sequence_length,
            generations=config.generations,
            entropy=config.entropy,
            seed=seed,
            use_lexicon=config.use_lexicon,
            use_dialects=config.use_dialects,
            dialect_mixing=config.dialect_mixing,
            use_evolution=config.use_evolution,
            mutation_rate=config.mutation_rate,
            tournament_size=config.tournament_size,
            crossover_rate=config.crossover_rate,
            use_chaotic_drift=config.use_chaotic_drift,
            logistic_r=config.logistic_r,
            logistic_x0=config.logistic_x0,
            storm_interval=config.storm_interval,
            track_grammar=config.track_grammar,
            use_sliding_window=config.use_sliding_window,
        )
        run = run_simulation(cfg, scenario=scenario_name)
        all_runs.append(run)

        if (i + 1) % 5 == 0:
            elapsed = time.time() - t0
            print(f"  {i + 1}/{episodes} done ({elapsed:.1f}s)")

    elapsed = time.time() - t0
    print(f"  Complete in {elapsed:.1f}s")

    # Aggregate
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result = EvaluationResult(
        version=version,
        scenario=scenario_name,
        episodes=episodes,
        generations=config.generations,
        timestamp=timestamp,
        final_shannon=_summarize([r.final_shannon_entropy for r in all_runs]),
        mean_shannon=_summarize([r.mean_shannon_entropy for r in all_runs]),
        final_diversity=_summarize([r.final_diversity_ratio for r in all_runs]),
        mean_diversity=_summarize([r.mean_diversity_ratio for r in all_runs]),
        final_unique_glyphs=_summarize(
            [float(r.final_unique_glyphs) for r in all_runs]
        ),
        final_unique_sequences=_summarize(
            [float(r.final_unique_sequences) for r in all_runs]
        ),
        template_adherence=_summarize(
            [r.template_adherence for r in all_runs]
        ) if config.use_lexicon else None,
        mean_jaccard=_summarize(
            [r.mean_jaccard for r in all_runs]
        ) if config.use_dialects else None,
        mean_foreign_frac=_summarize(
            [r.mean_foreign_frac for r in all_runs]
        ) if config.use_dialects else None,
        mean_fitness=_summarize(
            [r.mean_fitness for r in all_runs]
        ) if config.use_evolution else None,
        final_fitness=_summarize(
            [r.final_fitness for r in all_runs]
        ) if config.use_evolution else None,
        mean_mutation_rate=_summarize(
            [r.mean_mutation_rate for r in all_runs]
        ) if config.use_chaotic_drift else None,
        mutation_rate_variance=_summarize(
            [r.mutation_rate_variance for r in all_runs]
        ) if config.use_chaotic_drift else None,
        significant_bigrams=_summarize(
            [float(r.significant_bigrams) for r in all_runs]
        ) if config.track_grammar else None,
        significant_trigrams=_summarize(
            [float(r.significant_trigrams) for r in all_runs]
        ) if config.track_grammar else None,
        mutual_information_metric=_summarize(
            [r.mutual_information for r in all_runs]
        ) if config.track_grammar else None,
        compression_ratio=_summarize(
            [r.compression_ratio for r in all_runs]
        ) if config.track_grammar else None,
        permutation_bigrams=_summarize(
            [float(r.permutation_bigrams) for r in all_runs]
        ) if config.track_grammar else None,
        position_entropy=_summarize(
            [r.position_entropy for r in all_runs]
        ) if config.track_grammar else None,
        ncd_vs_shuffled=_summarize(
            [r.ncd_vs_shuffled for r in all_runs]
        ) if config.track_grammar else None,
    )

    # Print table
    _print_table(result)

    # Save
    _save_result(result, save_dir)

    return result


def _print_table(result: EvaluationResult) -> None:
    """Print results table."""
    print(f"\n{'=' * 60}")
    print(f"  {result.version} / {result.scenario}")
    print(f"  {result.episodes} episodes x {result.generations} generations")
    print(f"{'=' * 60}")

    rows = [
        ("final_shannon", result.final_shannon),
        ("mean_shannon", result.mean_shannon),
        ("final_diversity", result.final_diversity),
        ("mean_diversity", result.mean_diversity),
        ("unique_glyphs", result.final_unique_glyphs),
        ("unique_sequences", result.final_unique_sequences),
        ("template_adherence", result.template_adherence),
        ("mean_jaccard", result.mean_jaccard),
        ("mean_foreign_frac", result.mean_foreign_frac),
        ("mean_fitness", result.mean_fitness),
        ("final_fitness", result.final_fitness),
        ("mean_mutation_rate", result.mean_mutation_rate),
        ("mutation_rate_var", result.mutation_rate_variance),
        ("sig_bigrams", result.significant_bigrams),
        ("sig_trigrams", result.significant_trigrams),
        ("mutual_info", result.mutual_information_metric),
        ("compression_ratio", result.compression_ratio),
        ("perm_bigrams", result.permutation_bigrams),
        ("position_entropy", result.position_entropy),
        ("ncd_vs_shuffled", result.ncd_vs_shuffled),
    ]
    for name, summary in rows:
        if summary:
            print(f"  {name:25s} {summary}")
    print()


def _save_result(result: EvaluationResult, save_dir: str) -> None:
    """Save evaluation result to JSON."""
    os.makedirs(save_dir, exist_ok=True)
    filename = f"{result.version}_{result.scenario}_{result.timestamp}.json"
    path = os.path.join(save_dir, filename)

    data: dict[str, Any] = {
        "version": result.version,
        "scenario": result.scenario,
        "episodes": result.episodes,
        "generations": result.generations,
        "timestamp": result.timestamp,
    }

    for attr in [
        "final_shannon", "mean_shannon", "final_diversity", "mean_diversity",
        "final_unique_glyphs", "final_unique_sequences", "template_adherence",
        "mean_jaccard", "mean_foreign_frac", "mean_fitness", "final_fitness",
        "mean_mutation_rate", "mutation_rate_variance",
        "significant_bigrams", "significant_trigrams",
        "mutual_information_metric", "compression_ratio",
        "permutation_bigrams", "position_entropy", "ncd_vs_shuffled",
    ]:
        summary = getattr(result, attr)
        if summary:
            data[attr] = {
                "mean": summary.mean,
                "std": summary.std,
                "ci_95": summary.ci_95,
                "min": summary.min,
                "max": summary.max,
                "n": summary.n,
            }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved: {path}")
