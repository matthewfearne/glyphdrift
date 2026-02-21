"""Emergent grammar detection for GlyphDrift (v5).

Tracks n-gram frequencies, detects statistically significant patterns
(chi-squared test), computes mutual information between adjacent positions,
and measures compression ratio as a proxy for structural complexity.
"""

from __future__ import annotations

import gzip
import math
from collections import Counter
from typing import Sequence

import numpy as np

from .glyph import Glyph


def bigram_counts(population: list[list[Glyph]]) -> Counter[tuple[str, str]]:
    """Count all bigram occurrences across the population."""
    counts: Counter[tuple[str, str]] = Counter()
    for seq in population:
        for i in range(len(seq) - 1):
            counts[(seq[i].symbol, seq[i + 1].symbol)] += 1
    return counts


def trigram_counts(population: list[list[Glyph]]) -> Counter[tuple[str, str, str]]:
    """Count all trigram occurrences across the population."""
    counts: Counter[tuple[str, str, str]] = Counter()
    for seq in population:
        for i in range(len(seq) - 2):
            counts[(seq[i].symbol, seq[i + 1].symbol, seq[i + 2].symbol)] += 1
    return counts


def significant_bigrams(
    population: list[list[Glyph]],
    p_threshold: float = 0.05,
) -> list[tuple[tuple[str, str], float]]:
    """Find bigrams appearing significantly above chance (chi-squared test).

    Under the null hypothesis (independent glyph positions), the expected
    count for bigram (a, b) is: N * P(a) * P(b)
    where N is total bigrams and P(x) is the marginal frequency of x.

    Returns list of (bigram, chi2_statistic) for significant bigrams,
    sorted by chi2 descending.
    """
    bg = bigram_counts(population)
    if not bg:
        return []

    # Marginal frequencies
    all_glyphs = [g.symbol for seq in population for g in seq]
    total_glyphs = len(all_glyphs)
    if total_glyphs == 0:
        return []

    glyph_freq = Counter(all_glyphs)
    glyph_probs = {s: c / total_glyphs for s, c in glyph_freq.items()}

    total_bigrams = sum(bg.values())
    if total_bigrams == 0:
        return []

    # Chi-squared critical value for p < 0.05, df=1
    # Using 3.841 (standard chi2 critical value)
    chi2_critical = 3.841 if p_threshold == 0.05 else _chi2_critical(p_threshold)

    significant: list[tuple[tuple[str, str], float]] = []
    for (a, b), observed in bg.items():
        expected = total_bigrams * glyph_probs.get(a, 0) * glyph_probs.get(b, 0)
        if expected < 1.0:
            continue  # Skip rare combinations
        chi2 = (observed - expected) ** 2 / expected
        if chi2 >= chi2_critical and observed > expected:
            significant.append(((a, b), chi2))

    significant.sort(key=lambda x: x[1], reverse=True)
    return significant


def significant_trigrams(
    population: list[list[Glyph]],
    p_threshold: float = 0.05,
) -> list[tuple[tuple[str, str, str], float]]:
    """Find trigrams appearing significantly above chance.

    Expected count for trigram (a, b, c) = N * P(a) * P(b) * P(c).
    """
    tg = trigram_counts(population)
    if not tg:
        return []

    all_glyphs = [g.symbol for seq in population for g in seq]
    total_glyphs = len(all_glyphs)
    if total_glyphs == 0:
        return []

    glyph_freq = Counter(all_glyphs)
    glyph_probs = {s: c / total_glyphs for s, c in glyph_freq.items()}

    total_trigrams = sum(tg.values())
    if total_trigrams == 0:
        return []

    chi2_critical = 3.841 if p_threshold == 0.05 else _chi2_critical(p_threshold)

    significant: list[tuple[tuple[str, str, str], float]] = []
    for (a, b, c), observed in tg.items():
        expected = total_trigrams * glyph_probs.get(a, 0) * glyph_probs.get(b, 0) * glyph_probs.get(c, 0)
        if expected < 1.0:
            continue
        chi2 = (observed - expected) ** 2 / expected
        if chi2 >= chi2_critical and observed > expected:
            significant.append(((a, b, c), chi2))

    significant.sort(key=lambda x: x[1], reverse=True)
    return significant


def _chi2_critical(p: float) -> float:
    """Approximate chi-squared critical value for df=1."""
    # Common values
    if p == 0.05:
        return 3.841
    if p == 0.01:
        return 6.635
    if p == 0.001:
        return 10.828
    # Rough approximation for other values
    return 3.841


def grammar_rule_count(population: list[list[Glyph]]) -> tuple[int, int]:
    """Count significant bigram and trigram 'grammar rules'.

    Returns (significant_bigrams, significant_trigrams).
    """
    sb = significant_bigrams(population)
    st = significant_trigrams(population)
    return len(sb), len(st)


def mutual_information(population: list[list[Glyph]]) -> float:
    """Compute average mutual information between adjacent glyph positions.

    MI(X_i; X_{i+1}) = sum_{x,y} P(x,y) * log2(P(x,y) / (P(x)*P(y)))

    High MI means adjacent positions are statistically dependent —
    a signal of emergent structure.
    """
    if not population:
        return 0.0

    all_glyphs = [g.symbol for seq in population for g in seq]
    total = len(all_glyphs)
    if total == 0:
        return 0.0

    # Marginal probabilities
    glyph_freq = Counter(all_glyphs)
    marginal = {s: c / total for s, c in glyph_freq.items()}

    # Joint probabilities (bigram)
    bg = bigram_counts(population)
    total_bg = sum(bg.values())
    if total_bg == 0:
        return 0.0

    joint = {pair: c / total_bg for pair, c in bg.items()}

    # MI calculation
    mi = 0.0
    for (a, b), p_ab in joint.items():
        p_a = marginal.get(a, 0)
        p_b = marginal.get(b, 0)
        if p_a > 0 and p_b > 0 and p_ab > 0:
            mi += p_ab * math.log2(p_ab / (p_a * p_b))

    return mi


def compression_ratio(population: list[list[Glyph]]) -> float:
    """Compute gzip compression ratio of the population.

    compressed_size / raw_size. Lower = more compressible = more structure.
    """
    if not population:
        return 1.0

    # Encode population as UTF-8 string
    text = "\n".join(
        " ".join(g.symbol for g in seq) for seq in population
    ).encode("utf-8")

    raw_size = len(text)
    if raw_size == 0:
        return 1.0

    compressed = gzip.compress(text, compresslevel=9)
    return len(compressed) / raw_size


def grammar_summary(population: list[list[Glyph]]) -> dict[str, float]:
    """Compute all grammar metrics for a population.

    Returns dict with:
    - significant_bigrams: count of statistically significant bigram rules
    - significant_trigrams: count of significant trigram rules
    - mutual_information: MI between adjacent positions (bits)
    - compression_ratio: gzip compressed/raw size
    - top_bigrams: list of (bigram_str, chi2) for top 5
    """
    sb = significant_bigrams(population)
    st = significant_trigrams(population)
    mi = mutual_information(population)
    cr = compression_ratio(population)

    return {
        "significant_bigrams": float(len(sb)),
        "significant_trigrams": float(len(st)),
        "mutual_information": mi,
        "compression_ratio": cr,
    }
