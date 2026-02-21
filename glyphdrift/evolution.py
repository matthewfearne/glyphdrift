"""Evolutionary operators for GlyphDrift (v3).

Tournament selection, bigram-coherence fitness, mutation, crossover.
This is where random generation becomes an evolutionary system.
"""

from __future__ import annotations

from collections import Counter
from typing import Sequence

import numpy as np

from .glyph import Glyph, _zipf_base_probs


# ── Fitness ──────────────────────────────────────────────────────────

def bigram_frequencies(population: list[list[Glyph]]) -> Counter[tuple[str, str]]:
    """Count all bigram occurrences across the population."""
    counts: Counter[tuple[str, str]] = Counter()
    for seq in population:
        for i in range(len(seq) - 1):
            bigram = (seq[i].symbol, seq[i + 1].symbol)
            counts[bigram] += 1
    return counts


def fitness_bigram_coherence(
    sequence: list[Glyph],
    bigram_counts: Counter[tuple[str, str]],
) -> float:
    """Fitness = sum of population-wide bigram frequencies for this sequence.

    Sequences containing common bigram patterns score higher.
    This creates implicit selection pressure for shared structure
    without coding any grammar rules.
    """
    if len(sequence) < 2:
        return 0.0
    score = 0.0
    for i in range(len(sequence) - 1):
        bigram = (sequence[i].symbol, sequence[i + 1].symbol)
        score += bigram_counts[bigram]
    return score


def compute_population_fitness(
    population: list[list[Glyph]],
) -> list[float]:
    """Compute fitness for every sequence in the population."""
    bg = bigram_frequencies(population)
    return [fitness_bigram_coherence(seq, bg) for seq in population]


# ── Selection ────────────────────────────────────────────────────────

def tournament_select(
    rng: np.random.Generator,
    population: list[list[Glyph]],
    fitnesses: list[float],
    tournament_size: int = 3,
) -> list[Glyph]:
    """Select one individual via tournament selection."""
    n = len(population)
    indices = rng.choice(n, size=min(tournament_size, n), replace=False)
    best_idx = indices[0]
    best_fit = fitnesses[best_idx]
    for idx in indices[1:]:
        if fitnesses[idx] > best_fit:
            best_idx = idx
            best_fit = fitnesses[idx]
    return list(population[best_idx])  # Return a copy


# ── Mutation ─────────────────────────────────────────────────────────

def mutate_substitution(
    rng: np.random.Generator,
    sequence: list[Glyph],
    alphabet: list[Glyph],
    rate: float,
) -> list[Glyph]:
    """Replace each glyph with probability `rate`."""
    result = list(sequence)
    for i in range(len(result)):
        if rng.random() < rate:
            result[i] = alphabet[rng.integers(0, len(alphabet))]
    return result


def mutate_insertion(
    rng: np.random.Generator,
    sequence: list[Glyph],
    alphabet: list[Glyph],
) -> list[Glyph]:
    """Insert a random glyph at a random position."""
    pos = rng.integers(0, len(sequence) + 1)
    glyph = alphabet[rng.integers(0, len(alphabet))]
    result = list(sequence)
    result.insert(pos, glyph)
    return result


def mutate_deletion(
    rng: np.random.Generator,
    sequence: list[Glyph],
) -> list[Glyph]:
    """Delete a random glyph (if length > 1)."""
    if len(sequence) <= 1:
        return list(sequence)
    pos = rng.integers(0, len(sequence))
    result = list(sequence)
    del result[pos]
    return result


def mutate(
    rng: np.random.Generator,
    sequence: list[Glyph],
    alphabet: list[Glyph],
    rate: float,
    target_length: int,
) -> list[Glyph]:
    """Apply mutation operators to a sequence.

    Substitution is the primary operator (applied per-glyph at `rate`).
    Insertion/deletion happen at rate/5 each, then we trim/pad to target_length.
    """
    result = mutate_substitution(rng, sequence, alphabet, rate)

    # Occasional insertion
    if rng.random() < rate / 5:
        result = mutate_insertion(rng, result, alphabet)

    # Occasional deletion
    if rng.random() < rate / 5:
        result = mutate_deletion(rng, result)

    # Enforce target length
    if len(result) > target_length:
        result = result[:target_length]
    while len(result) < target_length:
        result.append(alphabet[rng.integers(0, len(alphabet))])

    return result


# ── Crossover ────────────────────────────────────────────────────────

def uniform_crossover(
    rng: np.random.Generator,
    parent1: list[Glyph],
    parent2: list[Glyph],
) -> list[Glyph]:
    """Uniform crossover: each position randomly takes from parent1 or parent2."""
    length = min(len(parent1), len(parent2))
    child: list[Glyph] = []
    for i in range(length):
        if rng.random() < 0.5:
            child.append(parent1[i])
        else:
            child.append(parent2[i])
    return child


# ── Generational step ────────────────────────────────────────────────

def evolve_generation(
    rng: np.random.Generator,
    population: list[list[Glyph]],
    alphabet: list[Glyph],
    mutation_rate: float,
    tournament_size: int,
    crossover_rate: float,
    target_length: int,
) -> list[list[Glyph]]:
    """Produce the next generation via selection, crossover, and mutation.

    Returns a new population of the same size.
    """
    fitnesses = compute_population_fitness(population)
    pop_size = len(population)
    new_pop: list[list[Glyph]] = []

    for _ in range(pop_size):
        # Select parent(s)
        parent1 = tournament_select(rng, population, fitnesses, tournament_size)

        # Crossover with probability crossover_rate
        if rng.random() < crossover_rate:
            parent2 = tournament_select(rng, population, fitnesses, tournament_size)
            child = uniform_crossover(rng, parent1, parent2)
        else:
            child = list(parent1)

        # Mutate
        child = mutate(rng, child, alphabet, mutation_rate, target_length)
        new_pop.append(child)

    return new_pop
