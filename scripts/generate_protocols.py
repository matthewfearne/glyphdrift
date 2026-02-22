"""Generate communication protocols from independent GlyphDrift evolutions.

Runs N independent evolutionary simulations and exports each final population
as a communication protocol for use in ChaosPot.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from glyphdrift.config import GlyphDriftConfig
from glyphdrift.engine import run_simulation
from glyphdrift.glyph import generate_population
from glyphdrift.hieroglyphs import get_alphabet
from glyphdrift.evolution import evolve_generation, bigram_frequencies, uniform_bigram_prior
from glyphdrift.drift import LogisticDrift
from glyphdrift.protocol import CommunicationProtocol, export_protocol, save_protocols


def generate_evolved_population(
    seed: int,
    alphabet_name: str = "alchemical",
    generations: int = 1000,
    pop_size: int = 100,
    seq_length: int = 10,
    entropy: float = 0.5,
    mutation_rate: float = 0.1,
    tournament_size: int = 3,
    crossover_rate: float = 0.5,
) -> list:
    """Run an independent evolution and return the final population."""
    rng = np.random.default_rng(seed)
    alphabet = get_alphabet(alphabet_name)
    population = generate_population(rng, alphabet, pop_size, seq_length, entropy)

    # Use sliding window fitness
    prev_bigrams = uniform_bigram_prior(alphabet, pop_size, seq_length)

    # Chaotic drift
    drift = LogisticDrift(r=3.9, x=0.5, base_rate=mutation_rate)

    for gen in range(generations):
        mut_rate = drift.step(gen)
        current_bigrams = bigram_frequencies(population)
        population = evolve_generation(
            rng, population, alphabet,
            mut_rate, tournament_size, crossover_rate, seq_length,
            prev_bigrams=prev_bigrams,
        )
        prev_bigrams = current_bigrams

    return population


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate GlyphDrift protocols")
    parser.add_argument("--n-protocols", type=int, default=4)
    parser.add_argument("--output", default="protocols.json")
    parser.add_argument("--alphabet", default="alchemical")
    parser.add_argument("--generations", type=int, default=1000)
    parser.add_argument("--top-n", type=int, default=10)
    args = parser.parse_args()

    protocols: list[CommunicationProtocol] = []
    seeds = [100, 200, 300, 400][:args.n_protocols]

    for i, seed in enumerate(seeds):
        print(f"Evolving protocol {i} (seed={seed})...")
        pop = generate_evolved_population(
            seed=seed,
            alphabet_name=args.alphabet,
            generations=args.generations,
        )
        proto = export_protocol(
            pop,
            protocol_id=i,
            top_n=args.top_n,
            scenario=f"evolved_{args.alphabet}",
            seed=seed,
        )
        protocols.append(proto)
        print(f"  Top bigrams: {proto.top_bigrams[:3]}...")
        print(f"  Compression: {proto.compression_ratio:.4f}")

    # Print similarity matrix
    print(f"\nSimilarity matrix:")
    for i, p1 in enumerate(protocols):
        row = [f"{p1.similarity(p2):.3f}" for p2 in protocols]
        print(f"  P{i}: {' '.join(row)}")

    save_protocols(protocols, args.output)
    print(f"\nSaved {len(protocols)} protocols to {args.output}")


if __name__ == "__main__":
    main()
