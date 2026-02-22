"""Communication protocol export for GlyphDrift (v8).

Exports evolved glyph populations as communication protocols that can be
used as agent identity markers in ChaosPot. Each protocol captures the
bigram vocabulary and frequency signature of an evolved population.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .evolution import bigram_frequencies
from .glyph import Glyph


@dataclass
class CommunicationProtocol:
    """An evolved communication protocol derived from a GlyphDrift population.

    Captures the top bigrams (vocabulary), glyph frequency distribution,
    and compression signature. Two protocols from the same evolutionary
    lineage will have high similarity; independent runs produce low similarity.
    """

    protocol_id: int
    top_bigrams: list[tuple[str, str]]  # Top N bigrams by frequency
    bigram_weights: list[float]         # Normalized weights for top bigrams
    glyph_freqs: dict[str, float]       # Symbol -> frequency (0-1)
    compression_ratio: float            # Compression signature
    source_scenario: str = ""
    source_seed: int = 0

    def similarity(self, other: CommunicationProtocol) -> float:
        """Jaccard similarity of top bigram sets.

        Returns 0.0 (no overlap) to 1.0 (identical vocabulary).
        """
        self_set = set(self.top_bigrams)
        other_set = set(other.top_bigrams)
        if not self_set and not other_set:
            return 1.0
        if not self_set or not other_set:
            return 0.0
        intersection = len(self_set & other_set)
        union = len(self_set | other_set)
        return intersection / union

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "protocol_id": self.protocol_id,
            "top_bigrams": [list(bg) for bg in self.top_bigrams],
            "bigram_weights": self.bigram_weights,
            "glyph_freqs": self.glyph_freqs,
            "compression_ratio": self.compression_ratio,
            "source_scenario": self.source_scenario,
            "source_seed": self.source_seed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CommunicationProtocol:
        """Deserialize from dict."""
        return cls(
            protocol_id=data["protocol_id"],
            top_bigrams=[tuple(bg) for bg in data["top_bigrams"]],
            bigram_weights=data["bigram_weights"],
            glyph_freqs=data["glyph_freqs"],
            compression_ratio=data["compression_ratio"],
            source_scenario=data.get("source_scenario", ""),
            source_seed=data.get("source_seed", 0),
        )


def export_protocol(
    population: list[list[Glyph]],
    protocol_id: int = 0,
    top_n: int = 10,
    scenario: str = "",
    seed: int = 0,
) -> CommunicationProtocol:
    """Export a population as a communication protocol.

    Extracts the top N bigrams by frequency as the protocol's vocabulary,
    along with glyph frequency distribution and compression signature.
    """
    from collections import Counter
    from .grammar import compression_ratio as compute_cr

    # Top bigrams
    bg = bigram_frequencies(population)
    total_bg = sum(bg.values())
    if total_bg == 0:
        return CommunicationProtocol(
            protocol_id=protocol_id,
            top_bigrams=[],
            bigram_weights=[],
            glyph_freqs={},
            compression_ratio=1.0,
            source_scenario=scenario,
            source_seed=seed,
        )

    sorted_bg = bg.most_common(top_n)
    top_bigrams = [(a, b) for (a, b), _ in sorted_bg]
    bigram_weights = [c / total_bg for _, c in sorted_bg]

    # Glyph frequencies
    all_glyphs = [g.symbol for seq in population for g in seq]
    total_glyphs = len(all_glyphs)
    glyph_counts = Counter(all_glyphs)
    glyph_freqs = {s: c / total_glyphs for s, c in glyph_counts.items()}

    # Compression signature
    cr = compute_cr(population)

    return CommunicationProtocol(
        protocol_id=protocol_id,
        top_bigrams=top_bigrams,
        bigram_weights=bigram_weights,
        glyph_freqs=glyph_freqs,
        compression_ratio=cr,
        source_scenario=scenario,
        source_seed=seed,
    )


def save_protocols(
    protocols: list[CommunicationProtocol],
    path: str,
) -> None:
    """Save a list of protocols to JSON."""
    data = [p.to_dict() for p in protocols]
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_protocols(path: str) -> list[CommunicationProtocol]:
    """Load protocols from JSON."""
    with open(path) as f:
        data = json.load(f)
    return [CommunicationProtocol.from_dict(d) for d in data]
