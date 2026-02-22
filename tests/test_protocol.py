"""Tests for communication protocol module."""

import json
import os
import tempfile

import numpy as np
import pytest

from glyphdrift.evolution import evolve_generation, bigram_frequencies, uniform_bigram_prior
from glyphdrift.glyph import DEFAULT_ALPHABET, generate_population
from glyphdrift.protocol import (
    CommunicationProtocol,
    export_protocol,
    load_protocols,
    save_protocols,
)


@pytest.fixture
def random_pop():
    rng = np.random.default_rng(42)
    return generate_population(rng, DEFAULT_ALPHABET, 100, 10, 0.5)


@pytest.fixture
def evolved_pop():
    rng = np.random.default_rng(42)
    pop = generate_population(rng, DEFAULT_ALPHABET, 100, 10, 0.5)
    prev = uniform_bigram_prior(DEFAULT_ALPHABET, 100, 10)
    for _ in range(200):
        current = bigram_frequencies(pop)
        pop = evolve_generation(rng, pop, DEFAULT_ALPHABET, 0.1, 3, 0.5, 10, prev)
        prev = current
    return pop


@pytest.fixture
def evolved_pop_b():
    """Independent evolution with different seed."""
    rng = np.random.default_rng(999)
    pop = generate_population(rng, DEFAULT_ALPHABET, 100, 10, 0.5)
    prev = uniform_bigram_prior(DEFAULT_ALPHABET, 100, 10)
    for _ in range(200):
        current = bigram_frequencies(pop)
        pop = evolve_generation(rng, pop, DEFAULT_ALPHABET, 0.1, 3, 0.5, 10, prev)
        prev = current
    return pop


class TestExportProtocol:
    def test_returns_protocol(self, random_pop):
        proto = export_protocol(random_pop, protocol_id=0)
        assert isinstance(proto, CommunicationProtocol)

    def test_has_bigrams(self, random_pop):
        proto = export_protocol(random_pop, protocol_id=0, top_n=10)
        assert len(proto.top_bigrams) == 10
        assert len(proto.bigram_weights) == 10

    def test_weights_sum_less_than_one(self, random_pop):
        proto = export_protocol(random_pop, protocol_id=0, top_n=10)
        assert sum(proto.bigram_weights) <= 1.0

    def test_has_glyph_freqs(self, random_pop):
        proto = export_protocol(random_pop, protocol_id=0)
        assert len(proto.glyph_freqs) > 0
        assert abs(sum(proto.glyph_freqs.values()) - 1.0) < 0.01

    def test_compression_ratio(self, random_pop):
        proto = export_protocol(random_pop, protocol_id=0)
        assert 0 < proto.compression_ratio < 1.5

    def test_empty_population(self):
        proto = export_protocol([], protocol_id=0)
        assert proto.top_bigrams == []
        assert proto.compression_ratio == 1.0


class TestSimilarity:
    def test_self_similarity(self, random_pop):
        proto = export_protocol(random_pop, protocol_id=0)
        assert proto.similarity(proto) == 1.0

    def test_same_population_same_protocol(self, random_pop):
        p1 = export_protocol(random_pop, protocol_id=0, top_n=10)
        p2 = export_protocol(random_pop, protocol_id=1, top_n=10)
        assert p1.similarity(p2) == 1.0

    def test_different_populations(self, evolved_pop, evolved_pop_b):
        p1 = export_protocol(evolved_pop, protocol_id=0, top_n=10)
        p2 = export_protocol(evolved_pop_b, protocol_id=1, top_n=10)
        sim = p1.similarity(p2)
        assert 0 <= sim <= 1.0

    def test_similarity_symmetric(self, evolved_pop, evolved_pop_b):
        p1 = export_protocol(evolved_pop, protocol_id=0, top_n=10)
        p2 = export_protocol(evolved_pop_b, protocol_id=1, top_n=10)
        assert p1.similarity(p2) == p2.similarity(p1)

    def test_empty_protocols(self):
        p1 = CommunicationProtocol(0, [], [], {}, 1.0)
        p2 = CommunicationProtocol(1, [], [], {}, 1.0)
        assert p1.similarity(p2) == 1.0


class TestSerialization:
    def test_to_dict(self, random_pop):
        proto = export_protocol(random_pop, protocol_id=0, top_n=5)
        d = proto.to_dict()
        assert d["protocol_id"] == 0
        assert len(d["top_bigrams"]) == 5

    def test_from_dict_roundtrip(self, random_pop):
        proto = export_protocol(random_pop, protocol_id=0, top_n=5,
                                scenario="test", seed=42)
        d = proto.to_dict()
        restored = CommunicationProtocol.from_dict(d)
        assert restored.protocol_id == proto.protocol_id
        assert restored.top_bigrams == proto.top_bigrams
        assert restored.bigram_weights == proto.bigram_weights
        assert restored.source_scenario == proto.source_scenario

    def test_save_load_roundtrip(self, random_pop, evolved_pop):
        p1 = export_protocol(random_pop, protocol_id=0, top_n=5)
        p2 = export_protocol(evolved_pop, protocol_id=1, top_n=5)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name

        try:
            save_protocols([p1, p2], path)
            loaded = load_protocols(path)
            assert len(loaded) == 2
            assert loaded[0].protocol_id == 0
            assert loaded[1].protocol_id == 1
            assert loaded[0].top_bigrams == p1.top_bigrams
        finally:
            os.unlink(path)

    def test_json_serializable(self, random_pop):
        proto = export_protocol(random_pop, protocol_id=0)
        d = proto.to_dict()
        # Should not raise
        json_str = json.dumps(d, ensure_ascii=False)
        assert len(json_str) > 0
