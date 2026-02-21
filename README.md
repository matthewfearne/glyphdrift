# GlyphDrift: Emergent Symbolic Language Through Entropy-Driven Evolution

Can structured symbolic patterns emerge from entropy + selection pressure — without coding any grammar rules?

GlyphDrift answers this by evolving populations of symbolic sequences through bigram-coherence fitness, chaotic mutation dynamics (logistic map), and statistical grammar detection. Each version adds one variable; every number comes from 20 episodes with fixed seeds (42-61).

## Key Findings

**v3 — Evolution works.** Selection pressure reduces Shannon entropy by 64% (5.02 → 1.81 bits) and collapses vocabulary from 40 to 16 glyphs at low mutation. Diversity drops from 100% unique sequences to 18%.

**v4 — Chaos amplifies convergence.** Logistic-map mutation (r=3.9) drives 53% further entropy reduction vs constant mutation. Entropy storms create punctuated equilibrium — purges, not perturbations.

**v5 — The grammar paradox.** Chi-squared tests find 14 "grammar rules" in random populations but ZERO in evolved ones. Evolved populations compress to 7% of raw size (vs 30% for random). Standard statistical tests are blind to the most important emergent structure — vocabulary convergence. Compression-based metrics detect what chi-squared misses.

## System Architecture

```
v0: 40 Unicode glyphs (4 roles × 10) with Zipf-entropy sampling
v1: Role-based phrase templates — structure INCREASES entropy (+8.7%)
v2: 4 dialects with cross-mixing — partitioning diversifies
v3: Tournament selection + bigram fitness — convergence breaks diversity=1.0
v4: Logistic map drives mutation rate — chaotic timing matters
v5: Grammar detection reveals the paradox — convergence ≠ complexity
```

## Results Summary

| Version | Shannon Entropy | Diversity | Unique Glyphs | Key Finding |
|---------|----------------|-----------|---------------|-------------|
| v0 base | 5.024 ± 0.014 | 1.000 | 40 | Entropy parameter works |
| v1 structured | 5.175 ± 0.009 | 1.000 | 40 | Structure diversifies (+3%) |
| v2 dialects | 5.084 ± 0.010 | 1.000 | 40 | Partitioning diversifies (+1.3%) |
| v3 evolution | 1.814 ± 0.151 | 0.857 | 39 | Convergence begins (-64% entropy) |
| v3 low_mut | 0.255 ± 0.093 | 0.181 | 16 | Vocabulary collapse (-95%) |
| v4 chaotic | 0.858 ± 0.098 | 0.560 | 35 | Chaos amplifies (-53% vs v3) |
| v4 storms | 0.702 ± 0.092 | 0.447 | 31 | Storms purge, not perturb |
| v5 grammar | 0.858 ± 0.098 | 0.560 | 35 | 0 grammar rules (paradox) |
| v5 no_select | 5.199 ± 0.013 | 0.982 | 40 | 14 grammar rules (random!) |

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# Run tests (96 passing)
pytest tests/ -v

# Run a single scenario
python3 -c "
from glyphdrift.evaluation import evaluate
evaluate('grammar_base', version='v5', episodes=5)
"
```

## Project Structure

```
glyphdrift/
  glyphdrift/
    glyph.py          # 40 Unicode glyphs, Zipf-entropy sampling
    lexicon.py         # Role-based phrase templates (v1)
    dialect.py         # 4 dialects with cross-mixing (v2)
    evolution.py       # Tournament selection, bigram fitness (v3)
    drift.py           # Logistic map mutation dynamics (v4)
    grammar.py         # Chi-squared grammar detection, MI, compression (v5)
    metrics.py         # Per-generation and episode metrics
    config.py          # Configuration dataclass
    scenarios.py       # 20 named scenario configs
    evaluation.py      # Batch runner (20 runs, seeds, 95% CIs)
    engine.py          # Core simulation loop
  tests/               # 96 tests
  results/             # 23 evaluation JSONs
  VERSION_LOG.md       # Ground truth data with observations
```

## Methodology

- **Episodes:** 20 per scenario, seeds 42-61
- **Generations:** 1000 per episode
- **Statistics:** Mean ± 95% CI (1.96 × SE)
- **Max Shannon entropy:** log2(40) = 5.322 bits
- **Reproducible:** Fixed seeds, deterministic RNG

## License

MIT
