"""Hieroglyphic alphabet for GlyphDrift (v7+).

80 Egyptian hieroglyphs (20 per role), curated from the Unicode
Egyptian Hieroglyphs block (U+13000-U+1342F). Selected for visual
diversity across Gardiner's sign list categories.
"""

from __future__ import annotations

from .glyph import Glyph, Role

# ── PARTICLES (20) — People and Animals ─────────────────────────────
# Gardiner A (people) and E-G (animals, birds)

HIERO_PARTICLES = [
    Glyph("𓀀", "seated_man", Role.PARTICLE),
    Glyph("𓀃", "man_praying", Role.PARTICLE),
    Glyph("𓀔", "man_dancing", Role.PARTICLE),
    Glyph("𓀭", "king", Role.PARTICLE),
    Glyph("𓁀", "god", Role.PARTICLE),
    Glyph("𓁐", "woman", Role.PARTICLE),
    Glyph("𓃒", "bull", Role.PARTICLE),
    Glyph("𓃗", "gazelle", Role.PARTICLE),
    Glyph("𓃟", "dog", Role.PARTICLE),
    Glyph("𓃠", "jackal", Role.PARTICLE),
    Glyph("𓃭", "lion", Role.PARTICLE),
    Glyph("𓃰", "hippo", Role.PARTICLE),
    Glyph("𓃻", "ibex", Role.PARTICLE),
    Glyph("𓄁", "ox_head", Role.PARTICLE),
    Glyph("𓅃", "falcon", Role.PARTICLE),
    Glyph("𓅓", "owl", Role.PARTICLE),
    Glyph("𓅟", "goose", Role.PARTICLE),
    Glyph("𓅹", "flamingo", Role.PARTICLE),
    Glyph("𓆈", "crocodile", Role.PARTICLE),
    Glyph("𓆣", "scarab", Role.PARTICLE),
]

# ── ROOTS (20) — Body Parts and Actions ─────────────────────────────
# Gardiner D (body parts), primarily arms, legs, eyes, hands

HIERO_ROOTS = [
    Glyph("𓁶", "head", Role.ROOT),
    Glyph("𓁷", "face", Role.ROOT),
    Glyph("𓁹", "eye", Role.ROOT),
    Glyph("𓂀", "eye_of_horus", Role.ROOT),
    Glyph("𓂋", "mouth", Role.ROOT),
    Glyph("𓂑", "arm_raised", Role.ROOT),
    Glyph("𓂝", "forearm", Role.ROOT),
    Glyph("𓂧", "hand", Role.ROOT),
    Glyph("𓂭", "arm_extended", Role.ROOT),
    Glyph("𓂷", "finger", Role.ROOT),
    Glyph("𓂻", "walking_legs", Role.ROOT),
    Glyph("𓃀", "foot", Role.ROOT),
    Glyph("𓂉", "face_profile", Role.ROOT),
    Glyph("𓂓", "arms_both", Role.ROOT),
    Glyph("𓂡", "arm_bread", Role.ROOT),
    Glyph("𓂱", "two_hands", Role.ROOT),
    Glyph("𓃂", "knee", Role.ROOT),
    Glyph("𓃄", "spine", Role.ROOT),
    Glyph("𓂜", "arm_negate", Role.ROOT),
    Glyph("𓂔", "arm_vessel", Role.ROOT),
]

# ── SUFFIXES (20) — Objects and Tools ───────────────────────────────
# Gardiner R-V (temple furniture, tools, vessels, rope, containers)

HIERO_SUFFIXES = [
    Glyph("𓋹", "ankh", Role.SUFFIX),
    Glyph("𓊽", "djed", Role.SUFFIX),
    Glyph("𓌃", "mace", Role.SUFFIX),
    Glyph("𓏏", "bread", Role.SUFFIX),
    Glyph("𓊪", "stool", Role.SUFFIX),
    Glyph("𓋴", "cloth", Role.SUFFIX),
    Glyph("𓌙", "vase", Role.SUFFIX),
    Glyph("𓊖", "town", Role.SUFFIX),
    Glyph("𓊹", "star_god", Role.SUFFIX),
    Glyph("𓍯", "jar", Role.SUFFIX),
    Glyph("𓍝", "plow", Role.SUFFIX),
    Glyph("𓌳", "sickle", Role.SUFFIX),
    Glyph("𓌡", "arrow", Role.SUFFIX),
    Glyph("𓋭", "crown", Role.SUFFIX),
    Glyph("𓋔", "rope", Role.SUFFIX),
    Glyph("𓍲", "knife", Role.SUFFIX),
    Glyph("𓌈", "axe", Role.SUFFIX),
    Glyph("𓊻", "obelisk", Role.SUFFIX),
    Glyph("𓎛", "wick", Role.SUFFIX),
    Glyph("𓊝", "building", Role.SUFFIX),
]

# ── MODIFIERS (20) — Nature and Abstract ────────────────────────────
# Gardiner M-N (plants, sky/earth), Aa (unclassified signs)

HIERO_MODIFIERS = [
    Glyph("𓇋", "reed", Role.MODIFIER),
    Glyph("𓇯", "sky", Role.MODIFIER),
    Glyph("𓇳", "sun_disk", Role.MODIFIER),
    Glyph("𓇹", "moon", Role.MODIFIER),
    Glyph("𓇼", "star", Role.MODIFIER),
    Glyph("𓈎", "mountain", Role.MODIFIER),
    Glyph("𓈖", "water", Role.MODIFIER),
    Glyph("𓈗", "ripple", Role.MODIFIER),
    Glyph("𓈒", "grain", Role.MODIFIER),
    Glyph("𓉐", "house", Role.MODIFIER),
    Glyph("𓊃", "bolt", Role.MODIFIER),
    Glyph("𓇿", "land", Role.MODIFIER),
    Glyph("𓏤", "stroke", Role.MODIFIER),
    Glyph("𓏞", "book", Role.MODIFIER),
    Glyph("𓇑", "plant", Role.MODIFIER),
    Glyph("𓈉", "hill", Role.MODIFIER),
    Glyph("𓉔", "pillar", Role.MODIFIER),
    Glyph("𓆇", "frog", Role.MODIFIER),
    Glyph("𓏭", "wind", Role.MODIFIER),
    Glyph("𓆑", "viper", Role.MODIFIER),
]

# Full hieroglyphic alphabet (all 80 glyphs)
HIEROGLYPH_ALPHABET: list[Glyph] = (
    HIERO_PARTICLES + HIERO_ROOTS + HIERO_SUFFIXES + HIERO_MODIFIERS
)

# Legacy alias for the original 40-glyph alchemical set
from .glyph import DEFAULT_ALPHABET as LEGACY_ALPHABET


def get_alphabet(name: str = "alchemical") -> list[Glyph]:
    """Return the named alphabet.

    Options:
        "alchemical" — Original 40 alchemical/astronomical glyphs (default)
        "hieroglyphic" — 80 Egyptian hieroglyphs
        "hieroglyphic_40" — First 40 of the hieroglyphic set (for size-controlled A/B)
    """
    if name == "alchemical":
        return list(LEGACY_ALPHABET)
    elif name == "hieroglyphic":
        return list(HIEROGLYPH_ALPHABET)
    elif name == "hieroglyphic_40":
        # 10 per role, matching alchemical size
        return (
            HIERO_PARTICLES[:10] + HIERO_ROOTS[:10]
            + HIERO_SUFFIXES[:10] + HIERO_MODIFIERS[:10]
        )
    else:
        raise ValueError(f"Unknown alphabet: {name!r}")
