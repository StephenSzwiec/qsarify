"""Tests for viz/style.py — Okabe-Ito palette, OKLCH, and CMYK utilities."""

from __future__ import annotations


import pytest

from qsarify.viz.style import (
    OKABE_ITO,
    PALETTE,
    hex_to_rgba,
    oklch_to_rgba,
    rgba_to_cmyk,
    rgba_to_oklch,
)


# ---------------------------------------------------------------------------
# Palette structure
# ---------------------------------------------------------------------------


def test_okabe_ito_has_eight_colors():
    assert len(OKABE_ITO) == 8


def test_palette_list_matches_dict():
    assert len(PALETTE) == len(OKABE_ITO)


def test_okabe_ito_named_colors_present():
    for name in (
        "black",
        "orange",
        "sky_blue",
        "bluish_green",
        "yellow",
        "blue",
        "vermillion",
        "reddish_purple",
    ):
        assert name in OKABE_ITO


def test_okabe_ito_values_are_hex_strings():
    for name, val in OKABE_ITO.items():
        assert isinstance(val, str), f"{name} should be a string"
        assert val.startswith("#"), f"{name}: {val!r} should start with '#'"
        assert len(val) == 7, f"{name}: {val!r} should be 7 characters"


def test_palette_entries_are_rgba_tuples():
    for entry in PALETTE:
        assert isinstance(entry, tuple)
        assert len(entry) == 4
        r, g, b, a = entry
        assert 0.0 <= r <= 1.0
        assert 0.0 <= g <= 1.0
        assert 0.0 <= b <= 1.0
        assert a == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# hex_to_rgba
# ---------------------------------------------------------------------------


def test_hex_to_rgba_black():
    r, g, b, a = hex_to_rgba("#000000")
    assert r == pytest.approx(0.0)
    assert g == pytest.approx(0.0)
    assert b == pytest.approx(0.0)
    assert a == pytest.approx(1.0)


def test_hex_to_rgba_white():
    r, g, b, a = hex_to_rgba("#FFFFFF")
    assert r == pytest.approx(1.0)
    assert g == pytest.approx(1.0)
    assert b == pytest.approx(1.0)
    assert a == pytest.approx(1.0)


def test_hex_to_rgba_orange():
    r, g, b, a = hex_to_rgba("#E69F00")
    assert r == pytest.approx(230 / 255, abs=1e-3)
    assert g == pytest.approx(159 / 255, abs=1e-3)
    assert b == pytest.approx(0.0, abs=1e-3)
    assert a == pytest.approx(1.0)


def test_hex_to_rgba_lowercase():
    r1, g1, b1, _ = hex_to_rgba("#e69f00")
    r2, g2, b2, _ = hex_to_rgba("#E69F00")
    assert r1 == pytest.approx(r2)
    assert g1 == pytest.approx(g2)
    assert b1 == pytest.approx(b2)


# ---------------------------------------------------------------------------
# rgba_to_oklch
# ---------------------------------------------------------------------------


def test_black_oklch():
    L, C, H = rgba_to_oklch(0.0, 0.0, 0.0)
    assert L == pytest.approx(0.0, abs=1e-4)
    assert C == pytest.approx(0.0, abs=1e-4)


def test_white_oklch():
    L, C, H = rgba_to_oklch(1.0, 1.0, 1.0)
    assert L == pytest.approx(1.0, abs=1e-3)
    assert C == pytest.approx(0.0, abs=1e-3)


def test_oklch_lightness_in_range():
    for hex_color in OKABE_ITO.values():
        r, g, b, _ = hex_to_rgba(hex_color)
        L, C, H = rgba_to_oklch(r, g, b)
        assert 0.0 <= L <= 1.0, f"L={L} out of range for {hex_color}"
        assert C >= 0.0, f"C={C} < 0 for {hex_color}"
        assert 0.0 <= H <= 360.0, f"H={H} out of range for {hex_color}"


def test_okabe_ito_colors_have_distinct_hues():
    """The 7 chromatic Okabe-Ito colors (excluding black) must have distinct hues."""
    hues = []
    for name, hex_color in OKABE_ITO.items():
        if name == "black":
            continue
        r, g, b, _ = hex_to_rgba(hex_color)
        _, C, H = rgba_to_oklch(r, g, b)
        if C > 0.01:  # only chromatic colors
            hues.append(H)
    # All hues should be different (no two within 5° of each other after sorting)
    hues_sorted = sorted(hues)
    for i in range(len(hues_sorted) - 1):
        assert hues_sorted[i + 1] - hues_sorted[i] > 4.0


# ---------------------------------------------------------------------------
# oklch_to_rgba (inverse)
# ---------------------------------------------------------------------------


def test_oklch_to_rgba_black():
    r, g, b, a = oklch_to_rgba(0.0, 0.0, 0.0)
    assert r == pytest.approx(0.0, abs=1e-3)
    assert g == pytest.approx(0.0, abs=1e-3)
    assert b == pytest.approx(0.0, abs=1e-3)
    assert a == pytest.approx(1.0)


def test_oklch_to_rgba_white():
    r, g, b, a = oklch_to_rgba(1.0, 0.0, 0.0)
    assert r == pytest.approx(1.0, abs=1e-3)
    assert g == pytest.approx(1.0, abs=1e-3)
    assert b == pytest.approx(1.0, abs=1e-3)


def test_rgba_oklch_roundtrip():
    """sRGB → OKLCH → sRGB roundtrip must be accurate to within 1/255."""
    for hex_color in OKABE_ITO.values():
        r0, g0, b0, _ = hex_to_rgba(hex_color)
        L, C, H = rgba_to_oklch(r0, g0, b0)
        r1, g1, b1, _ = oklch_to_rgba(L, C, H)
        assert r1 == pytest.approx(r0, abs=1 / 255 + 1e-4), (
            f"R mismatch for {hex_color}"
        )
        assert g1 == pytest.approx(g0, abs=1 / 255 + 1e-4), (
            f"G mismatch for {hex_color}"
        )
        assert b1 == pytest.approx(b0, abs=1 / 255 + 1e-4), (
            f"B mismatch for {hex_color}"
        )


# ---------------------------------------------------------------------------
# rgba_to_cmyk
# ---------------------------------------------------------------------------


def test_cmyk_black():
    c, m, y_ch, k = rgba_to_cmyk(0.0, 0.0, 0.0)
    assert c == pytest.approx(0.0)
    assert m == pytest.approx(0.0)
    assert y_ch == pytest.approx(0.0)
    assert k == pytest.approx(1.0)


def test_cmyk_white():
    c, m, y_ch, k = rgba_to_cmyk(1.0, 1.0, 1.0)
    assert c == pytest.approx(0.0)
    assert m == pytest.approx(0.0)
    assert y_ch == pytest.approx(0.0)
    assert k == pytest.approx(0.0)


def test_cmyk_pure_red():
    # Red (1, 0, 0) → K=0, C=0, M=1, Y=1
    c, m, y_ch, k = rgba_to_cmyk(1.0, 0.0, 0.0)
    assert k == pytest.approx(0.0, abs=1e-6)
    assert c == pytest.approx(0.0, abs=1e-6)
    assert m == pytest.approx(1.0, abs=1e-6)
    assert y_ch == pytest.approx(1.0, abs=1e-6)


def test_cmyk_pure_green():
    # Green (0, 1, 0) → K=0, C=1, M=0, Y=1
    c, m, y_ch, k = rgba_to_cmyk(0.0, 1.0, 0.0)
    assert k == pytest.approx(0.0, abs=1e-6)
    assert c == pytest.approx(1.0, abs=1e-6)
    assert m == pytest.approx(0.0, abs=1e-6)
    assert y_ch == pytest.approx(1.0, abs=1e-6)


def test_cmyk_pure_blue():
    # Blue (0, 0, 1) → K=0, C=1, M=1, Y=0
    c, m, y_ch, k = rgba_to_cmyk(0.0, 0.0, 1.0)
    assert k == pytest.approx(0.0, abs=1e-6)
    assert c == pytest.approx(1.0, abs=1e-6)
    assert m == pytest.approx(1.0, abs=1e-6)
    assert y_ch == pytest.approx(0.0, abs=1e-6)


def test_cmyk_all_channels_in_range():
    for hex_color in OKABE_ITO.values():
        r, g, b, _ = hex_to_rgba(hex_color)
        c, m, y_ch, k = rgba_to_cmyk(r, g, b)
        for ch_name, ch_val in [("C", c), ("M", m), ("Y", y_ch), ("K", k)]:
            assert 0.0 <= ch_val <= 1.0, (
                f"{ch_name}={ch_val} out of [0,1] for {hex_color}"
            )


def test_cmyk_with_alpha_ignored():
    """Alpha should be ignored; result depends only on RGB channels."""
    r, g, b = 0.5, 0.3, 0.7
    c1, m1, y1, k1 = rgba_to_cmyk(r, g, b, a=1.0)
    c2, m2, y2, k2 = rgba_to_cmyk(r, g, b, a=0.5)
    assert c1 == pytest.approx(c2)
    assert m1 == pytest.approx(m2)
    assert y1 == pytest.approx(y2)
    assert k1 == pytest.approx(k2)
