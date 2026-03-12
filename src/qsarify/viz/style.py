"""Accessible visual style for QSARify plots.

Implements the Okabe-Ito colorblind-friendly palette as the default color
cycle and provides utilities for working with colors in OKLCH (perceptually
uniform polar) space and CMYK (print) space.

References
----------
Okabe, M. & Ito, K. (2008).  Color Universal Design (CUD) — How to make
figures and presentations that are friendly to colorblind people.
https://jfly.uni-koeln.de/color/

Ottosson, B. (2020).  A perceptual color space for image processing.
https://bottosson.github.io/posts/oklab/

NFR4.1
------
All generated plots use this high-contrast, colorblind-friendly palette by
default.  OKLCH is used as the internal color model for any programmatic
contrast adjustment.  Conversion utilities to RGBA (for Matplotlib) and CMYK
(for print export) are provided.
"""

from __future__ import annotations

import math
from typing import Final

import matplotlib as mpl
from cycler import cycler as mpl_cycler
import numpy as np

__all__ = [
    "OKABE_ITO",
    "PALETTE",
    "apply_qsarify_style",
    "hex_to_rgba",
    "oklch_to_rgba",
    "rgba_to_cmyk",
    "rgba_to_oklch",
]


# ---------------------------------------------------------------------------
# Okabe-Ito palette — 8 colors (sRGB hex)
# ---------------------------------------------------------------------------

#: Okabe-Ito colorblind-friendly palette, keyed by descriptive name.
#: Colors distinguish deuteranopia, protanopia, and tritanopia; they also
#: remain distinct in grayscale.
OKABE_ITO: Final[dict[str, str]] = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
}

#: Ordered list of Okabe-Ito colors as normalized RGBA tuples, ready for
#: direct use as a Matplotlib color cycle.  Populated at bottom of module
#: after hex_to_rgba is defined.
PALETTE: list[tuple[float, float, float, float]] = []


# ---------------------------------------------------------------------------
# sRGB ↔ linear RGB (gamma encode / decode)
# ---------------------------------------------------------------------------


def _srgb_to_linear(c: float) -> float:
    """Decode a single sRGB channel value [0, 1] to linear light."""
    if c <= 0.04045:
        return c / 12.92
    return float(((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c: float) -> float:
    """Encode a single linear-light channel value [0, 1] to sRGB."""
    c = max(0.0, min(1.0, c))
    if c <= 0.0031308:
        return 12.92 * c
    return float(1.055 * c ** (1.0 / 2.4) - 0.055)


# ---------------------------------------------------------------------------
# Linear sRGB ↔ CIE XYZ (D65)
# ---------------------------------------------------------------------------

# sRGB primaries, D65 white point (IEC 61966-2-1)
_M_RGB_TO_XYZ: Final = np.array(
    [
        [0.4123907992659595, 0.357584339383878, 0.1804807884018343],
        [0.21263900587151027, 0.715168678767756, 0.07219231536073371],
        [0.01933081871559182, 0.11919477979462598, 0.9505321522496607],
    ],
    dtype=np.float64,
)

_M_XYZ_TO_RGB: Final = np.array(
    [
        [3.2409699419045226, -1.5373831775700939, -0.4986107602930034],
        [-0.9692436362808796, 1.8759675015077202, 0.04155505740717559],
        [0.05563007981249285, -0.20397695888897654, 1.0569715142428786],
    ],
    dtype=np.float64,
)


# ---------------------------------------------------------------------------
# CIE XYZ ↔ OKLab (Björn Ottosson, 2020)
# ---------------------------------------------------------------------------

# Step 1: XYZ → LMS (non-linear cone-like space)
_M1: Final = np.array(
    [
        [0.8189330101, 0.3618667424, -0.1288597137],
        [0.0329845436, 0.9293118715, 0.0361456387],
        [0.0482003018, 0.2643662691, 0.6338517070],
    ],
    dtype=np.float64,
)

# Step 2: LMS^(1/3) → OKLab
_M2: Final = np.array(
    [
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660],
    ],
    dtype=np.float64,
)

# Inverses (pre-computed; verified against numpy.linalg.inv)
_M1_INV: Final = np.array(
    [
        [1.2270138511035211, -0.5577999806518222, 0.2812561489664678],
        [-0.0405801784232806, 1.1122568696168302, -0.0716766786656012],
        [-0.0763812845057069, -0.4214819784180127, 1.5861632204407947],
    ],
    dtype=np.float64,
)

_M2_INV: Final = np.array(
    [
        [1.0, 0.3963377774, 0.2158037573],
        [1.0, -0.1055613458, -0.0638541728],
        [1.0, -0.0894841775, -1.2914855480],
    ],
    dtype=np.float64,
)


# ---------------------------------------------------------------------------
# Public color conversion utilities
# ---------------------------------------------------------------------------


def hex_to_rgba(
    hex_color: str, alpha: float = 1.0
) -> tuple[float, float, float, float]:
    """Convert a CSS hex color string to a normalized RGBA tuple.

    Parameters
    ----------
    hex_color : str
        Six-character hex color, with or without leading ``'#'``.
        E.g. ``"#E69F00"`` or ``"e69f00"``.
    alpha : float, optional
        Alpha channel value in [0, 1].  Default 1.0 (fully opaque).

    Returns
    -------
    tuple[float, float, float, float]
        ``(R, G, B, A)`` each in [0, 1].
    """
    h = hex_color.lstrip("#")
    r = int(h[0:2], 16) / 255.0
    g = int(h[2:4], 16) / 255.0
    b = int(h[4:6], 16) / 255.0
    return (r, g, b, alpha)


def rgba_to_oklch(
    r: float, g: float, b: float, a: float = 1.0
) -> tuple[float, float, float]:
    """Convert an sRGB color to OKLCH (perceptually uniform polar space).

    Conversion path: sRGB → linear RGB → CIE XYZ → OKLab → OKLCH.

    Parameters
    ----------
    r, g, b : float
        sRGB channel values in [0, 1].
    a : float, optional
        Alpha (ignored in color conversion).

    Returns
    -------
    tuple[float, float, float]
        ``(L, C, H)`` where:

        - *L* ∈ [0, 1] — perceived lightness
        - *C* ≥ 0     — chroma (colorfulness)
        - *H* ∈ [0, 360) — hue angle in degrees
    """
    # sRGB → linear RGB
    r_lin = _srgb_to_linear(r)
    g_lin = _srgb_to_linear(g)
    b_lin = _srgb_to_linear(b)
    rgb_lin = np.array([r_lin, g_lin, b_lin], dtype=np.float64)

    # linear RGB → XYZ
    xyz = _M_RGB_TO_XYZ @ rgb_lin

    # XYZ → LMS
    lms = _M1 @ xyz

    # LMS → LMS^(1/3)  (cube root, preserving sign for numerical safety)
    lms_cbrt = np.cbrt(lms)

    # LMS^(1/3) → OKLab
    lab = _M2 @ lms_cbrt

    # OKLab → OKLCH
    L_val = float(lab[0])
    a_val = float(lab[1])
    b_val = float(lab[2])

    C = math.sqrt(a_val**2 + b_val**2)
    H_rad = math.atan2(b_val, a_val)
    H_deg = math.degrees(H_rad) % 360.0  # normalise to [0, 360)

    return (L_val, C, H_deg)


def oklch_to_rgba(
    L: float, C: float, H: float, alpha: float = 1.0
) -> tuple[float, float, float, float]:
    """Convert an OKLCH color to sRGB RGBA.

    Conversion path: OKLCH → OKLab → CIE XYZ → linear RGB → sRGB.

    Parameters
    ----------
    L : float
        Perceived lightness in [0, 1].
    C : float
        Chroma (colorfulness), ≥ 0.
    H : float
        Hue angle in degrees [0, 360).
    alpha : float, optional
        Alpha channel value.  Default 1.0.

    Returns
    -------
    tuple[float, float, float, float]
        ``(R, G, B, A)`` each clamped to [0, 1].
    """
    # OKLCH → OKLab
    H_rad = math.radians(H)
    a_val = C * math.cos(H_rad)
    b_val = C * math.sin(H_rad)
    lab = np.array([L, a_val, b_val], dtype=np.float64)

    # OKLab → LMS^(1/3)
    lms_cbrt = _M2_INV @ lab

    # LMS^(1/3) → LMS  (cube)
    lms = lms_cbrt**3

    # LMS → XYZ
    xyz = _M1_INV @ lms

    # XYZ → linear RGB
    rgb_lin = _M_XYZ_TO_RGB @ xyz

    # linear RGB → sRGB (with clamping)
    r = _linear_to_srgb(float(rgb_lin[0]))
    g = _linear_to_srgb(float(rgb_lin[1]))
    b = _linear_to_srgb(float(rgb_lin[2]))

    return (r, g, b, max(0.0, min(1.0, alpha)))


def rgba_to_cmyk(
    r: float, g: float, b: float, a: float = 1.0
) -> tuple[float, float, float, float]:
    """Convert an sRGB color to CMYK (subtractive color model for print).

    Parameters
    ----------
    r, g, b : float
        sRGB channel values in [0, 1].
    a : float, optional
        Alpha (ignored; CMYK has no alpha channel).

    Returns
    -------
    tuple[float, float, float, float]
        ``(C, M, Y, K)`` each in [0, 1]:

        - *C* = Cyan
        - *M* = Magenta
        - *Y* = Yellow
        - *K* = Key (black)
    """
    k = 1.0 - max(r, g, b)
    if k >= 1.0:
        return (0.0, 0.0, 0.0, 1.0)
    denom = 1.0 - k
    c = (1.0 - r - k) / denom
    m = (1.0 - g - k) / denom
    y = (1.0 - b - k) / denom
    return (
        max(0.0, min(1.0, c)),
        max(0.0, min(1.0, m)),
        max(0.0, min(1.0, y)),
        max(0.0, min(1.0, k)),
    )


# ---------------------------------------------------------------------------
# Matplotlib / Seaborn style application
# ---------------------------------------------------------------------------


def apply_qsarify_style() -> None:
    """Apply QSARify's accessible, publication-quality Matplotlib style.

    Sets:

    - Color cycle to the Okabe-Ito palette.
    - White figure and axes backgrounds for high contrast.
    - Clean, minimal spine layout (top and right spines removed).
    - Readable font sizes suitable for publication figures.
    - Tight layout as default.

    Call this function once at the start of a plotting session.  The style
    is global and affects all subsequent ``matplotlib`` figures.

    Examples
    --------
    >>> from qsarify.viz.style import apply_qsarify_style
    >>> apply_qsarify_style()
    """
    # Color cycle: Okabe-Ito (excluding black as default cycle start to avoid
    # invisible points on white backgrounds; black is still available explicitly)
    color_cycle = [
        OKABE_ITO[k]
        for k in (
            "orange",
            "sky_blue",
            "bluish_green",
            "blue",
            "vermillion",
            "reddish_purple",
            "yellow",
            "black",
        )
    ]

    mpl.rcParams.update(
        {
            # Color cycle
            "axes.prop_cycle": mpl_cycler(color=color_cycle),
            # Backgrounds
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            # Spines
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#333333",
            # Grid
            "axes.grid": True,
            "grid.color": "#E0E0E0",
            "grid.linewidth": 0.5,
            "grid.alpha": 0.7,
            # Font sizes
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            # Lines and markers
            "lines.linewidth": 1.5,
            "lines.markersize": 5,
            # DPI
            "figure.dpi": 100,
            "savefig.dpi": 300,
            # Layout
            "figure.autolayout": True,
            "savefig.bbox": "tight",
            # Font family (prefer a system font that renders cleanly at small sizes)
            "font.family": "sans-serif",
        }
    )


# ---------------------------------------------------------------------------
# Patch PALETTE definition (forward reference resolved after hex_to_rgba)
# ---------------------------------------------------------------------------
# Re-assign here so PALETTE is populated correctly (the list-comprehension at
# module level would fail before hex_to_rgba is defined under some import
# orderings).  We clear and repopulate in-place.

PALETTE.clear()
PALETTE.extend(hex_to_rgba(h) for h in OKABE_ITO.values())
