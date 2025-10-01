"""
This module defines the color palette for the qsarify package, with a focus on
accessibility and colorblind-friendliness. The colors are based on the palette
proposed by Wong, B. (2011) Points of view: Color blindness, Nature Methods, 8, 441.
"""

# --- Primary Colors ---
SKY_BLUE = "#56b4e9"  # oklch(0.7345, 0.1174, 236.18)
ORANGE = "#e69f00"    # oklch(0.7527, 0.1576, 76.77)
TEAL_GREEN = "#009e73" # oklch(0.6198 0.1295 165.46)
LEMON_YELLOW = "#f0e442" # oklch(0.9016 0.1721 105.04)
VERMILLION = "#d55e00" # oklch(0.6213 0.170473 47.5147)
PALE_VIOLET = "#cc79a7" # oklch(0.6794 0.1177 346.32)
DEEP_BLUE = "#0072b2" # oklch(0.5319 0.1313 244.05)
BLACK = "#000000"

# --- Default Palette ---
QSARIFY_PALETTE = [
    SKY_BLUE,
    ORANGE,
    TEAL_GREEN,
    LEMON_YELLOW,
    VERMILLION,
    PALE_VIOLET,
    DEEP_BLUE,
]

# --- Specific Mappings ---
TRAIN_COLOR = SKY_BLUE
TEST_COLOR = ORANGE
YSCR_R2_COLOR = TEAL_GREEN
YSCR_Q2_COLOR = LEMON_YELLOW
