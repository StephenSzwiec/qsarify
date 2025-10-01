import matplotlib.pyplot as plt
import seaborn as sns

from ..utils.colors import QSARIFY_PALETTE

def set_qsarify_style():
    """
    Sets a colorblind-friendly and publication-quality style for plots.
    """
    sns.set_style("whitegrid")
    plt.rcParams.update({
        'figure.figsize': (10, 8),
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'savefig.transparent': True,
        'axes.grid': True,
        'grid.linestyle': '--',
        'grid.alpha': 0.6,
    })
    # Use the custom colorblind-friendly palette
    sns.set_palette(QSARIFY_PALETTE)
