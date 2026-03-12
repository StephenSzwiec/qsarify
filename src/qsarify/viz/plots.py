"""Diagnostic plots for QSAR model evaluation.

All plot functions:

- Apply the QSARify accessible style (Okabe-Ito palette, colorblind-safe).
- Accept an optional ``ax`` / ``axes`` parameter to embed into a caller's
  figure, or create their own figure when none is provided.
- Accept an optional ``save_path`` to write publication-quality PNG/SVG.
- Return the :class:`matplotlib.figure.Figure` in all cases.

Plot catalogue
--------------
- :func:`plot_residuals`   — Standardised residuals vs. fitted values.
- :func:`plot_qq`          — Normal Q-Q plot of standardised residuals.
- :func:`plot_williams`    — Williams plot (leverage vs. standardised residuals).
- :func:`plot_y_scrambling` — R² / Q² distributions from Y-scrambling.

References
----------
Williams, D. A. (1987). Generalized linear model diagnostics using the
deviance and single case deletions. Applied Statistics, 36(2), 181–191.

Roy, K. et al. (2016). A Primer on QSAR/QSPR Modeling. Springer.
"""

from __future__ import annotations

import numpy as np
import scipy.stats as scipy_stats

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from numpy.typing import NDArray

from qsarify.modeling.clustering import ClusterResult
from qsarify.results.model_result import ModelResult
from qsarify.validation.procedures import YScramblingResult
from qsarify.viz.style import OKABE_ITO, apply_qsarify_style

__all__ = [
    "plot_cluster",
    "plot_residuals",
    "plot_qq",
    "plot_williams",
    "plot_y_scrambling",
]

# Colorblind-friendly color assignments (consistent across all plots)
_COLOR_POINTS = OKABE_ITO["blue"]
_COLOR_OUTLIER = OKABE_ITO["vermillion"]
_COLOR_HLINE = OKABE_ITO["black"]
_COLOR_VLINE = OKABE_ITO["orange"]
_COLOR_HIST_SCRAM = OKABE_ITO["sky_blue"]
_COLOR_ORIG = OKABE_ITO["vermillion"]
_COLOR_REF_LINE = OKABE_ITO["bluish_green"]

# Standardised residual threshold used in Williams plot
_STD_RESID_THRESHOLD = 3.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _require_field(result: ModelResult, field: str) -> NDArray[np.float64]:
    """Return ``getattr(result, field)`` or raise :class:`ValueError`."""
    val = getattr(result, field, None)
    if val is None:
        raise ValueError(
            f"ModelResult.{field} is None.  Ensure the model was fitted and "
            f"the field was populated before calling this plot function."
        )
    return np.asarray(val, dtype=np.float64)


def _make_or_use_figure(ax: Axes | None) -> tuple[Figure, Axes]:
    """Return *(fig, ax)*, creating a new figure if *ax* is ``None``."""
    apply_qsarify_style()
    if ax is not None:
        _parent = ax.get_figure()
        if not isinstance(_parent, Figure):
            raise TypeError("ax must belong to a top-level Figure, not a SubFigure.")
        return _parent, ax
    fig, ax_new = plt.subplots(figsize=(6, 5))
    return fig, ax_new


def _save_and_return(fig: Figure, save_path: str | None) -> Figure:
    """Optionally save *fig* to *save_path* and return it."""
    if save_path is not None:
        fig.savefig(save_path)
    return fig


# ---------------------------------------------------------------------------
# Cophenetic cluster cohesion histogram
# ---------------------------------------------------------------------------


def plot_cluster(
    result: ClusterResult,
    ax: Axes | None = None,
    title: str | None = None,
    save_path: str | None = None,
) -> Figure:
    """Histogram of mean intra-cluster absolute Pearson correlation values.

    Each data point is the mean |r| for one multi-member cluster.  Singletons
    are excluded.  When no multi-member clusters exist (all singletons), an
    empty axes is returned without raising.

    Parameters
    ----------
    result : ClusterResult
        Output of :func:`~qsarify.modeling.clustering.cophenetic_cluster`.
    ax : matplotlib.axes.Axes or None, optional
        Axes to draw on.  A new figure is created when ``None``.
    title : str or None, optional
        Figure title.  Defaults to ``"Intra-Cluster Correlation Distribution"``.
    save_path : str or None, optional
        Path to save the figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax_use = _make_or_use_figure(ax)

    cohesion_values = list(result.cohesion.values())

    if cohesion_values:
        n_bins = max(5, len(cohesion_values) // 2 + 1)
        ax_use.hist(
            cohesion_values,
            bins=n_bins,
            color=_COLOR_POINTS,
            edgecolor="white",
            linewidth=0.5,
            alpha=0.85,
            range=(0.0, 1.0),
        )

    ax_use.set_xlabel("Mean intra-cluster |r|")
    ax_use.set_ylabel("Frequency")
    ax_use.set_title(title or "Intra-Cluster Correlation Distribution")
    ax_use.set_xlim(0.0, 1.0)

    return _save_and_return(fig, save_path)


# ---------------------------------------------------------------------------
# Residuals vs. fitted values
# ---------------------------------------------------------------------------


def plot_residuals(
    result: ModelResult,
    ax: Axes | None = None,
    title: str | None = None,
    save_path: str | None = None,
) -> Figure:
    """Plot standardised residuals against fitted (predicted) values.

    Useful for assessing homoscedasticity and detecting systematic bias.
    Points outside ±3 standard deviations are highlighted in a contrasting
    colour.

    Parameters
    ----------
    result : ModelResult
        A fitted model result.  Must have ``y_pred_train`` and
        ``std_residuals`` populated.
    ax : matplotlib.axes.Axes or None, optional
        Axes to draw on.  A new figure is created when ``None``.
    title : str or None, optional
        Figure title.  Defaults to ``"Residuals vs. Fitted Values"``.
    save_path : str or None, optional
        If given, save the figure to this path (format inferred from
        extension: ``.png``, ``.svg``, ``.pdf``, etc.).

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the plot.

    Raises
    ------
    ValueError
        If ``result.y_pred_train`` or ``result.std_residuals`` is ``None``.
    """
    y_pred = _require_field(result, "y_pred_train")
    std_resid = _require_field(result, "std_residuals")

    fig, ax_use = _make_or_use_figure(ax)

    # Classify points as normal vs. potential outliers
    is_outlier = np.abs(std_resid) > _STD_RESID_THRESHOLD
    is_normal = ~is_outlier

    ax_use.scatter(
        y_pred[is_normal],
        std_resid[is_normal],
        color=_COLOR_POINTS,
        alpha=0.75,
        s=40,
        zorder=3,
        label="Observations",
    )
    if is_outlier.any():
        ax_use.scatter(
            y_pred[is_outlier],
            std_resid[is_outlier],
            color=_COLOR_OUTLIER,
            alpha=0.9,
            s=55,
            marker="^",
            zorder=4,
            label=f"|z| > {_STD_RESID_THRESHOLD}",
        )

    # Reference lines
    ax_use.axhline(0.0, color=_COLOR_HLINE, linewidth=1.0, linestyle="--", zorder=2)
    ax_use.axhline(
        _STD_RESID_THRESHOLD,
        color=_COLOR_HLINE,
        linewidth=0.7,
        linestyle=":",
        alpha=0.5,
        zorder=2,
    )
    ax_use.axhline(
        -_STD_RESID_THRESHOLD,
        color=_COLOR_HLINE,
        linewidth=0.7,
        linestyle=":",
        alpha=0.5,
        zorder=2,
    )

    ax_use.set_xlabel("Fitted values")
    ax_use.set_ylabel("Standardised residuals")
    ax_use.set_title(title or "Residuals vs. Fitted Values")
    if is_outlier.any():
        ax_use.legend(frameon=False)

    return _save_and_return(fig, save_path)


# ---------------------------------------------------------------------------
# Normal Q-Q plot
# ---------------------------------------------------------------------------


def plot_qq(
    result: ModelResult,
    ax: Axes | None = None,
    title: str | None = None,
    save_path: str | None = None,
) -> Figure:
    """Normal Q-Q plot of standardised residuals.

    Displays whether residuals follow an approximately normal distribution.
    Deviations from the diagonal reference line indicate departures from
    normality.

    Parameters
    ----------
    result : ModelResult
        A fitted model result.  Must have ``std_residuals`` populated.
    ax : matplotlib.axes.Axes or None, optional
        Axes to draw on.  A new figure is created when ``None``.
    title : str or None, optional
        Figure title.  Defaults to ``"Normal Q-Q Plot"``.
    save_path : str or None, optional
        Path to save the figure.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If ``result.std_residuals`` is ``None``.
    """
    std_resid = _require_field(result, "std_residuals")

    fig, ax_use = _make_or_use_figure(ax)

    # Compute theoretical and sample quantiles
    (theoretical_q, sample_q), (slope, intercept, r) = scipy_stats.probplot(
        std_resid, dist="norm"
    )

    # Sample quantile points
    ax_use.scatter(
        theoretical_q,
        sample_q,
        color=_COLOR_POINTS,
        alpha=0.75,
        s=40,
        zorder=3,
        label="Sample quantiles",
    )

    # Reference line (fitted through the data by scipy.stats.probplot)
    x_ref = np.array([theoretical_q.min(), theoretical_q.max()])
    ax_use.plot(
        x_ref,
        slope * x_ref + intercept,
        color=_COLOR_REF_LINE,
        linewidth=1.5,
        linestyle="--",
        zorder=2,
        label="Reference line",
    )

    ax_use.set_xlabel("Theoretical quantiles")
    ax_use.set_ylabel("Sample quantiles (standardised residuals)")
    ax_use.set_title(title or "Normal Q-Q Plot")
    ax_use.legend(frameon=False)

    return _save_and_return(fig, save_path)


# ---------------------------------------------------------------------------
# Williams plot (applicability domain)
# ---------------------------------------------------------------------------


def plot_williams(
    result: ModelResult,
    ax: Axes | None = None,
    title: str | None = None,
    save_path: str | None = None,
) -> Figure:
    """Williams plot for applicability domain analysis.

    Plots standardised residuals against leverage values.  The warning
    threshold h* = 3p/n is shown as a vertical line.  Observations in the
    upper-right quadrant (high leverage AND large residual) are potentially
    influential outliers outside the applicability domain.

    Parameters
    ----------
    result : ModelResult
        A fitted model result.  Must have ``leverage``, ``std_residuals``,
        and ``leverage_threshold`` populated.
    ax : matplotlib.axes.Axes or None, optional
        Axes to draw on.  A new figure is created when ``None``.
    title : str or None, optional
        Figure title.  Defaults to ``"Williams Plot"``.
    save_path : str or None, optional
        Path to save the figure.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If ``result.leverage`` or ``result.std_residuals`` is ``None``.
    """
    leverage = _require_field(result, "leverage")
    std_resid = _require_field(result, "std_residuals")

    h_star = result.leverage_threshold
    if h_star is None:
        n = len(leverage)
        p = result.n_features if result.n_features else 1
        h_star = 3.0 * p / n

    fig, ax_use = _make_or_use_figure(ax)

    # Classify points by zone
    in_ad = (leverage <= h_star) & (np.abs(std_resid) <= _STD_RESID_THRESHOLD)
    out_ad = ~in_ad

    ax_use.scatter(
        leverage[in_ad],
        std_resid[in_ad],
        color=_COLOR_POINTS,
        alpha=0.75,
        s=40,
        zorder=3,
        label="Within AD",
    )
    if out_ad.any():
        ax_use.scatter(
            leverage[out_ad],
            std_resid[out_ad],
            color=_COLOR_OUTLIER,
            alpha=0.9,
            s=55,
            marker="^",
            zorder=4,
            label="Outside AD",
        )

    # h* vertical threshold line
    ax_use.axvline(
        h_star,
        color=_COLOR_VLINE,
        linewidth=1.5,
        linestyle="--",
        zorder=2,
        label=f"h* = {h_star:.3f}",
    )

    # ±3 horizontal threshold lines
    ax_use.axhline(
        _STD_RESID_THRESHOLD,
        color=_COLOR_HLINE,
        linewidth=0.8,
        linestyle=":",
        alpha=0.6,
        zorder=2,
    )
    ax_use.axhline(
        -_STD_RESID_THRESHOLD,
        color=_COLOR_HLINE,
        linewidth=0.8,
        linestyle=":",
        alpha=0.6,
        zorder=2,
    )
    ax_use.axhline(0.0, color=_COLOR_HLINE, linewidth=0.5, linestyle="-", alpha=0.3)

    ax_use.set_xlabel("Leverage $h_i$")
    ax_use.set_ylabel("Standardised residuals")
    ax_use.set_title(title or "Williams Plot")
    ax_use.legend(frameon=False)

    return _save_and_return(fig, save_path)


# ---------------------------------------------------------------------------
# Y-scrambling plot
# ---------------------------------------------------------------------------


def plot_y_scrambling(
    y_scram_result: YScramblingResult,
    result: ModelResult,
    axes: tuple[Axes, Axes] | None = None,
    title: str | None = None,
    save_path: str | None = None,
) -> Figure:
    """Plot R² and Q²_LOO distributions from Y-scrambling validation.

    Shows histograms of scrambled R² and Q²_LOO values alongside vertical
    lines marking the original model's values.  A valid, non-chance model
    should have its original values far to the right of the scrambled
    distributions.

    Parameters
    ----------
    y_scram_result : YScramblingResult
        Output of :func:`~qsarify.validation.procedures.run_y_scrambling`.
    result : ModelResult
        Original model result (used for the original R² / Q² reference lines,
        taken from ``y_scram_result.r2_original`` and
        ``y_scram_result.q2_original``).
    axes : tuple[Axes, Axes] or None, optional
        A pair of :class:`~matplotlib.axes.Axes` to draw on.  When ``None``,
        a new figure with two subplots (side by side) is created.
    title : str or None, optional
        Figure super-title.  Defaults to ``"Y-Scrambling Validation"``.
    save_path : str or None, optional
        Path to save the figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    apply_qsarify_style()

    if axes is not None:
        ax_r2, ax_q2 = axes
        _parent = ax_r2.get_figure()
        if not isinstance(_parent, Figure):
            raise TypeError("ax_r2 must belong to a top-level Figure, not a SubFigure.")
        fig = _parent
    else:
        fig, (ax_r2, ax_q2) = plt.subplots(1, 2, figsize=(11, 5))

    n_bins = max(10, y_scram_result.n_iterations // 10)

    # ── R² panel ──────────────────────────────────────────────────────────
    ax_r2.hist(
        y_scram_result.r2_scrambled,
        bins=n_bins,
        color=_COLOR_HIST_SCRAM,
        edgecolor="white",
        linewidth=0.5,
        alpha=0.85,
        label=f"Scrambled (n={y_scram_result.n_iterations})",
    )
    ax_r2.axvline(
        y_scram_result.r2_original,
        color=_COLOR_ORIG,
        linewidth=2.0,
        linestyle="--",
        zorder=5,
        label=f"Original R² = {y_scram_result.r2_original:.3f}",
    )
    ax_r2.set_xlabel("R² (scrambled models)")
    ax_r2.set_ylabel("Count")
    ax_r2.set_title("R² Distribution")
    ax_r2.legend(frameon=False)

    # ── Q²_LOO panel ──────────────────────────────────────────────────────
    ax_q2.hist(
        y_scram_result.q2_scrambled,
        bins=n_bins,
        color=_COLOR_HIST_SCRAM,
        edgecolor="white",
        linewidth=0.5,
        alpha=0.85,
        label=f"Scrambled (n={y_scram_result.n_iterations})",
    )
    ax_q2.axvline(
        y_scram_result.q2_original,
        color=_COLOR_ORIG,
        linewidth=2.0,
        linestyle="--",
        zorder=5,
        label=f"Original Q²_LOO = {y_scram_result.q2_original:.3f}",
    )
    ax_q2.set_xlabel("Q²_LOO (scrambled models)")
    ax_q2.set_ylabel("Count")
    ax_q2.set_title("Q²_LOO Distribution")
    ax_q2.legend(frameon=False)

    fig.suptitle(title or "Y-Scrambling Validation", fontsize=13)
    fig.tight_layout()

    return _save_and_return(fig, save_path)
