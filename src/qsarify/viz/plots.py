import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from typing import Any, Dict, Optional

from ..utils import colors, statistics
from .style import set_qsarify_style


def plot_y_scrambling(
    y_scrambling_results: Dict[str, Any],
    title: str = "Y-scrambling Plot",
    figsize: tuple[float, float] = (8.5, 8.5),
    save_path: Optional[str] = None,
):
    """
    Generates a Y-scrambling plot from the results of the y_scrambling procedure.
    """
    set_qsarify_style()

    original_metrics = y_scrambling_results["original"]
    scrambled_metrics = y_scrambling_results["scrambled"]

    fig, ax = plt.subplots(figsize=figsize)

    # Plot original model
    ax.scatter(
        original_metrics["Kxy"],
        original_metrics["R2_train"],
        s=180,
        color=colors.TRAIN_COLOR,
        label="model $R^2_{train}$",
        marker="o",
        edgecolor="black",
        linewidth=0.6,
        zorder=4,
    )
    ax.scatter(
        original_metrics["Kxy"],
        original_metrics["Q2_LOO"],
        s=180,
        color=colors.TEST_COLOR,  # Using TEST_COLOR for Q2_LOO as it's a validation metric
        label="model $Q^2_{LOO}$",
        marker="o",
        edgecolor="black",
        linewidth=0.6,
        zorder=4,
    )

    # Plot scrambled models
    ax.scatter(
        scrambled_metrics["Kxy"],
        scrambled_metrics["R2_train"],
        s=60,
        color=colors.YSCR_R2_COLOR,
        alpha=0.6,
        label="$Y_{scr}$ $R^{2}$",
        marker="o",
        edgecolor="black",
        linewidth=0.5,
        zorder=2,
    )
    ax.scatter(
        scrambled_metrics["Kxy"],
        scrambled_metrics["Q2_LOO"],
        s=60,
        color=colors.YSCR_Q2_COLOR,
        alpha=0.6,
        label="$Y_{scr}$ $Q^{2}$",
        marker="o",
        edgecolor="black",
        linewidth=0.5,
        zorder=2,
    )

    # Axes labels, limits, legend
    all_kxy = [original_metrics["Kxy"]] + scrambled_metrics["Kxy"]
    x_max = max(all_kxy)
    x_min = min(all_kxy)
    x_pad = 0.05 * (x_max - x_min if x_max > x_min else 1.0)

    all_metrics = (
        [original_metrics["R2_train"], original_metrics["Q2_LOO"]]
        + scrambled_metrics["R2_train"]
        + scrambled_metrics["Q2_LOO"]
    )
    y_min = min(-0.1, min(all_metrics) - 0.05)
    y_max = max(1.05, max(all_metrics) + 0.05)

    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min, y_max)

    ax.set_title(title, fontsize=20)
    ax.set_xlabel(r"$K_{xy}$", fontsize=20)
    ax.set_ylabel("Squared Correlation Coefficient", fontsize=20)
    ax.tick_params(axis="both", labelsize=14)

    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")

    ax.legend(
        bbox_to_anchor=(0.05, 1.0),
        loc="upper left",
        fontsize=14,
        borderaxespad=0.0,
    )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300)

    return fig, ax

def plot_residuals_vs_fitted(
    model_result: Any, # Should be ModelResult
    title: str = "Residuals vs. Fitted Values",
    figsize: tuple[float, float] = (8.5, 8.5),
    save_path: Optional[str] = None,
):
    """
    Generates a residuals vs. fitted values plot.
    """
    set_qsarify_style()

    fig, ax = plt.subplots(figsize=figsize)

    # Training data
    if model_result.y_true_train is not None and model_result.y_pred_train is not None:
        residuals_train = model_result.y_true_train - model_result.y_pred_train
        ax.scatter(
            model_result.y_pred_train,
            residuals_train,
            alpha=0.6,
            label="Training set",
            color=colors.TRAIN_COLOR,
        )

    # Test data
    if model_result.y_true_test is not None and model_result.y_pred_test is not None:
        residuals_test = model_result.y_true_test - model_result.y_pred_test
        ax.scatter(
            model_result.y_pred_test,
            residuals_test,
            alpha=0.6,
            label="Test set",
            color=colors.TEST_COLOR,
            marker="x",
        )

    ax.axhline(y=0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Fitted Values", fontsize=14)
    ax.set_ylabel("Residuals", fontsize=14)
    ax.set_title(title, fontsize=20)
    ax.legend()

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300)

    return fig, ax


def plot_qq(
    model_result: Any, # Should be ModelResult
    title: str = "Q-Q Plot of Residuals",
    figsize: tuple[float, float] = (8.5, 8.5),
    save_path: Optional[str] = None,
):
    """
    Generates a Q-Q plot of the residuals.
    """
    set_qsarify_style()

    fig, ax = plt.subplots(figsize=figsize)

    residuals = []
    if model_result.y_true_train is not None and model_result.y_pred_train is not None:
        residuals.extend(model_result.y_true_train - model_result.y_pred_train)
    
    if model_result.y_true_test is not None and model_result.y_pred_test is not None:
        residuals.extend(model_result.y_true_test - model_result.y_pred_test)

    if not residuals:
        raise ValueError("No residuals found in ModelResult.")

    stats.probplot(residuals, dist="norm", plot=ax)

    ax.set_title(title, fontsize=20)
    ax.get_lines()[0].set_markerfacecolor(colors.TRAIN_COLOR)
    ax.get_lines()[0].set_markeredgecolor(colors.TRAIN_COLOR)
    ax.get_lines()[1].set_color('black')
    ax.get_lines()[1].set_linestyle('--')


    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300)

    return fig, ax

def plot_williams(
    model_result: Any = None, # Should be ModelResult
    model: Any = None,
    X_train: Optional[pd.DataFrame] = None,
    X_test: Optional[pd.DataFrame] = None,
    y_train: Optional[pd.Series] = None,
    y_test: Optional[pd.Series] = None,
    title: str = "Williams Plot of Applicability Domain",
    figsize: tuple[float, float] = (8.5, 8.5),
    save_path: Optional[str] = None,
):
    """
    Generates a Williams plot for applicability domain analysis.
    Can accept either a model result object or model + data directly for integration
    with qsarify model components.
    """
    set_qsarify_style()

    # Extract data from model_result or direct inputs
    if model_result is not None:
        # Use model_result
        fitted_model = getattr(model_result, 'model', None)
        y_true_train = getattr(model_result, 'y_true_train', None)
        y_pred_train = getattr(model_result, 'y_pred_train', None)
        y_true_test = getattr(model_result, 'y_true_test', None)
        y_pred_test = getattr(model_result, 'y_pred_test', None)
        
        # Get X data - may need to be passed separately
        if X_train is None:
            raise ValueError("X_train must be provided even when using model_result")
            
    elif model is not None:
        # Use model + data directly
        fitted_model = model
        y_true_train = y_train
        y_true_test = y_test
        # Calculate predictions
        y_pred_train = model.predict(X_train) if X_train is not None else None
        y_pred_test = model.predict(X_test) if X_test is not None else None
    else:
        raise ValueError("Either model_result or model + data must be provided")

    if fitted_model is None or X_train is None or y_true_train is None or y_pred_train is None:
        raise ValueError("Insufficient data for Williams plot")

    ad_metrics = statistics.calculate_applicability_domain_metrics(
        X_train.values,
        y_true_train.values if hasattr(y_true_train, 'values') else y_true_train,
        y_pred_train.values if hasattr(y_pred_train, 'values') else y_pred_train,
        fitted_model,
        X_test.values if X_test is not None else None,
    )

    h_train = ad_metrics['h_train']
    h_test = ad_metrics['h_test']
    rstd_train = ad_metrics['rstd_train']
    h_star = ad_metrics['h_star']

    fig, ax = plt.subplots(figsize=figsize)

    # --- Aesthetics & limits - keeping existing style ---
    all_rstd = []
    if rstd_train is not None:
        all_rstd.extend(rstd_train)
    
    max_abs_r = float(np.nanmax(np.abs(all_rstd))) if all_rstd else 3.0
    y_pad = max(0.5, min(2.0, 0.15 * max_abs_r))
    y_max = max(3.0, np.ceil(max_abs_r + y_pad))
    y_min = -y_max

    all_h = []
    if h_train is not None:
        all_h.extend(h_train)
    if h_test is not None:
        all_h.extend(h_test)
        
    x_min = 0.0
    x_max_data = float(np.nanmax(all_h)) if all_h else h_star
    x_pad = max(0.02, 0.1 * x_max_data)
    x_max = max(h_star * 1.2, x_max_data + x_pad)

    # --- Scatter points - keeping existing style ---
    if h_train is not None and rstd_train is not None:
        ax.scatter(
            h_train,
            rstd_train,
            color=colors.TRAIN_COLOR,
            marker='o',
            s=120,
            linewidths=0.5,
            edgecolor='k',
            label="Training set",
        )
    if h_test is not None and y_true_test is not None and y_pred_test is not None:
        residuals_test = y_true_test - y_pred_test
        rstd_test = residuals_test / ad_metrics['s_resid']
        ax.scatter(
            h_test,
            rstd_test,
            color=colors.TEST_COLOR,
            marker='o',
            s=120,
            linewidths=0.5,
            edgecolor='k',
            label="Test set",
        )

    # --- Lines & annotations - keeping existing style ---
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    ax.plot([h_star, h_star], [y_min, y_max], lw=1, ls='dashed', color='black')
    ax.plot([x_min, x_max], [3, 3], lw=1, ls='dashed', color='black')
    ax.plot([x_min, x_max], [-3, -3], lw=1, ls='dashed', color='black')

    display_h = f'h*={h_star:.4f}'
    ax.text(h_star, y_min + 0.45, display_h,
             va='top', ha='right', color='black', fontsize=15, fontweight='normal')

    xm_left = (h_star + x_min) / 2.0
    xm_right = (h_star + x_max) / 2.0
    for xm in (xm_left, xm_right):
        ax.text(xm, 3.50 if y_max >= 3.50 else 3.0 + 0.5, 'Outlier zone',
                 va='center', ha='center', color='black', fontsize=15, fontweight='normal')
        ax.text(xm, -3.50 if y_min <= -3.50 else -3.0 - 0.5, 'Outlier zone',
                 va='center', ha='center', color='black', fontsize=15, fontweight='normal')

    ax.add_patch(
        patches.Rectangle(xy=(x_min, y_min),
                          width=(x_max - x_min), height=(abs(y_min) - 3),
                          linewidth=1, color='lightgray', fill=True, alpha=0.4))
    ax.add_patch(
        patches.Rectangle(xy=(x_min, 3),
                          width=(x_max - x_min), height=(y_max - 3),
                          linewidth=1, color='lightgray', fill=True, alpha=0.4))

    ax.grid(True, ls='dashed', color='black', lw=0.4, alpha=0.2)
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")

    ax.set_title(title, fontsize=20)
    ax.set_ylabel('Std. Residuals', fontsize=20)
    ax.set_xlabel('Leverage', fontsize=20)
    ax.tick_params(axis='both', labelsize=15)
    ax.legend(
        bbox_to_anchor=(0.99, 0.29), loc='upper right',
        fontsize=14, edgecolor='black'
    )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300)

    return fig, ax

def plot_regression(
    model_result: Any = None, # Should be ModelResult
    model: Any = None,
    X_train: Optional[pd.DataFrame] = None,
    X_test: Optional[pd.DataFrame] = None, 
    y_train: Optional[pd.Series] = None,
    y_test: Optional[pd.Series] = None,
    title: str = "Regression Plot - True vs Predicted",
    figsize: tuple[float, float] = (8.5, 8.5),
    save_path: Optional[str] = None,
):
    """
    Enhanced regression plot with 45-degree line showing true vs predicted values.
    Can accept either a model result object or model + data directly for integration
    with qsarify model components.
    """
    set_qsarify_style()
    
    # Extract data from model_result or direct inputs
    if model_result is not None:
        y_true_train = getattr(model_result, 'y_true_train', None)
        y_pred_train = getattr(model_result, 'y_pred_train', None)
        y_true_test = getattr(model_result, 'y_true_test', None)
        y_pred_test = getattr(model_result, 'y_pred_test', None)
    elif model is not None:
        # Calculate predictions from model and data
        y_pred_train = model.predict(X_train) if X_train is not None else None
        y_pred_test = model.predict(X_test) if X_test is not None else None
        y_true_train = y_train
        y_true_test = y_test
    else:
        raise ValueError("Either model_result or model + data must be provided")

    fig, ax = plt.subplots(figsize=figsize)

    all_true = []
    all_pred = []

    # Training data - keeping existing style
    if y_true_train is not None and y_pred_train is not None:
        ax.scatter(
            y_true_train,
            y_pred_train,
            alpha=0.6,
            label="Training set",
            color=colors.TRAIN_COLOR,
        )
        all_true.extend(y_true_train)
        all_pred.extend(y_pred_train)

    # Test data - keeping existing style
    if y_true_test is not None and y_pred_test is not None:
        ax.scatter(
            y_true_test,
            y_pred_test,
            alpha=0.6,
            label="Test set",
            color=colors.TEST_COLOR,
            marker="x",
        )
        all_true.extend(y_true_test)
        all_pred.extend(y_pred_test)

    # 45-degree line - keeping existing style
    if all_true and all_pred:
        min_val = min(min(all_true), min(all_pred))
        max_val = max(max(all_true), max(all_pred))
        ax.plot([min_val, max_val], [min_val, max_val], color="black", linestyle="--", linewidth=1)

    ax.set_xlabel("True Values", fontsize=14)
    ax.set_ylabel("Predicted Values", fontsize=14)
    ax.set_title(title, fontsize=20)
    ax.legend()

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300)

    return fig, ax


def plot_experimental_vs_predicted(
    model_result: Any, # Should be ModelResult
    title: str = "Experimental vs. Predicted",
    figsize: tuple[float, float] = (8.5, 8.5),
    save_path: Optional[str] = None,
):
    """
    Generates a plot of experimental vs. predicted values.
    """
    return plot_regression(model_result=model_result, title=title, 
                         figsize=figsize, save_path=save_path)