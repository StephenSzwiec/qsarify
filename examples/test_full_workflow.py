"""
Full qsarify library workflow example.

This script demonstrates how to use qsarify as a standalone library, for
instance in a Jupyter notebook or an automated analysis pipeline.

The workflow covers:
    1. Data Loading: Load a QSAR dataset from a CSV file.
    2. Preprocessing: Filter, scale, and split the data.
    3. Model Building: Train multiple QSAR models (GA-MLR, Ridge, etc.).
    4. Analysis: Select the best model and inspect its performance.
    5. Visualization: Generate and save diagnostic plots.
    6. Validation: Perform Y-scrambling to ensure model robustness.

To run this example:
    uv run python examples/test_full_workflow.py
"""

# %%
# =============================================================================
# Setup
# =============================================================================
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np

# Use a non-interactive backend for saving plots
matplotlib.use("Agg")

from qsarify.io.loaders import load_csv_dataset
from qsarify.modeling.clustering import cophenetic_cluster
from qsarify.modeling.genetic_algorithm import run_ga_mlr
from qsarify.modeling.models import (
    GradientBoostingModel,
    LassoModel,
    RandomForestModel,
    RidgeModel,
    SVRModel,
)
from qsarify.preprocessing.pipeline import preprocessing
from qsarify.results.model_result import ModelResult
from qsarify.results.result_set import ResultSet
from qsarify.validation.procedures import YScramblingResult, run_y_scrambling
from qsarify.viz.plots import (
    plot_qq,
    plot_residuals,
    plot_williams,
    plot_y_scrambling,
)

# %%
# =============================================================================
# Configuration
# =============================================================================

# Path to the dataset CSV file
# This file is expected to have:
# - A header row with descriptor names.
# - An optional first column with compound IDs.
# - The last column as the response variable (e.g., activity).
CSV_PATH = Path("examples/28BenzeneDescriptors.csv").resolve()

# Directory for saving outputs (plots, results)
OUTPUT_DIR = CSV_PATH.parent / f"{CSV_PATH.stem}_outputs"
PLOT_DIR = OUTPUT_DIR / "plots"
PLOT_DIR.mkdir(exist_ok=True, parents=True)

# Multiprocessing settings
N_WORKERS = 4


# %%
# =============================================================================
# Helper Functions
# =============================================================================
def print_section(title: str):
    """Prints a formatted section header."""
    print("\n" + "=" * 70)
    print(f"// {title.upper()}")
    print("=" * 70)


def print_summary(result: ModelResult, title: str = "Model Summary"):
    """Prints a formatted summary of a ModelResult."""

    def fmt(v):
        return f"{v:.4f}" if isinstance(v, float) else "N/A"

    print("-" * 70)
    print(f"// {title}")
    print("-" * 70)
    print(f"  Model Type:            {result.model_type.upper()}")
    print(f"  Features:              {result.n_features}")
    print(f"  Training Samples:      {result.n_train}")
    print(f"  Test Samples:          {result.n_test or 'N/A'}")
    print("\n  TRAINING METRICS")
    print(f"    R²:                    {fmt(result.r_squared)}")
    print(f"    Adjusted R²:           {fmt(result.r_squared_adj)}")
    print(f"    RMSE:                  {fmt(result.rmse)}")
    print("\n  CROSS-VALIDATION")
    print(f"    Q² (LOO):              {fmt(result.q_squared_loo)}")
    if result.n_test:
        print("\n  EXTERNAL VALIDATION (TEST SET)")
        print(f"    Q²_F1:                 {fmt(result.q_squared_f1)}")
        print(f"    Q²_F2:                 {fmt(result.q_squared_f2)}")
        print(f"    Q²_F3:                 {fmt(result.q_squared_f3)}")
    print("-" * 70)


# %%
# =============================================================================
# Main Workflow
# =============================================================================

if __name__ == "__main__":
    if not CSV_PATH.exists():
        print(f"Error: Dataset not found at {CSV_PATH}", file=sys.stderr)
        sys.exit(1)

    # %%
    # -------------------------------------------------------------------------
    # Step 1: Data Import and Preprocessing
    # -------------------------------------------------------------------------
    print_section("Step 1: Data Import and Preprocessing")

    ds = load_csv_dataset(CSV_PATH)
    print(f"Loaded dataset: {CSV_PATH.name}")
    print(f"  - Raw shape: {ds.X_df.shape[0]} samples, {ds.X_df.shape[1]} descriptors")
    print(f"  - Response variable: '{ds.y_series.name}'")

    X_tr, X_te, y_tr, y_te = preprocessing(
        ds.X_df,
        ds.y_series,
        split="random",
        test_size=0.20,
        constant_threshold=0.01,
        correlation_threshold=0.90,
        normalize=True,
        random_seed=42,
    )

    descriptor_names = list(X_tr.columns)
    X_train, y_train = np.asarray(X_tr), np.asarray(y_tr)
    X_test, y_test = np.asarray(X_te), np.asarray(y_te)

    print("\nPreprocessing complete:")
    print(f"  - Descriptors remaining: {X_train.shape[1]}")
    print(f"  - Train set size: {X_train.shape[0]}")
    print(f"  - Test set size: {X_test.shape[0]}")

    # %%
    # -------------------------------------------------------------------------
    # Step 2: Model Building
    # -------------------------------------------------------------------------
    print_section("Step 2: Model Building")
    result_set = ResultSet()

    # GA-MLR
    print("\nRunning GA-MLR...")
    MAX_VARS = max(1, min(5, X_train.shape[0] // 5))
    cluster_result = cophenetic_cluster(X_train)
    ga_models = run_ga_mlr(
        X_train,
        y_train,
        cluster_map=cluster_result.cluster_map,
        X_test=X_test,
        y_test=y_test,
        max_variables=MAX_VARS,
        population_size=1000,
        max_generations=100,
        keep_best=10,
        random_seed=42,
        n_workers=N_WORKERS,
    )
    for m in ga_models:
        result_set.add(m.get_results())
    print(f"  -> Completed. Found {len(ga_models)} GA-MLR models.")

    # Other models
    models_to_run = {
        "Ridge": RidgeModel(alpha=1.0, n_jobs=N_WORKERS),
        "Lasso": LassoModel(alpha=0.1, max_iter=10000, n_jobs=N_WORKERS),
        "SVR": SVRModel(C=1.0, kernel="rbf", n_jobs=N_WORKERS),
        "Random Forest": RandomForestModel(n_estimators=100, n_jobs=N_WORKERS, random_seed=42),
        "Gradient Boosting": GradientBoostingModel(
            n_estimators=100, learning_rate=0.1, max_depth=3, random_seed=42
        ),
    }

    for name, model in models_to_run.items():
        print(f"\nRunning {name}...")
        model.fit(X_train, y_train, X_test=X_test, y_test=y_test)
        result_set.add(model.get_results())
        print(f"  -> Completed.")

    print(f"\nTotal models in result set: {len(result_set)}")

    # %%
    # -------------------------------------------------------------------------
    # Step 3: Analyze and Visualize the Best Model
    # -------------------------------------------------------------------------
    print_section("Step 3: Analyze and Visualize Best Model")

    # Select the best GA-MLR model based on Q²_LOO
    best_model_result = result_set.filter(model_type="mlr").best(by="q_squared_loo")[0]

    print_summary(best_model_result, title="Best GA-MLR Model")

    selected_indices = best_model_result.selected_descriptors or []
    selected_names = [descriptor_names[i] for i in selected_indices]
    print(f"\nSelected Descriptors ({len(selected_names)}):")
    for name in selected_names:
        print(f"  - {name}")

    # Generate diagnostic plots
    print("\nGenerating diagnostic plots...")
    plot_residuals(best_model_result, save_path=str(PLOT_DIR / "residuals.png"))
    plot_qq(best_model_result, save_path=str(PLOT_DIR / "qq.png"))
    plot_williams(best_model_result, save_path=str(PLOT_DIR / "williams.png"))
    print(f"  -> Plots saved to {PLOT_DIR}")

    # %%
    # -------------------------------------------------------------------------
    # Step 4: Y-Scrambling Validation
    # -------------------------------------------------------------------------
    print_section("Step 4: Y-Scrambling Validation")

    # Re-fit the model on the selected descriptors for validation
    from sklearn.linear_model import LinearRegression
    X_val = X_train[:, selected_indices]

    y_scram_result: YScramblingResult = run_y_scrambling(
        LinearRegression(),
        X_val,
        y_train,
        r2_original=best_model_result.r_squared or 0.0,
        q2_original=best_model_result.q_squared_loo or 0.0,
        n_iterations=100,
        random_seed=42,
        n_workers=N_WORKERS,
    )

    print("Y-Scrambling Results:")
    print(f"  - R² (Original):   {y_scram_result.r2_original:.4f}")
    print(f"  - Q² (Original):   {y_scram_result.q2_original:.4f}")
    print(f"  - Mean R² (Scram): {y_scram_result.mean_r2_scrambled:.4f} \u00b1 {y_scram_result.std_r2_scrambled:.4f}")
    print(f"  - Mean Q² (Scram): {y_scram_result.mean_q2_scrambled:.4f} \u00b1 {y_scram_result.std_q2_scrambled:.4f}")

    # Plot Y-scrambling results
    plot_y_scrambling(y_scram_result, best_model_result, save_path=str(PLOT_DIR / "y_scrambling.png"))
    print(f"\n  -> Y-scrambling plot saved to {PLOT_DIR}")

    print("\n" + "=" * 70)
    print("\u2705 Workflow completed successfully!")
    print(f"All outputs are in: {OUTPUT_DIR}")
    print("=" * 70)
