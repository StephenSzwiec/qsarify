"""Integration test: full qsarify library workflow.

Demonstrates intended usage as a standalone library (e.g. from a Jupyter
notebook or script):

    Step 1 — Load CSV, preprocess (filter, scale, split).
    Step 2 — Run all model types with default parameters and multiprocessing.
    Step 3 — Pick the top GA-MLR model by Q²_LOO, print its full summary,
              and generate residuals, Q-Q, Williams, and Y-scrambling plots.

Usage::

    uv run python tests/integration/test_full_workflow.py path/to/dataset.csv

The script expects a standard QSARINS-layout CSV:
  - Header row with column names.
  - Optional first column of string compound IDs (auto-detected).
  - All remaining columns numeric; the **last** column is the response variable.

Plots are written next to the CSV, in a sub-directory named after the CSV stem
(e.g. ``dataset_outputs/``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend; must precede pyplot import

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

DIVIDER = "=" * 64
SUB_DIVIDER = "─" * 64


def section(title: str) -> None:
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)


def fmt(v: object) -> str:
    """Format a scalar metric value for display."""
    if isinstance(v, float):
        return f"{v:.6f}"
    return "—"


# ─────────────────────────────────────────────────────────────────────────────
# Entry point guard — required so multiprocessing workers don't re-run the
# full script body when spawning on macOS/Windows.
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ─────────────────────────────────────────────────────────────────────────
    # CLI argument
    # ─────────────────────────────────────────────────────────────────────────

    if len(sys.argv) < 2:
        print(
            "Usage: uv run python tests/integration/test_full_workflow.py <dataset.csv>",
            file=sys.stderr,
        )
        sys.exit(1)

    CSV_PATH = Path(sys.argv[1]).resolve()
    if not CSV_PATH.exists():
        print(f"Error: file not found: {CSV_PATH}", file=sys.stderr)
        sys.exit(1)

    OUTPUT_DIR = CSV_PATH.parent / f"{CSV_PATH.stem}_outputs"
    OUTPUT_DIR.mkdir(exist_ok=True)
    PLOT_DIR = OUTPUT_DIR / "plots"
    PLOT_DIR.mkdir(exist_ok=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Step 1 — Data import and preprocessing
    # ─────────────────────────────────────────────────────────────────────────

    section("Step 1 · Data Import and Preprocessing")

    from qsarify.io.loaders import load_csv_dataset
    from qsarify.preprocessing import preprocessing

    ds = load_csv_dataset(CSV_PATH)
    y_col = str(ds.y_series.name)
    n_raw, p_raw = ds.X_df.shape
    print(f"CSV:      {CSV_PATH.name}")
    print(f"Loaded:   {n_raw} samples  |  {p_raw} descriptors")
    print(f"Response: '{y_col}'")

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
    X_train = np.asarray(X_tr, dtype=np.float64)
    X_test = np.asarray(X_te, dtype=np.float64)
    y_train = np.asarray(y_tr, dtype=np.float64)
    y_test = np.asarray(y_te, dtype=np.float64)

    n_tr_int, n_feat = X_train.shape
    n_te_int = X_test.shape[0]
    print(f"After preprocessing: {n_feat} descriptors remain (removed {p_raw - n_feat})")
    print(f"Train / Test:  {n_tr_int} / {n_te_int}")

    # ─────────────────────────────────────────────────────────────────────────
    # Step 2 — Run all model types
    # ─────────────────────────────────────────────────────────────────────────

    section("Step 2 · Model Generation  (all types, default parameters)")

    from qsarify.modeling.clustering import cophenetic_cluster
    from qsarify.modeling.genetic_algorithm import run_ga_mlr
    from qsarify.modeling.models import (
        GradientBoostingModel,
        LassoModel,
        RandomForestModel,
        RidgeModel,
        SVRModel,
    )
    from qsarify.results.result_set import ResultSet

    result_set: ResultSet = ResultSet()
    N_WORKERS = 4

    # Respect the n/5 statistical limit for GA-MLR max_variables
    MAX_VARS = max(1, min(5, n_tr_int // 5))

    # GA-MLR — cophenetic clustering + exhaustive enumeration + genetic algorithm
    print(
        f"  GA-MLR  (pop=1000, gen=100, keep_best=10, "
        f"max_vars={MAX_VARS}, {N_WORKERS} workers)…"
    )
    cluster_result = cophenetic_cluster(X_train)
    cluster_map = cluster_result.cluster_map

    for m in run_ga_mlr(
        X_train,
        y_train,
        cluster_map=cluster_map,
        X_test=X_test,
        y_test=y_test,
        min_variables=1,
        max_variables=MAX_VARS,
        population_size=1000,
        max_generations=100,
        keep_best=10,
        fitness_function="q2_loo",
        quik_delta=0.05,
        random_seed=42,
        n_workers=N_WORKERS,
    ):
        result_set.add(m.get_results())

    n_ga = sum(1 for r in result_set if r.model_type == "mlr")
    print(f"  GA-MLR complete — {n_ga} models retained")

    # Ridge -----------------------------------------------------------------------
    print("  Ridge (α=1.0)…")
    result_set.add(
        RidgeModel(alpha=1.0, n_jobs=N_WORKERS)
        .fit(X_train, y_train, X_test=X_test, y_test=y_test)
        .get_results()
    )
    print("  Ridge complete")

    # Lasso -----------------------------------------------------------------------
    print("  Lasso (α=0.1)…")
    result_set.add(
        LassoModel(alpha=0.1, max_iter=10000, n_jobs=N_WORKERS)
        .fit(X_train, y_train, X_test=X_test, y_test=y_test)
        .get_results()
    )
    print("  Lasso complete")

    # SVR -------------------------------------------------------------------------
    print("  SVR (C=1.0, rbf)…")
    result_set.add(
        SVRModel(C=1.0, gamma="scale", kernel="rbf", n_jobs=N_WORKERS)
        .fit(X_train, y_train, X_test=X_test, y_test=y_test)
        .get_results()
    )
    print("  SVR complete")

    # Random Forest ---------------------------------------------------------------
    print("  Random Forest (100 trees)…")
    result_set.add(
        RandomForestModel(n_estimators=100, n_jobs=N_WORKERS)
        .fit(X_train, y_train, X_test=X_test, y_test=y_test)
        .get_results()
    )
    print("  Random Forest complete")

    # Gradient Boosting -----------------------------------------------------------
    print("  Gradient Boosting (100 stages, lr=0.1)…")
    result_set.add(
        GradientBoostingModel(n_estimators=100, learning_rate=0.1, max_depth=3)
        .fit(X_train, y_train, X_test=X_test, y_test=y_test)
        .get_results()
    )
    print("  Gradient Boosting complete")

    total_models = len(result_set)
    print(f"\nTotal models in result set: {total_models}")
    assert total_models >= 6, f"Expected ≥ 6 models, got {total_models}"

    # ─────────────────────────────────────────────────────────────────────────
    # Step 3 — Top GA-MLR model: summary, plots, Y-scrambling
    # ─────────────────────────────────────────────────────────────────────────

    section("Step 3 · Top GA-MLR Model — Summary, Plots, Y-Scrambling")

    # Select the GA-MLR result with the highest Q²_LOO
    ga_results = [
        (i, r)
        for i, r in enumerate(result_set)
        if r.model_type == "mlr"
    ]
    assert ga_results, "No GA-MLR models found in the result set."

    best_idx, best = max(
        ga_results,
        key=lambda t: t[1].q_squared_loo if t[1].q_squared_loo is not None else -9.0,
    )

    selected_names = [
        descriptor_names[i] for i in (best.selected_descriptors or [])
    ]
    print(f"Best GA-MLR model: result_set[{best_idx}]")
    print(f"  Selected descriptors: {selected_names}")
    print()

    # Model summary
    print(SUB_DIVIDER)
    print("  MODEL SUMMARY")
    print(SUB_DIVIDER)
    print(f"  Type:                  {best.model_type}")
    print(f"  Features:              {best.n_features}")
    print(f"  Training samples:      {best.n_train}")
    print(f"  Test samples:          {fmt(best.n_test)}")
    print()
    print("  Training metrics")
    print(f"    R²                   {fmt(best.r_squared)}")
    print(f"    R²_adj               {fmt(best.r_squared_adj)}")
    print(f"    RMSE                 {fmt(best.rmse)}")
    print(f"    MAE                  {fmt(best.mae)}")
    print(f"    MSE                  {fmt(best.mse)}")
    print(f"    RSS                  {fmt(best.rss)}")
    print(f"    TSS                  {fmt(best.tss)}")
    print(f"    MSS                  {fmt(best.mss)}")
    print(f"    s (std err est)      {fmt(best.std_error_estimate)}")
    print(f"    F-statistic          {fmt(best.f_statistic)}")
    print(f"    LOF (Friedman's)     {fmt(best.lof)}")
    print(f"    CCC                  {fmt(best.ccc)}")
    print()
    print("  Cross-validation")
    print(f"    Q²_LOO               {fmt(best.q_squared_loo)}")
    print()
    print("  External validation (test set)")
    print(f"    Q²_F1                {fmt(best.q_squared_f1)}")
    print(f"    Q²_F2                {fmt(best.q_squared_f2)}")
    print(f"    Q²_F3                {fmt(best.q_squared_f3)}")
    print(f"    R²_ext               {fmt(best.r_squared_ext)}")
    print(f"    PRESS_ext            {fmt(best.press_ext)}")
    print()
    print("  Golbraikh-Tropsha / Roy criteria")
    print(f"    k  (slope origin)    {fmt(best.slope_origin)}")
    print(f"    k' (reverse)         {fmt(best.slope_origin_reverse)}")
    print(f"    R²_0                 {fmt(best.r_squared_origin)}")
    print(f"    R'²_0                {fmt(best.r_squared_origin_reverse)}")
    print(f"    r̄²_m  (Roy mean)     {fmt(best.roy_r_squared_m_mean)}")
    print(f"    Δr²_m (Roy delta)    {fmt(best.roy_r_squared_m_delta)}")
    print(f"    clos                 {fmt(best.closeness)}")
    print(f"    clos'                {fmt(best.closeness_reverse)}")
    print()
    print("  Applicability domain")
    print(f"    h* (leverage thresh) {fmt(best.leverage_threshold)}")
    if best.leverage is not None:
        outside_ad = int(np.sum(best.leverage > (best.leverage_threshold or np.inf)))
        print(f"    Compounds outside AD {outside_ad} / {best.n_train}")

    if best.coef_p_values is not None:
        print()
        print("  Coefficient statistics")
        for name, pval in zip(selected_names, best.coef_p_values):
            print(f"    p({name:<10}) = {pval:.4f}")
    print(SUB_DIVIDER)

    # Diagnostic plots
    print()
    print("  Generating diagnostic plots…")

    from qsarify.viz.plots import plot_qq, plot_residuals, plot_williams, plot_y_scrambling

    residuals_path = PLOT_DIR / "residuals.png"
    plot_residuals(best, save_path=str(residuals_path))
    print(f"  [ok] residuals.png")

    qq_path = PLOT_DIR / "qq.png"
    plot_qq(best, save_path=str(qq_path))
    print(f"  [ok] qq.png")

    williams_path = PLOT_DIR / "williams.png"
    plot_williams(best, save_path=str(williams_path))
    print(f"  [ok] williams.png")

    # Y-scrambling
    print()
    print(f"  Running Y-scrambling (100 iterations, {N_WORKERS} workers)…")

    from sklearn.linear_model import LinearRegression

    from qsarify.validation.procedures import YScramblingResult, run_y_scrambling

    selected_cols = np.array(best.selected_descriptors or [], dtype=np.intp)
    X_val = X_train[:, selected_cols]

    y_scram: YScramblingResult = run_y_scrambling(
        LinearRegression(),
        X_val,
        y_train,
        r2_original=best.r_squared or 0.0,
        q2_original=best.q_squared_loo or 0.0,
        n_iterations=100,
        random_seed=42,
        n_workers=N_WORKERS,
    )

    print(f"  Y-scrambling summary:")
    print(f"    R²(original)    = {y_scram.r2_original:.4f}")
    print(f"    Q²(original)    = {y_scram.q2_original:.4f}")
    print(f"    mean R²(scram)  = {y_scram.mean_r2_scrambled:.4f} ± {y_scram.std_r2_scrambled:.4f}")
    print(f"    mean Q²(scram)  = {y_scram.mean_q2_scrambled:.4f} ± {y_scram.std_q2_scrambled:.4f}")

    yscram_path = PLOT_DIR / "y_scrambling.png"
    plot_y_scrambling(y_scram, best, save_path=str(yscram_path))
    print(f"  [ok] y_scrambling.png")

    print()
    print(f"  Plots saved to: {PLOT_DIR}/")

    # ─────────────────────────────────────────────────────────────────────────
    # Structural assertions (dataset-agnostic)
    # ─────────────────────────────────────────────────────────────────────────

    section("Sanity checks")

    assert best.r_squared is not None, "R² should be populated"
    assert best.q_squared_loo is not None, "Q²_LOO should be populated"
    assert best.n_features is not None and best.n_features >= 1
    assert residuals_path.exists() and residuals_path.stat().st_size > 0, "residuals.png missing"
    assert qq_path.exists() and qq_path.stat().st_size > 0, "qq.png missing"
    assert williams_path.exists() and williams_path.stat().st_size > 0, "williams.png missing"
    assert yscram_path.exists() and yscram_path.stat().st_size > 0, "y_scrambling.png missing"
    assert y_scram.r2_original >= y_scram.mean_r2_scrambled, (
        "Scrambled R² should not exceed original (model must outperform chance)"
    )

    print(f"  [ok] R² = {best.r_squared:.4f}  |  Q²_LOO = {best.q_squared_loo:.4f}")
    print(f"  [ok] {best.n_features} descriptor(s): {selected_names}")
    print("  [ok] All four plot files written and non-empty")
    print(
        f"  [ok] R²(original)={y_scram.r2_original:.4f} > "
        f"mean R²(scrambled)={y_scram.mean_r2_scrambled:.4f}"
    )

    print()
    print("Integration test complete.")
    print(f"Outputs: {OUTPUT_DIR}/")
