"""ResultsScreen: Step IV — view results, run validation, generate plots."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # non-interactive backend; must be set before pyplot import

from textual.app import ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, DataTable, Footer, Header, Label, Static
from textual.widgets import SelectionList
from textual.widgets.selection_list import Selection
from textual import work

__all__ = ["ResultsScreen"]


# Columns shown in the results table (header label, ModelResult field name)
_DISPLAY_COLS: list[tuple[str, str]] = [
    ("Type", "model_type"),
    ("Feats", "n_features"),
    ("Train", "n_train"),
    ("Test", "n_test"),
    ("R²", "r_squared"),
    ("R²adj", "r_squared_adj"),
    ("RMSE", "rmse"),
    ("MAE", "mae"),
    ("Q²_LOO", "q_squared_loo"),
    ("Q²_F1", "q_squared_f1"),
    ("Q²_F2", "q_squared_f2"),
    ("CCC", "ccc"),
    ("F", "f_statistic"),
]


class ResultsScreen(Screen[None]):
    """Step IV of the QSARINS workflow: View and Evaluate Models.

    - Scrollable table of all built models and their metrics.
    - Model selection list for targeted validation / plotting.
    - LMO and Y-scrambling validation (results marked inline).
    - Diagnostic plot generation saved to a user-specified file.
    """

    def compose(self) -> ComposeResult:
        yield Header()
        yield Label("Step IV  ·  Results & Validation", classes="step-banner")
        with VerticalScroll():
            # ── Results table ──────────────────────────────────────────
            yield Label("Model Results", classes="section-title")
            yield DataTable(id="results-table", show_cursor=True)

            # ── Model selector ─────────────────────────────────────────
            yield Label(
                "Select Models for Validation / Plotting",
                classes="section-title",
            )
            yield SelectionList(id="model-selector")

            # ── Validation ─────────────────────────────────────────────
            yield Label("Validation", classes="section-title")
            with Horizontal(classes="button-bar"):
                yield Button("Run LMO…", id="btn-lmo", variant="primary")
                yield Button("Run Y-Scrambling…", id="btn-yscram", variant="primary")

            # ── Plots ──────────────────────────────────────────────────
            yield Label("Diagnostic Plots", classes="section-title")
            with Horizontal(classes="button-bar"):
                yield Button("Generate Plot…", id="btn-plot", variant="default")

            yield Static("", id="results-status", classes="muted")

            # ── Navigation ─────────────────────────────────────────────
            with Horizontal(classes="button-bar"):
                yield Button("← Model Building", id="btn-back", variant="default")
        yield Footer()

    def on_mount(self) -> None:
        self._setup_table()
        self._refresh_all()

    # ── Table setup ────────────────────────────────────────────────────

    def _setup_table(self) -> None:
        table = self.query_one("#results-table", DataTable)
        table.add_column("#")
        for label, _ in _DISPLAY_COLS:
            table.add_column(label)
        table.add_column("Validated")

    # ── Refresh helpers ────────────────────────────────────────────────

    def _refresh_all(self) -> None:
        self._refresh_table()
        self._refresh_model_selector()

    def _refresh_table(self) -> None:
        table = self.query_one("#results-table", DataTable)
        table.clear()
        for idx, result in enumerate(
            self.app.project.result_set  # type: ignore[attr-defined]
        ):
            row: list[str] = [str(idx)]
            for _, field in _DISPLAY_COLS:
                val = getattr(result, field, None)
                if isinstance(val, float):
                    row.append(f"{val:.4f}")
                elif val is None:
                    row.append("—")
                else:
                    row.append(str(val))
            # Validation badge
            badges: list[str] = []
            if result.lmo_results is not None:
                badges.append("LMO")
            if result.y_scrambling_results is not None:
                badges.append("Y-scram")
            row.append(", ".join(badges) if badges else "—")
            table.add_row(*row)

    def _refresh_model_selector(self) -> None:
        sel = self.query_one("#model-selector", SelectionList)
        sel.clear_options()
        for idx, result in enumerate(
            self.app.project.result_set  # type: ignore[attr-defined]
        ):
            r2 = result.r_squared
            label = (
                f"[{idx}] {result.model_type}"
                f"  ({result.n_features} feats"
                + (f", R²={r2:.3f}" if r2 is not None else "")
                + ")"
            )
            sel.add_option(Selection(label, idx, False))

    def _selected_indices(self) -> list[int]:
        return list(self.query_one("#model-selector", SelectionList).selected)

    def _set_status(self, msg: str) -> None:
        self.query_one("#results-status", Static).update(msg)

    # ── Button handler ─────────────────────────────────────────────────

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id
        if bid == "btn-lmo":
            indices = self._selected_indices()
            if not indices:
                self.notify("Select at least one model first.", severity="warning")
                return
            from qsarify.tui.views._modals import LMOConfigModal

            captured = list(indices)
            self.app.push_screen(
                LMOConfigModal(),
                lambda cfg: self._run_lmo(captured, cfg),
            )
        elif bid == "btn-yscram":
            indices = self._selected_indices()
            if not indices:
                self.notify("Select at least one model first.", severity="warning")
                return
            from qsarify.tui.views._modals import YScramblingConfigModal

            captured = list(indices)
            self.app.push_screen(
                YScramblingConfigModal(),
                lambda cfg: self._run_yscram(captured, cfg),
            )
        elif bid == "btn-plot":
            indices = self._selected_indices()
            if not indices:
                self.notify("Select a model for the plot.", severity="warning")
                return
            from qsarify.tui.views._modals import PlotConfigModal

            idx = indices[0]
            self.app.push_screen(
                PlotConfigModal(),
                lambda cfg: self._generate_plot(idx, cfg),
            )
        elif bid == "btn-back":
            self.app.pop_screen()

    # ── Validation runners ─────────────────────────────────────────────

    def _run_lmo(self, indices: list[int], cfg: dict | None) -> None:
        if cfg is None:
            return
        self._set_status(f"Running LMO on {len(indices)} model(s)…")
        self._worker_lmo(indices, cfg["holdout_fraction"], cfg["n_iterations"])

    def _run_yscram(self, indices: list[int], cfg: dict | None) -> None:
        if cfg is None:
            return
        self._set_status(f"Running Y-scrambling on {len(indices)} model(s)…")
        self._worker_yscram(indices, cfg["n_iterations"])

    def _generate_plot(self, model_idx: int, cfg: dict | None) -> None:
        if cfg is None:
            return
        results_list = list(
            self.app.project.result_set  # type: ignore[attr-defined]
        )
        if model_idx >= len(results_list):
            self.notify(f"Model index {model_idx} out of range.", severity="error")
            return
        result = results_list[model_idx]
        self._worker_plot(result, cfg["plot_type"], cfg["output_path"])

    # ── Workers ────────────────────────────────────────────────────────

    @work(thread=True)
    def _worker_lmo(
        self, indices: list[int], holdout_fraction: float, n_iterations: int
    ) -> None:
        try:
            self.app.project.run_lmo_validation(  # type: ignore[attr-defined]
                model_indices=indices,
                holdout_fraction=holdout_fraction,
                n_iterations=n_iterations,
            )
            self.app.call_from_thread(self._on_lmo_done)
        except Exception as exc:
            msg = str(exc)
            self.app.call_from_thread(
                lambda: self.notify(msg, title="LMO failed", severity="error")
            )
            self.app.call_from_thread(lambda: self._set_status(""))

    def _on_lmo_done(self) -> None:
        self._set_status("LMO validation complete.")
        self._refresh_all()
        for result in self.app.project.result_set:  # type: ignore[attr-defined]
            if result.lmo_results:
                lmo = result.lmo_results
                self.notify(
                    f"Q²_LMO = {lmo['mean_q2']:.4f} ± {lmo['std_q2']:.4f}",
                    title="LMO result",
                )
                break

    @work(thread=True)
    def _worker_yscram(self, indices: list[int], n_iterations: int) -> None:
        try:
            self.app.project.run_y_scrambling(  # type: ignore[attr-defined]
                model_indices=indices,
                n_iterations=n_iterations,
            )
            self.app.call_from_thread(self._on_yscram_done)
        except Exception as exc:
            msg = str(exc)
            self.app.call_from_thread(
                lambda: self.notify(msg, title="Y-scrambling failed", severity="error")
            )
            self.app.call_from_thread(lambda: self._set_status(""))

    def _on_yscram_done(self) -> None:
        self._set_status("Y-scrambling complete.")
        self._refresh_all()
        for result in self.app.project.result_set:  # type: ignore[attr-defined]
            if result.y_scrambling_results:
                ys = result.y_scrambling_results
                self.notify(
                    f"mean R²(scrambled) = {ys['mean_r2_scrambled']:.4f}  "
                    f"vs  R²(original) = {ys['r2_original']:.4f}",
                    title="Y-scrambling result",
                )
                break

    @work(thread=True)
    def _worker_plot(
        self, result: object, plot_type: str, output_path: str
    ) -> None:
        from qsarify.results.model_result import ModelResult
        from qsarify.viz.plots import plot_qq, plot_residuals, plot_williams

        if not isinstance(result, ModelResult):
            self.app.call_from_thread(
                lambda: self.notify("Invalid model result.", severity="error")
            )
            return

        try:
            if plot_type == "residuals":
                plot_residuals(result, save_path=output_path)
            elif plot_type == "qq":
                plot_qq(result, save_path=output_path)
            elif plot_type == "williams":
                plot_williams(result, save_path=output_path)
            elif plot_type == "y_scrambling":
                # Per-iteration arrays are not persisted; show text summary.
                if result.y_scrambling_results is None:
                    self.app.call_from_thread(
                        lambda: self.notify(
                            "Run Y-scrambling validation first.",
                            title="No data",
                            severity="warning",
                        )
                    )
                    return
                ys = result.y_scrambling_results
                summary = (
                    f"Y-Scrambling ({ys['n_iterations']} iterations)\n"
                    f"  R²(scrambled):  mean={ys['mean_r2_scrambled']:.4f}"
                    f" ± {ys['std_r2_scrambled']:.4f}\n"
                    f"  Q²(scrambled):  mean={ys['mean_q2_scrambled']:.4f}"
                    f" ± {ys['std_q2_scrambled']:.4f}\n"
                    f"  R²(original) = {ys['r2_original']:.4f}"
                    f"  |  Q²(original) = {ys['q2_original']:.4f}"
                )
                self.app.call_from_thread(
                    lambda: self.notify(
                        summary, title="Y-Scrambling Summary", timeout=10.0
                    )
                )
                return
            path_str = output_path
            self.app.call_from_thread(
                lambda: self.notify(
                    f"Saved to {path_str}", title="Plot generated"
                )
            )
        except Exception as exc:
            msg = str(exc)
            self.app.call_from_thread(
                lambda: self.notify(msg, title="Plot failed", severity="error")
            )
