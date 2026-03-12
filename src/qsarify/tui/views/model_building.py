"""ModelBuildingScreen: Step III — run models and accumulate results."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, DataTable, Footer, Header, Label, Static
from textual import work

__all__ = ["ModelBuildingScreen"]


_TABLE_COLS: list[tuple[str, str]] = [
    ("Model", "model_type"),
    ("Feats", "n_features"),
    ("R²", "r_squared"),
    ("R²adj", "r_squared_adj"),
    ("RMSE", "rmse"),
    ("Q²_LOO", "q_squared_loo"),
]


class ModelBuildingScreen(Screen[None]):
    """Step III of the QSARINS workflow: Variable Selection and Model Building.

    Allows the user to run any combination of GA-MLR, Ridge, Lasso, SVR,
    Random Forest, and Gradient Boosting models.  Each run appends results
    to :attr:`~qsarify.project.QSARProject.result_set`.  The embedded table
    refreshes after each model completes.
    """

    def compose(self) -> ComposeResult:
        yield Header()
        yield Label("Step III  ·  Model Building", classes="step-banner")
        with VerticalScroll():
            # ── Data summary ───────────────────────────────────────────
            yield Label("Training Data", classes="section-title")
            yield Static("(loading…)", id="data-summary", classes="info-panel")

            # ── Model buttons ──────────────────────────────────────────
            yield Label("Run Models", classes="section-title")
            with Horizontal(classes="button-bar"):
                yield Button("GA-MLR", id="btn-ga", variant="primary")
                yield Button("Ridge", id="btn-ridge", variant="default")
                yield Button("Lasso", id="btn-lasso", variant="default")
            with Horizontal(classes="button-bar"):
                yield Button("SVR", id="btn-svr", variant="default")
                yield Button("Random Forest", id="btn-rf", variant="default")
                yield Button("Gradient Boosting", id="btn-gbr", variant="default")

            yield Static("", id="run-status", classes="muted")

            # ── Built models table ─────────────────────────────────────
            yield Label("Built Models", classes="section-title")
            yield DataTable(id="models-table", show_cursor=False)

            # ── Navigation ─────────────────────────────────────────────
            with Horizontal(classes="button-bar"):
                yield Button(
                    "View Results  →",
                    id="btn-results",
                    variant="primary",
                    classes="primary-action",
                    disabled=True,
                )
                yield Button("← Data Setup", id="btn-back", variant="default")
        yield Footer()

    def on_mount(self) -> None:
        self._setup_table()
        self._refresh_data_summary()
        self._refresh_table()
        self._update_nav_btn()

    # ── Table helpers ──────────────────────────────────────────────────

    def _setup_table(self) -> None:
        table = self.query_one("#models-table", DataTable)
        for label, _ in _TABLE_COLS:
            table.add_column(label)

    def _refresh_data_summary(self) -> None:
        p = self.app.project  # type: ignore[attr-defined]
        n_tr = p.X_train.shape[0] if p.X_train is not None else 0
        n_te = p.X_test.shape[0] if p.X_test is not None else 0
        n_feat = len(p.descriptor_names)
        self.query_one("#data-summary", Static).update(
            f"Train: {n_tr}  |  Test: {n_te}  |  Descriptors: {n_feat}"
        )

    def _refresh_table(self) -> None:
        table = self.query_one("#models-table", DataTable)
        table.clear()
        for result in self.app.project.result_set:  # type: ignore[attr-defined]
            row: list[str] = []
            for _, field in _TABLE_COLS:
                val = getattr(result, field, None)
                if isinstance(val, float):
                    row.append(f"{val:.4f}")
                elif val is None:
                    row.append("—")
                else:
                    row.append(str(val))
            table.add_row(*row)

    def _update_nav_btn(self) -> None:
        from qsarify.project import ProjectState

        can_proceed = (
            self.app.project.state >= ProjectState.MODELS_BUILT  # type: ignore[attr-defined]
        )
        self.query_one("#btn-results", Button).disabled = not can_proceed

    def _set_status(self, msg: str) -> None:
        self.query_one("#run-status", Static).update(msg)

    def _on_model_done(self, model_name: str) -> None:
        n = len(self.app.project.result_set)  # type: ignore[attr-defined]
        self._set_status(f"{model_name} complete.  Total models in set: {n}")
        self._refresh_table()
        self._update_nav_btn()

    # ── Button handler ─────────────────────────────────────────────────

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id
        if bid == "btn-ga":
            from qsarify.tui.views._modals import GAMLRConfigModal

            self.app.push_screen(GAMLRConfigModal(), self._run_ga)
        elif bid == "btn-ridge":
            from qsarify.tui.views._modals import RidgeConfigModal

            self.app.push_screen(RidgeConfigModal(), self._run_ridge)
        elif bid == "btn-lasso":
            from qsarify.tui.views._modals import LassoConfigModal

            self.app.push_screen(LassoConfigModal(), self._run_lasso)
        elif bid == "btn-svr":
            from qsarify.tui.views._modals import SVRConfigModal

            self.app.push_screen(SVRConfigModal(), self._run_svr)
        elif bid == "btn-rf":
            from qsarify.tui.views._modals import RFConfigModal

            self.app.push_screen(RFConfigModal(), self._run_rf)
        elif bid == "btn-gbr":
            from qsarify.tui.views._modals import GBRConfigModal

            self.app.push_screen(GBRConfigModal(), self._run_gbr)
        elif bid == "btn-results":
            self.app.go_to_results()  # type: ignore[attr-defined]
        elif bid == "btn-back":
            self.app.pop_screen()

    # ── Modal callbacks ────────────────────────────────────────────────

    def _run_ga(self, cfg: dict | None) -> None:
        if cfg is None:
            return
        self._set_status("Running GA-MLR…  (this may take a while)")
        self._worker_ga(**cfg)

    def _run_ridge(self, cfg: dict | None) -> None:
        if cfg is None:
            return
        self._set_status("Running Ridge regression…")
        self._worker_ridge(**cfg)

    def _run_lasso(self, cfg: dict | None) -> None:
        if cfg is None:
            return
        self._set_status("Running Lasso regression…")
        self._worker_lasso(**cfg)

    def _run_svr(self, cfg: dict | None) -> None:
        if cfg is None:
            return
        self._set_status("Running SVR…")
        self._worker_svr(**cfg)

    def _run_rf(self, cfg: dict | None) -> None:
        if cfg is None:
            return
        self._set_status("Running Random Forest…")
        self._worker_rf(**cfg)

    def _run_gbr(self, cfg: dict | None) -> None:
        if cfg is None:
            return
        self._set_status("Running Gradient Boosting…")
        self._worker_gbr(**cfg)

    # ── Workers ────────────────────────────────────────────────────────

    @work(thread=True)
    def _worker_ga(
        self,
        max_variables: int = 5,
        min_variables: int = 1,
        population_size: int = 100,
        max_generations: int = 100,
        mutation_rate: float = 0.01,
        keep_best: int = 10,
        exhaustive_max_vars: int = 3,
        fitness_function: str = "q2_loo",
        quik_delta: float | None = 0.05,
        n_workers: int = 1,
    ) -> None:
        try:
            self.app.project.build_ga_mlr(  # type: ignore[attr-defined]
                min_variables=min_variables,
                max_variables=max_variables,
                population_size=population_size,
                max_generations=max_generations,
                mutation_rate=mutation_rate,
                keep_best=keep_best,
                fitness_function=fitness_function,
                quik_delta=quik_delta,
                n_workers=n_workers,
                exhaustive_max_vars=exhaustive_max_vars,
            )
            self.app.call_from_thread(lambda: self._on_model_done("GA-MLR"))
        except Exception as exc:
            msg = str(exc)
            self.app.call_from_thread(
                lambda: self.notify(msg, title="GA-MLR failed", severity="error")
            )
            self.app.call_from_thread(lambda: self._set_status(""))

    @work(thread=True)
    def _worker_ridge(self, alpha: float = 1.0) -> None:
        try:
            self.app.project.build_ridge(alpha=alpha)  # type: ignore[attr-defined]
            self.app.call_from_thread(lambda: self._on_model_done("Ridge"))
        except Exception as exc:
            msg = str(exc)
            self.app.call_from_thread(
                lambda: self.notify(msg, title="Ridge failed", severity="error")
            )
            self.app.call_from_thread(lambda: self._set_status(""))

    @work(thread=True)
    def _worker_lasso(self, alpha: float = 0.1, max_iter: int = 10000) -> None:
        try:
            self.app.project.build_lasso(alpha=alpha, max_iter=max_iter)  # type: ignore[attr-defined]
            self.app.call_from_thread(lambda: self._on_model_done("Lasso"))
        except Exception as exc:
            msg = str(exc)
            self.app.call_from_thread(
                lambda: self.notify(msg, title="Lasso failed", severity="error")
            )
            self.app.call_from_thread(lambda: self._set_status(""))

    @work(thread=True)
    def _worker_svr(
        self, C: float = 1.0, gamma: str | float = "scale", kernel: str = "rbf"
    ) -> None:
        try:
            self.app.project.build_svr(C=C, gamma=gamma, kernel=kernel)  # type: ignore[attr-defined]
            self.app.call_from_thread(lambda: self._on_model_done("SVR"))
        except Exception as exc:
            msg = str(exc)
            self.app.call_from_thread(
                lambda: self.notify(msg, title="SVR failed", severity="error")
            )
            self.app.call_from_thread(lambda: self._set_status(""))

    @work(thread=True)
    def _worker_rf(self, n_estimators: int = 100) -> None:
        try:
            self.app.project.build_random_forest(  # type: ignore[attr-defined]
                n_estimators=n_estimators
            )
            self.app.call_from_thread(lambda: self._on_model_done("Random Forest"))
        except Exception as exc:
            msg = str(exc)
            self.app.call_from_thread(
                lambda: self.notify(msg, title="Random Forest failed", severity="error")
            )
            self.app.call_from_thread(lambda: self._set_status(""))

    @work(thread=True)
    def _worker_gbr(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
    ) -> None:
        try:
            self.app.project.build_gradient_boosting(  # type: ignore[attr-defined]
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
            )
            self.app.call_from_thread(lambda: self._on_model_done("Gradient Boosting"))
        except Exception as exc:
            msg = str(exc)
            self.app.call_from_thread(
                lambda: self.notify(
                    msg, title="Gradient Boosting failed", severity="error"
                )
            )
            self.app.call_from_thread(lambda: self._set_status(""))
