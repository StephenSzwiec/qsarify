"""DataSetupScreen: Step II — variable selection, normalisation, and train-test split."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Input, Label, Select, Static, Switch
from textual.widgets import SelectionList
from textual.widgets.selection_list import Selection
from textual import work

__all__ = ["DataSetupScreen"]


_SPLIT_OPTIONS: list[tuple[str, str]] = [
    ("Random split", "random"),
    ("Stratified (ordered by Y)", "stratified"),
]


class DataSetupScreen(Screen[None]):
    """Step II of the QSARINS workflow: Data Setup.

    - Displays the pre-assigned response variable.
    - Lets the user choose which descriptor columns to include.
    - Toggles z-score normalisation for X and/or Y.
    - Configures the train-test split fraction and method.
    - Calls :meth:`~qsarify.project.QSARProject.configure_variables` and
      advances to the Model Building screen.
    """

    def compose(self) -> ComposeResult:
        yield Header()
        yield Label("Step II  ·  Data Setup", classes="step-banner")
        with VerticalScroll():
            # ── Response variable ──────────────────────────────────────
            yield Label("Response Variable", classes="section-title")
            yield Static("(loading…)", id="y-col-label", classes="info-panel")

            # ── Descriptor columns ─────────────────────────────────────
            yield Label(
                "Descriptor Columns  (check to include)",
                classes="section-title",
            )
            yield SelectionList(id="x-selector")

            # ── Normalisation ──────────────────────────────────────────
            yield Label("Normalisation", classes="section-title")
            with Horizontal(classes="field-row"):
                yield Label("Normalize descriptors (X):", classes="field-label")
                yield Switch(value=False, id="sw-norm-x", classes="field-input")
            with Horizontal(classes="field-row"):
                yield Label("Normalize response (Y):", classes="field-label")
                yield Switch(value=False, id="sw-norm-y", classes="field-input")

            # ── Train-test split ───────────────────────────────────────
            yield Label("Train-Test Split", classes="section-title")
            with Horizontal(classes="field-row"):
                yield Label("Test fraction [0–1]:", classes="field-label")
                yield Input(value="0.20", id="test-size", classes="field-input")
            with Horizontal(classes="field-row"):
                yield Label("Split method:", classes="field-label")
                yield Select(
                    _SPLIT_OPTIONS,
                    id="split-method",
                    value="random",
                    allow_blank=False,
                )

            # ── Navigation ─────────────────────────────────────────────
            with Horizontal(classes="button-bar"):
                yield Button(
                    "Configure & Proceed to Models  →",
                    id="btn-configure",
                    variant="primary",
                    classes="primary-action",
                )
                yield Button("← Data Import", id="btn-back", variant="default")
        yield Footer()

    def on_mount(self) -> None:
        project = self.app.project  # type: ignore[attr-defined]
        if project.dataset is None:
            return

        ds = project.dataset
        y_name = ds.y_series.name
        self.query_one("#y-col-label", Static).update(
            f"Response column:  '{y_name}'   ({ds.y_series.shape[0]} samples)"
        )

        x_cols = (
            list(project._X_work.columns)
            if project._X_work is not None
            else list(ds.X_df.columns)
        )
        sel = self.query_one("#x-selector", SelectionList)
        for col in x_cols:
            sel.add_option(Selection(col, col, True))

    # ── Button handler ─────────────────────────────────────────────────

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-configure":
            self._do_configure()
        elif event.button.id == "btn-back":
            self.app.pop_screen()

    # ── Read form and dispatch worker ──────────────────────────────────

    def _do_configure(self) -> None:
        project = self.app.project  # type: ignore[attr-defined]
        if project.dataset is None:
            self.notify("No dataset loaded.", severity="error")
            return

        y_col: str = project.dataset.y_series.name
        x_cols_selected: list[str] = list(
            self.query_one("#x-selector", SelectionList).selected
        )
        if not x_cols_selected:
            self.notify("Select at least one descriptor column.", severity="warning")
            return

        norm_x: bool = self.query_one("#sw-norm-x", Switch).value
        norm_y: bool = self.query_one("#sw-norm-y", Switch).value

        try:
            test_size = float(self.query_one("#test-size", Input).value)
        except ValueError:
            test_size = 0.20

        split_val = self.query_one("#split-method", Select).value
        split_method = str(split_val) if split_val else "random"

        self._run_configure(y_col, x_cols_selected, norm_x, norm_y, test_size, split_method)

    @work(thread=True)
    def _run_configure(
        self,
        y_col: str,
        x_cols: list[str],
        normalize_x: bool,
        normalize_y: bool,
        test_size: float,
        split_method: str,
    ) -> None:
        try:
            self.app.project.configure_variables(  # type: ignore[attr-defined]
                y_col=y_col,
                x_cols=x_cols,
                normalize_x=normalize_x,
                normalize_y=normalize_y,
                test_size=test_size,
                split_method=split_method,
            )
            self.app.call_from_thread(self._on_configure_done)
        except Exception as exc:
            msg = str(exc)
            self.app.call_from_thread(
                lambda: self.notify(msg, title="Configuration failed", severity="error")
            )

    def _on_configure_done(self) -> None:
        p = self.app.project  # type: ignore[attr-defined]
        n_tr = p.X_train.shape[0] if p.X_train is not None else 0
        n_te = p.X_test.shape[0] if p.X_test is not None else 0
        n_feat = len(p.descriptor_names)
        self.notify(
            f"Train: {n_tr}  |  Test: {n_te}  |  Descriptors: {n_feat}",
            title="Data configured",
        )
        self.app.go_to_model_building()  # type: ignore[attr-defined]
