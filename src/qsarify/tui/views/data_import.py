"""DataImportScreen: Step I — load a CSV dataset and optionally filter descriptors."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Input, Label, Static
from textual import work

__all__ = ["DataImportScreen"]


class DataImportScreen(Screen[None]):
    """Step I of the QSARINS workflow: Data Import.

    - Load a CSV dataset via file path input.
    - Optionally filter near-constant and highly correlated descriptors.
    - Advance to the Data Setup screen.
    """

    def compose(self) -> ComposeResult:
        yield Header()
        yield Label("Step I  ·  Data Import", classes="step-banner")
        with VerticalScroll():
            # ── Load CSV ───────────────────────────────────────────────
            yield Label("Load Dataset", classes="section-title")
            with Horizontal(classes="field-row"):
                yield Label("CSV path:", classes="field-label")
                yield Input(
                    placeholder="path/to/dataset.csv",
                    id="csv-path",
                    classes="field-input",
                )
            with Horizontal(classes="button-bar"):
                yield Button("Load Dataset", id="btn-load", variant="primary")

            yield Static(
                "No dataset loaded.",
                id="status-panel",
                classes="info-panel muted",
            )

            # ── Filter descriptors ─────────────────────────────────────
            yield Label("Filter Descriptors  (optional)", classes="section-title")
            with Horizontal(classes="field-row"):
                yield Label("Constant threshold:", classes="field-label")
                yield Input(value="0.01", id="const-thresh", classes="field-input")
            with Horizontal(classes="field-row"):
                yield Label("Correlation threshold:", classes="field-label")
                yield Input(value="0.90", id="corr-thresh", classes="field-input")
            with Horizontal(classes="button-bar"):
                yield Button(
                    "Apply Filters",
                    id="btn-filter",
                    variant="default",
                    disabled=True,
                )

            # ── Navigation ─────────────────────────────────────────────
            with Horizontal(classes="button-bar"):
                yield Button(
                    "Proceed to Data Setup  →",
                    id="btn-proceed",
                    variant="primary",
                    classes="primary-action",
                    disabled=True,
                )
                yield Button("← Welcome", id="btn-back", variant="default")
        yield Footer()

    def on_mount(self) -> None:
        self._refresh_status()

    # ── Status helpers ─────────────────────────────────────────────────

    def _refresh_status(self) -> None:
        project = self.app.project  # type: ignore[attr-defined]
        panel = self.query_one("#status-panel", Static)
        if project.dataset is None:
            panel.update("No dataset loaded.")
            panel.remove_class("success")
            panel.add_class("muted")
            self.query_one("#btn-filter", Button).disabled = True
            self.query_one("#btn-proceed", Button).disabled = True
        else:
            ds = project.dataset
            n, p_raw = ds.X_df.shape
            p_work = project._X_work.shape[1] if project._X_work is not None else p_raw
            y_name = ds.y_series.name
            lines = [
                f"Loaded:  {n} samples  |  {p_raw} raw descriptors  |  response = '{y_name}'",
                f"Working descriptors (after filtering): {p_work}",
            ]
            panel.update("\n".join(lines))
            panel.remove_class("muted")
            panel.add_class("success")
            self.query_one("#btn-filter", Button).disabled = False
            self.query_one("#btn-proceed", Button).disabled = False

    # ── Button handler ─────────────────────────────────────────────────

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id
        if bid == "btn-load":
            path = self.query_one("#csv-path", Input).value.strip()
            if path:
                self._do_load(path)
            else:
                self.notify("Enter a CSV file path first.", severity="warning")
        elif bid == "btn-filter":
            try:
                ct = float(self.query_one("#const-thresh", Input).value)
            except ValueError:
                ct = 0.01
            try:
                corr = float(self.query_one("#corr-thresh", Input).value)
            except ValueError:
                corr = 0.90
            self._do_filter(ct, corr)
        elif bid == "btn-proceed":
            self.app.go_to_data_setup()  # type: ignore[attr-defined]
        elif bid == "btn-back":
            self.app.pop_screen()

    # ── Background workers ─────────────────────────────────────────────

    @work(thread=True)
    def _do_load(self, path: str) -> None:
        try:
            self.app.project.load_data(path)  # type: ignore[attr-defined]
            self.app.call_from_thread(self._on_load_done)
        except Exception as exc:
            msg = str(exc)
            self.app.call_from_thread(
                lambda: self.notify(msg, title="Load failed", severity="error")
            )

    def _on_load_done(self) -> None:
        ds = self.app.project.dataset  # type: ignore[attr-defined]
        n, p = ds.X_df.shape
        self.notify(
            f"Loaded {n} samples × {p} descriptors.",
            title="Dataset ready",
        )
        self._refresh_status()

    @work(thread=True)
    def _do_filter(
        self, constant_threshold: float, correlation_threshold: float
    ) -> None:
        try:
            self.app.project.filter_descriptors(  # type: ignore[attr-defined]
                constant_threshold=constant_threshold,
                correlation_threshold=correlation_threshold,
            )
            self.app.call_from_thread(self._on_filter_done)
        except Exception as exc:
            msg = str(exc)
            self.app.call_from_thread(
                lambda: self.notify(msg, title="Filter failed", severity="error")
            )

    def _on_filter_done(self) -> None:
        p = self.app.project._X_work.shape[1]  # type: ignore[attr-defined]
        self.notify(f"{p} descriptors remain after filtering.", title="Filters applied")
        self._refresh_status()
