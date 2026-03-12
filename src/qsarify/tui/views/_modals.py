"""Reusable modal dialogs for the QSARify TUI.

All modals are :class:`textual.app.ModalScreen` subclasses.  They accept
a result callback and dismiss themselves with the collected value when the
user confirms, or with ``None`` when the user cancels.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.screen import ModalScreen
from textual.containers import Vertical
from textual.widgets import Button, Input, Label, Select

__all__ = [
    "ConfirmModal",
    "SaveModal",
    "LoadModal",
]


# ---------------------------------------------------------------------------
# Confirm / cancel
# ---------------------------------------------------------------------------


class ConfirmModal(ModalScreen[bool]):
    """Yes / No confirmation dialog.

    Parameters
    ----------
    question : str
        The question to display.
    confirm_label : str
        Label for the confirm button.  Default ``"Yes, proceed"``.
    """

    def __init__(
        self,
        question: str,
        confirm_label: str = "Yes, proceed",
    ) -> None:
        super().__init__()
        self._question = question
        self._confirm_label = confirm_label

    def compose(self) -> ComposeResult:
        with Vertical(classes="modal-container"):
            yield Label("⚠  Confirmation Required", classes="modal-title")
            yield Label(self._question)
            with Vertical(classes="modal-buttons"):
                yield Button(self._confirm_label, id="confirm", variant="warning")
                yield Button("Cancel", id="cancel", variant="default")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "confirm")


# ---------------------------------------------------------------------------
# File path inputs
# ---------------------------------------------------------------------------


class SaveModal(ModalScreen[str | None]):
    """Prompt the user for a file path to save the project."""

    DEFAULT_PATH = "project.sqlite3"

    def compose(self) -> ComposeResult:
        with Vertical(classes="modal-container"):
            yield Label("Save Project", classes="modal-title")
            yield Label("File path:")
            yield Input(
                value=self.DEFAULT_PATH,
                placeholder="project.sqlite3",
                id="path_input",
            )
            with Vertical(classes="modal-buttons"):
                yield Button("Save", id="save", variant="primary")
                yield Button("Cancel", id="cancel", variant="default")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "save":
            path = self.query_one("#path_input", Input).value.strip()
            self.dismiss(path if path else None)
        else:
            self.dismiss(None)


class LoadModal(ModalScreen[str | None]):
    """Prompt the user for a file path to load a project."""

    def compose(self) -> ComposeResult:
        with Vertical(classes="modal-container"):
            yield Label("Open Project", classes="modal-title")
            yield Label("File path:")
            yield Input(
                placeholder="project.sqlite3",
                id="path_input",
            )
            with Vertical(classes="modal-buttons"):
                yield Button("Open", id="open", variant="primary")
                yield Button("Cancel", id="cancel", variant="default")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "open":
            path = self.query_one("#path_input", Input).value.strip()
            self.dismiss(path if path else None)
        else:
            self.dismiss(None)


# ---------------------------------------------------------------------------
# GA-MLR configuration modal
# ---------------------------------------------------------------------------


class GAMLRConfigModal(ModalScreen[dict | None]):
    """Configure GA-MLR parameters before running."""

    _FITNESS_OPTIONS = [
        ("Q²_LOO (default)", "q2_loo"),
        ("R²_adj", "r2_adj"),
        ("LOF (Friedman's)", "lof"),
        ("RMSE_CV", "rmse_cv"),
    ]

    _QUIK_OPTIONS = [
        ("0.05 (default)", "0.05"),
        ("Disabled", "none"),
        ("0.0001", "0.0001"),
        ("0.001", "0.001"),
        ("0.01", "0.01"),
        ("0.10", "0.10"),
        ("0.15", "0.15"),
        ("0.20", "0.20"),
    ]

    def compose(self) -> ComposeResult:
        with Vertical(classes="modal-container"):
            yield Label("GA-MLR Configuration", classes="modal-title")

            yield Label("Max variables (≤ n/5):")
            yield Input(value="5", id="max_vars", placeholder="5")

            yield Label("Min variables:")
            yield Input(value="1", id="min_vars", placeholder="1")

            yield Label("Population size:")
            yield Input(value="100", id="pop_size", placeholder="100")

            yield Label("Max generations:")
            yield Input(value="100", id="max_gen", placeholder="100")

            yield Label("Mutation rate [0–1]:")
            yield Input(value="0.01", id="mut_rate", placeholder="0.01")

            yield Label("Keep best (per variable count):")
            yield Input(value="10", id="keep_best", placeholder="10")

            yield Label("Exhaustive enum (1–3 vars, 0=skip):")
            yield Input(value="3", id="exhaust_vars", placeholder="3")

            yield Label("Fitness function:")
            yield Select(
                self._FITNESS_OPTIONS,
                id="fitness_fn",
                value="q2_loo",
                allow_blank=False,
            )

            yield Label("QUIK rule δ_k:")
            yield Select(
                self._QUIK_OPTIONS,
                id="quik_delta",
                value="0.05",
                allow_blank=False,
            )

            yield Label("Workers (1=sequential):")
            yield Input(value="1", id="n_workers", placeholder="1")

            with Vertical(classes="modal-buttons"):
                yield Button("Run GA-MLR", id="run", variant="primary")
                yield Button("Cancel", id="cancel", variant="default")

    def _int(self, widget_id: str, default: int) -> int:
        try:
            return int(self.query_one(widget_id, Input).value)
        except (ValueError, TypeError):
            return default

    def _float(self, widget_id: str, default: float) -> float:
        try:
            return float(self.query_one(widget_id, Input).value)
        except (ValueError, TypeError):
            return default

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id != "run":
            self.dismiss(None)
            return

        quik_raw = str(self.query_one("#quik_delta", Select).value)
        quik_delta: float | None = None if quik_raw == "none" else float(quik_raw)

        fitness_val = self.query_one("#fitness_fn", Select).value
        fitness_fn = str(fitness_val) if fitness_val else "q2_loo"

        self.dismiss(
            {
                "max_variables": self._int("#max_vars", 5),
                "min_variables": self._int("#min_vars", 1),
                "population_size": self._int("#pop_size", 100),
                "max_generations": self._int("#max_gen", 100),
                "mutation_rate": self._float("#mut_rate", 0.01),
                "keep_best": self._int("#keep_best", 10),
                "exhaustive_max_vars": self._int("#exhaust_vars", 3),
                "fitness_function": fitness_fn,
                "quik_delta": quik_delta,
                "n_workers": self._int("#n_workers", 1),
            }
        )


# ---------------------------------------------------------------------------
# Sklearn model configuration modals
# ---------------------------------------------------------------------------


class RidgeConfigModal(ModalScreen[dict | None]):
    """Configure Ridge regression parameters."""

    def compose(self) -> ComposeResult:
        with Vertical(classes="modal-container"):
            yield Label("Ridge Regression", classes="modal-title")
            yield Label("Alpha (L2 strength):")
            yield Input(value="1.0", id="alpha", placeholder="1.0")
            with Vertical(classes="modal-buttons"):
                yield Button("Run Ridge", id="run", variant="primary")
                yield Button("Cancel", id="cancel", variant="default")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id != "run":
            self.dismiss(None)
            return
        try:
            alpha = float(self.query_one("#alpha", Input).value)
        except ValueError:
            alpha = 1.0
        self.dismiss({"alpha": alpha})


class LassoConfigModal(ModalScreen[dict | None]):
    """Configure Lasso regression parameters."""

    def compose(self) -> ComposeResult:
        with Vertical(classes="modal-container"):
            yield Label("Lasso Regression", classes="modal-title")
            yield Label("Alpha (L1 strength):")
            yield Input(value="0.1", id="alpha", placeholder="0.1")
            yield Label("Max iterations:")
            yield Input(value="10000", id="max_iter", placeholder="10000")
            with Vertical(classes="modal-buttons"):
                yield Button("Run Lasso", id="run", variant="primary")
                yield Button("Cancel", id="cancel", variant="default")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id != "run":
            self.dismiss(None)
            return
        try:
            alpha = float(self.query_one("#alpha", Input).value)
        except ValueError:
            alpha = 0.1
        try:
            max_iter = int(self.query_one("#max_iter", Input).value)
        except ValueError:
            max_iter = 10000
        self.dismiss({"alpha": alpha, "max_iter": max_iter})


class SVRConfigModal(ModalScreen[dict | None]):
    """Configure Support Vector Regression parameters."""

    _KERNEL_OPTIONS = [
        ("RBF (default)", "rbf"),
        ("Linear", "linear"),
        ("Polynomial", "poly"),
    ]

    def compose(self) -> ComposeResult:
        with Vertical(classes="modal-container"):
            yield Label("Support Vector Regression", classes="modal-title")
            yield Label("C (regularisation):")
            yield Input(value="1.0", id="C", placeholder="1.0")
            yield Label("Gamma:")
            yield Input(value="scale", id="gamma", placeholder="scale")
            yield Label("Kernel:")
            yield Select(
                self._KERNEL_OPTIONS,
                id="kernel",
                value="rbf",
                allow_blank=False,
            )
            with Vertical(classes="modal-buttons"):
                yield Button("Run SVR", id="run", variant="primary")
                yield Button("Cancel", id="cancel", variant="default")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id != "run":
            self.dismiss(None)
            return
        try:
            C = float(self.query_one("#C", Input).value)
        except ValueError:
            C = 1.0
        gamma_raw = self.query_one("#gamma", Input).value.strip()
        gamma: str | float = (
            gamma_raw
            if gamma_raw in ("scale", "auto")
            else float(gamma_raw)
            if gamma_raw
            else "scale"
        )
        kernel_val = self.query_one("#kernel", Select).value
        kernel = str(kernel_val) if kernel_val else "rbf"
        self.dismiss({"C": C, "gamma": gamma, "kernel": kernel})


class RFConfigModal(ModalScreen[dict | None]):
    """Configure Random Forest parameters."""

    def compose(self) -> ComposeResult:
        with Vertical(classes="modal-container"):
            yield Label("Random Forest Regression", classes="modal-title")
            yield Label("Number of trees:")
            yield Input(value="100", id="n_est", placeholder="100")
            with Vertical(classes="modal-buttons"):
                yield Button("Run Random Forest", id="run", variant="primary")
                yield Button("Cancel", id="cancel", variant="default")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id != "run":
            self.dismiss(None)
            return
        try:
            n = int(self.query_one("#n_est", Input).value)
        except ValueError:
            n = 100
        self.dismiss({"n_estimators": n})


class GBRConfigModal(ModalScreen[dict | None]):
    """Configure Gradient Boosting parameters."""

    def compose(self) -> ComposeResult:
        with Vertical(classes="modal-container"):
            yield Label("Gradient Boosting Regression", classes="modal-title")
            yield Label("Number of stages:")
            yield Input(value="100", id="n_est", placeholder="100")
            yield Label("Learning rate:")
            yield Input(value="0.1", id="lr", placeholder="0.1")
            yield Label("Max tree depth:")
            yield Input(value="3", id="depth", placeholder="3")
            with Vertical(classes="modal-buttons"):
                yield Button("Run Gradient Boosting", id="run", variant="primary")
                yield Button("Cancel", id="cancel", variant="default")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id != "run":
            self.dismiss(None)
            return
        try:
            n = int(self.query_one("#n_est", Input).value)
        except ValueError:
            n = 100
        try:
            lr = float(self.query_one("#lr", Input).value)
        except ValueError:
            lr = 0.1
        try:
            d = int(self.query_one("#depth", Input).value)
        except ValueError:
            d = 3
        self.dismiss({"n_estimators": n, "learning_rate": lr, "max_depth": d})


# ---------------------------------------------------------------------------
# Validation configuration
# ---------------------------------------------------------------------------


class LMOConfigModal(ModalScreen[dict | None]):
    """Configure Leave-Many-Out validation parameters."""

    _HOLDOUT_OPTIONS = [
        ("30% (default)", "0.30"),
        ("20%", "0.20"),
        ("33%", "0.33"),
    ]

    def compose(self) -> ComposeResult:
        with Vertical(classes="modal-container"):
            yield Label("LMO Cross-Validation", classes="modal-title")
            yield Label("Holdout fraction:")
            yield Select(
                self._HOLDOUT_OPTIONS,
                id="holdout",
                value="0.30",
                allow_blank=False,
            )
            yield Label("Iterations:")
            yield Input(value="100", id="n_iter", placeholder="100")
            with Vertical(classes="modal-buttons"):
                yield Button("Run LMO", id="run", variant="primary")
                yield Button("Cancel", id="cancel", variant="default")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id != "run":
            self.dismiss(None)
            return
        try:
            n = int(self.query_one("#n_iter", Input).value)
        except ValueError:
            n = 100
        holdout_raw = self.query_one("#holdout", Select).value
        holdout = float(str(holdout_raw)) if holdout_raw else 0.30
        self.dismiss({"holdout_fraction": holdout, "n_iterations": n})


class YScramblingConfigModal(ModalScreen[dict | None]):
    """Configure Y-scrambling parameters."""

    def compose(self) -> ComposeResult:
        with Vertical(classes="modal-container"):
            yield Label("Y-Scrambling Validation", classes="modal-title")
            yield Label("Iterations (≥100 recommended):")
            yield Input(value="100", id="n_iter", placeholder="100")
            with Vertical(classes="modal-buttons"):
                yield Button("Run Y-Scrambling", id="run", variant="primary")
                yield Button("Cancel", id="cancel", variant="default")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id != "run":
            self.dismiss(None)
            return
        try:
            n = int(self.query_one("#n_iter", Input).value)
        except ValueError:
            n = 100
        self.dismiss({"n_iterations": max(n, 100)})


# ---------------------------------------------------------------------------
# Plot configuration
# ---------------------------------------------------------------------------


class PlotConfigModal(ModalScreen[dict | None]):
    """Select plot type and output path."""

    _PLOT_OPTIONS = [
        ("Residuals vs Fitted", "residuals"),
        ("Q-Q Plot", "qq"),
        ("Williams Plot (Applicability Domain)", "williams"),
        ("Y-Scrambling Results", "y_scrambling"),
    ]

    def compose(self) -> ComposeResult:
        with Vertical(classes="modal-container"):
            yield Label("Generate Plot", classes="modal-title")
            yield Label("Plot type:")
            yield Select(
                self._PLOT_OPTIONS,
                id="plot_type",
                value="residuals",
                allow_blank=False,
            )
            yield Label("Save path:")
            yield Input(
                value="plot.png",
                id="output_path",
                placeholder="plot.png",
            )
            with Vertical(classes="modal-buttons"):
                yield Button("Generate", id="generate", variant="primary")
                yield Button("Cancel", id="cancel", variant="default")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id != "generate":
            self.dismiss(None)
            return
        plot_val = self.query_one("#plot_type", Select).value
        plot_type = str(plot_val) if plot_val else "residuals"
        path = self.query_one("#output_path", Input).value.strip() or "plot.png"
        self.dismiss({"plot_type": plot_type, "output_path": path})
