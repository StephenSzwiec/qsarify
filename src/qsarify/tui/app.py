"""QSARify TUI — main application class.

The ``QSARifyApp`` owns a single :class:`~qsarify.project.QSARProject`
instance and is the only object that modifies it.  Screens are stateless
presentation layers that read from and call into the project through
callbacks passed at construction time.

Navigation follows the FSM directly:

    EMPTY         → WelcomeScreen
    DATA_IMPORTED → DataImportScreen
    DATA_CONFIGURED → ModelBuildingScreen
    MODELS_BUILT / MODELS_EVALUATED → ResultsScreen
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, Header

from qsarify.project import ProjectState, QSARProject

__all__ = ["QSARifyApp"]

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

_STEP_LABELS: dict[ProjectState, str] = {
    ProjectState.EMPTY: "Welcome",
    ProjectState.DATA_IMPORTED: "I · Data Import",
    ProjectState.DATA_CONFIGURED: "II · Data Setup  →  III · Model Building",
    ProjectState.MODELS_BUILT: "IV · Results",
    ProjectState.MODELS_EVALUATED: "IV · Results",
}


class QSARifyApp(App[None]):
    """Root Textual application for the QSARify guided workflow.

    Parameters
    ----------
    project : QSARProject or None
        Pre-existing project to resume.  ``None`` creates a blank project.
    checkpoint_path : Path or None
        Forwarded to the :class:`~qsarify.project.QSARProject` constructor
        for automatic checkpointing.  ``None`` disables auto-save.
    """

    TITLE = "QSARify  |  QSAR/QSPR Modeling Workflow"
    SUB_TITLE = "Step: Welcome"

    CSS = """
    /* ── Global ────────────────────────────────────────────── */
    Screen {
        background: $background;
        layers: base overlay;
    }

    /* ── Workflow step banner ───────────────────────────────── */
    .step-banner {
        background: $primary;
        color: $text;
        height: 1;
        padding: 0 2;
        text-style: bold;
    }

    /* ── Section titles ─────────────────────────────────────── */
    .section-title {
        background: $primary-darken-2;
        color: $text;
        padding: 0 2;
        margin-bottom: 1;
        text-style: bold;
    }

    /* ── Labelled rows ──────────────────────────────────────── */
    .field-row {
        height: auto;
        padding: 0 2;
        margin-bottom: 1;
        layout: horizontal;
    }
    .field-label {
        width: 28;
        padding-top: 1;
        text-align: right;
    }
    .field-input {
        width: 1fr;
    }

    /* ── Info / status panels ───────────────────────────────── */
    .info-panel {
        border: solid $primary;
        padding: 1 2;
        margin: 1 2;
        height: auto;
    }
    .success { color: $success; }
    .error   { color: $error;   }
    .muted   { color: $text-muted; }

    /* ── Action button bar ──────────────────────────────────── */
    .button-bar {
        layout: horizontal;
        height: auto;
        padding: 1 2;
        margin-top: 1;
    }
    .button-bar Button {
        margin-right: 1;
    }
    Button.primary-action {
        background: $primary;
    }
    Button.danger-action {
        background: $error;
    }

    /* ── Modal dialogs ──────────────────────────────────────── */
    ModalScreen {
        background: $background 60%;
    }
    .modal-container {
        background: $surface;
        border: solid $primary;
        padding: 1 2;
        width: 70;
        height: auto;
        max-height: 80%;
    }
    .modal-title {
        text-style: bold;
        margin-bottom: 1;
        color: $primary-lighten-1;
    }
    .modal-buttons {
        layout: horizontal;
        height: auto;
        margin-top: 1;
    }
    .modal-buttons Button {
        margin-right: 1;
    }
    """

    BINDINGS = [
        Binding("ctrl+s", "save_project", "Save", show=True),
        Binding("ctrl+q", "quit", "Quit", show=True),
    ]

    def __init__(
        self,
        project: QSARProject | None = None,
        checkpoint_path: Path | None = None,
    ) -> None:
        super().__init__()
        if project is not None:
            self.project = project
        else:
            self.project = QSARProject(checkpoint_path=checkpoint_path)

    # -----------------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------------

    def on_mount(self) -> None:
        """Navigate to the screen matching the current project state."""
        self._go_to_state_screen(self.project.state)

    # -----------------------------------------------------------------------
    # Navigation helpers (called by screens)
    # -----------------------------------------------------------------------

    def _go_to_state_screen(self, state: ProjectState) -> None:
        """Push the canonical screen for *state*, clearing the stack first."""
        # Import lazily to avoid circular deps and speed startup
        from qsarify.tui.views.data_import import DataImportScreen
        from qsarify.tui.views.data_setup import DataSetupScreen
        from qsarify.tui.views.model_building import ModelBuildingScreen
        from qsarify.tui.views.results import ResultsScreen
        from qsarify.tui.views.welcome import WelcomeScreen

        # Pop everything and push the target screen
        while self.screen_stack:
            try:
                self.pop_screen()
            except Exception:
                break

        self.sub_title = f"Step: {_STEP_LABELS.get(state, '')}"

        if state == ProjectState.EMPTY:
            self.push_screen(WelcomeScreen())
        elif state == ProjectState.DATA_IMPORTED:
            self.push_screen(DataImportScreen())
        elif state == ProjectState.DATA_CONFIGURED:
            self.push_screen(ModelBuildingScreen())
        else:  # MODELS_BUILT or MODELS_EVALUATED
            self.push_screen(ResultsScreen())

    def go_to_data_import(self) -> None:
        """Navigate to the Data Import screen (Step I)."""
        from qsarify.tui.views.data_import import DataImportScreen

        self.sub_title = f"Step: {_STEP_LABELS[ProjectState.DATA_IMPORTED]}"
        self.push_screen(DataImportScreen())

    def go_to_data_setup(self) -> None:
        """Navigate to the Data Setup screen (Step II)."""
        from qsarify.tui.views.data_setup import DataSetupScreen

        self.sub_title = f"Step: {_STEP_LABELS[ProjectState.DATA_IMPORTED]}"
        self.push_screen(DataSetupScreen())

    def go_to_model_building(self) -> None:
        """Navigate to the Model Building screen (Step III)."""
        from qsarify.tui.views.model_building import ModelBuildingScreen

        self.sub_title = f"Step: {_STEP_LABELS[ProjectState.DATA_CONFIGURED]}"
        self.push_screen(ModelBuildingScreen())

    def go_to_results(self) -> None:
        """Navigate to the Results screen (Step IV)."""
        from qsarify.tui.views.results import ResultsScreen

        self.sub_title = f"Step: {_STEP_LABELS[ProjectState.MODELS_BUILT]}"
        self.push_screen(ResultsScreen())

    # -----------------------------------------------------------------------
    # Global key actions
    # -----------------------------------------------------------------------

    def action_save_project(self) -> None:
        """Show the save-project modal from anywhere."""
        from qsarify.tui.views._modals import SaveModal

        def _do_save(path: str | None) -> None:
            if path:
                try:
                    self.project.save(path)
                    self.notify(f"Project saved to {path}", title="Saved")
                except Exception as exc:
                    self.notify(str(exc), title="Save failed", severity="error")

        self.push_screen(SaveModal(), _do_save)

    async def action_quit(self) -> None:
        self.exit()
