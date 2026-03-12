"""WelcomeScreen: entry point displayed in the EMPTY project state."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Center, Vertical
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Label

__all__ = ["WelcomeScreen"]


class WelcomeScreen(Screen[None]):
    """Landing screen shown when the project state is EMPTY.

    Provides three actions: start a new (blank) project, open an existing
    project from a SQLite3 save file, or quit the application.
    """

    CSS = """
    #welcome-box {
        width: 50;
        height: auto;
        padding: 2 4;
        border: solid $primary;
        background: $surface;
    }
    #welcome-logo {
        text-style: bold;
        color: $primary-lighten-1;
        text-align: center;
        width: 100%;
        margin-bottom: 0;
    }
    #welcome-tagline {
        color: $text-muted;
        text-align: center;
        width: 100%;
        margin-bottom: 2;
    }
    .welcome-btn {
        width: 100%;
        margin-bottom: 1;
    }
    """

    def compose(self) -> ComposeResult:
        yield Header()
        with Center():
            with Vertical(id="welcome-box"):
                yield Label("QSARify", id="welcome-logo")
                yield Label(
                    "QSAR/QSPR Modeling Workflow",
                    id="welcome-tagline",
                )
                yield Button(
                    "New Project",
                    id="btn-new",
                    variant="primary",
                    classes="welcome-btn",
                )
                yield Button(
                    "Open Project…",
                    id="btn-open",
                    variant="default",
                    classes="welcome-btn",
                )
                yield Button(
                    "Quit",
                    id="btn-quit",
                    variant="error",
                    classes="welcome-btn",
                )
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id
        if bid == "btn-new":
            self.app.go_to_data_import()  # type: ignore[attr-defined]
        elif bid == "btn-open":
            from qsarify.tui.views._modals import LoadModal

            self.app.push_screen(LoadModal(), self._do_load)
        elif bid == "btn-quit":
            self.app.exit()

    def _do_load(self, path: str | None) -> None:
        if not path:
            return
        try:
            from qsarify.project import QSARProject

            self.app.project = QSARProject.load(path)  # type: ignore[attr-defined]
            self.app.notify(f"Opened {path}", title="Project loaded")  # type: ignore[attr-defined]
            self.app._go_to_state_screen(self.app.project.state)  # type: ignore[attr-defined]
        except Exception as exc:
            self.app.notify(str(exc), title="Load failed", severity="error")  # type: ignore[attr-defined]
