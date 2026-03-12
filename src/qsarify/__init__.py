def main() -> None:
    """Launch the QSARify TUI (entry point for ``qsarify`` CLI script)."""
    from qsarify.tui.app import QSARifyApp

    app = QSARifyApp()
    app.run()
