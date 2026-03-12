"""Entry point for ``python -m qsarify``."""

from qsarify.tui.app import QSARifyApp


def main() -> None:
    """Launch the QSARify TUI application."""
    app = QSARifyApp()
    app.run()


if __name__ == "__main__":
    main()
