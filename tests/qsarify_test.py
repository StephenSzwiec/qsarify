from qsarify import main


def test_main_is_callable() -> None:
    """Verify that the main entry point is importable and callable."""
    assert callable(main)
