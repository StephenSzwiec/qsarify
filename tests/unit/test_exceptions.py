"""Tests for the custom exception hierarchy."""

import pytest

from qsarify.exceptions import (
    DataImportError,
    ModelFitError,
    PersistenceError,
    QSARifyError,
    ValidationError,
    WorkflowError,
)


def test_base_exception_is_exception() -> None:
    assert issubclass(QSARifyError, Exception)


def test_all_subclasses_inherit_base() -> None:
    for cls in (DataImportError, ModelFitError, ValidationError, WorkflowError, PersistenceError):
        assert issubclass(cls, QSARifyError)


def test_raise_and_catch_base() -> None:
    with pytest.raises(QSARifyError):
        raise QSARifyError("base error")


def test_raise_and_catch_as_base() -> None:
    with pytest.raises(QSARifyError):
        raise DataImportError("bad csv")


@pytest.mark.parametrize(
    "exc_cls",
    [DataImportError, ModelFitError, ValidationError, WorkflowError, PersistenceError],
)
def test_subclass_carries_message(exc_cls: type) -> None:
    msg = "test message"
    exc = exc_cls(msg)
    assert str(exc) == msg
