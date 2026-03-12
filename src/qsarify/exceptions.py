"""Custom exception hierarchy for QSARify.

All QSARify-specific errors inherit from :class:`QSARifyError` so callers can
catch the entire family with a single ``except QSARifyError`` clause.
"""


class QSARifyError(Exception):
    """Base exception for all QSARify errors."""


class DataImportError(QSARifyError):
    """Raised when data import or parsing fails.

    Parameters
    ----------
    message : str
        Human-readable description of the import failure.
    """


class ModelFitError(QSARifyError):
    """Raised when model fitting fails.

    Parameters
    ----------
    message : str
        Human-readable description of the fitting failure.
    """


class ValidationError(QSARifyError):
    """Raised when a validation procedure encounters an error.

    Parameters
    ----------
    message : str
        Human-readable description of the validation failure.
    """


class WorkflowError(QSARifyError):
    """Raised when an operation is attempted in an invalid FSM state.

    Parameters
    ----------
    message : str
        Human-readable description of the workflow violation.
    """


class PersistenceError(QSARifyError):
    """Raised when project save or load operations fail.

    Parameters
    ----------
    message : str
        Human-readable description of the persistence failure.
    """


class WorkflowRegressionWarning(UserWarning):
    """Issued when an operation causes a backward FSM transition.

    A backward transition clears all downstream state (models, validation
    results).  The TUI presents a confirmation dialog; the library API issues
    this warning automatically before clearing.
    """
