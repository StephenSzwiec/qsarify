"""
Custom exception hierarchy for the QSARify application.

This module defines a base exception class, QSARifyError, from which all
application-specific exceptions inherit. This allows for a unified error
handling strategy, enabling consumers of the library to catch all QSARify-related
errors with a single `except QSARifyError:` block if they choose to.

Defining specific subclasses for different error conditions (e.g., DataImportError,
ModelFitError) allows for more granular error handling when required.
"""

class QSARifyError(Exception):
    """Base class for all exceptions raised by the QSARify library."""
    pass

class DataImportError(QSARifyError):
    """Raised for errors encountered during data loading and parsing."""
    pass

class ModelFitError(QSARifyError):
    """Raised for errors encountered during the model fitting process."""
    pass

class ValidationError(QSARifyError):
    """Raised for errors during model validation procedures."""
    pass

class PreprocessingError(QSARifyError):
    """Raised for errors during data preprocessing steps."""
    pass
