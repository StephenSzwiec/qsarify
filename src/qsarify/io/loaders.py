"""
Data loading utilities for QSARify.

This module provides functions for loading datasets from various file formats
into data structures suitable for the QSARify workflow.
"""
import pandas as pd
from typing import Tuple, Optional
from qsarify.exceptions import DataImportError

def load_csv_dataset(
    file_path: str,
    id_column: Optional[str] = None,
    response_column: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Loads a dataset from a CSV file.

    The function expects the CSV to have a specific structure:
    - A series of numeric columns representing the descriptors (X variables).
    - Optionally specify which column is the response variable (Y values), otherwise, it is assumed to be the last column.

    Args:
        file_path: The absolute path to the CSV file.
        id_column: Optional name of the column containing row descriptions.
        response_column: Optional name of the column containing the response variable (Y values).

    Returns:
        A tuple containing:
        - X_df (pd.DataFrame): DataFrame of descriptor variables, optionally named with IDs from the row description column.
        - y_series (pd.Series): Series of the response variable.

    Raises:
        DataImportError: If the file cannot be found, is empty, has fewer than
                         two columns, or contains non-numeric data in the
                         descriptor or response columns.
    """
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        raise DataImportError(f"Failed to load CSV file: {e}")

    if df.empty or df.shape[1] < 2:
        raise DataImportError("CSV file is empty or contains fewer than two columns.")

    if id_column and id_column not in df.columns:
        raise DataImportError(f"Specified ID column '{id_column}' not found in CSV.")

    if response_column:
        if response_column not in df.columns:
            raise DataImportError(f"Specified response column '{response_column}' not found in CSV.")
        y_series = df[response_column]
        X_df = df.drop(columns=[response_column])
    else:
        y_series = df.iloc[:, -1]
        X_df = df.iloc[:, :-1]

    if id_column:
        X_df.index = df[id_column]
        X_df = X_df.drop(columns=[id_column], errors='ignore')
    else:
        X_df.index = pd.RangeIndex(start=1, stop=X_df.shape[0]+1, step=1)

    
    # Check that all X and y values are numeric
    if not all(X_df.dtypes.apply(pd.api.types.is_numeric_dtype)):
        raise DataImportError("Non-numeric data found in descriptor columns")

    if not pd.api.types.is_numeric_dtype(y_series):
        raise DataImportError("Response column contains non-numeric data")

    return X_df, y_series
