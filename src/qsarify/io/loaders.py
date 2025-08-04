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
    description_column_name: Optional[str] = None,
    response_column_name: Optional[str] = None,

    ) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Loads a dataset from a CSV file.

    The function expects the CSV to have a specific structure:
    - An optional column for row descriptions (if specified). 
    - A series of numeric columns representing the descriptors (X variables).
    - Optionally specify which column is the response variable (Y values), otherwise, it is assumed to be the last column.

    Args:
        file_path: The absolute path to the CSV file.
        description_column_name: Optional name of the column containing row descriptions.
        response_column_name: Optional name of the column containing the response variable (Y values).

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
        df = pd.read_csv(file_path, index_col=False)
    except FileNotFoundError:
        raise DataImportError(f"The file was not found at: {file_path}")
    except pd.errors.EmptyDataError:
        raise DataImportError(f"The file is empty: {file_path}")

    if df.shape[1] < 2:
        raise DataImportError(
            "The dataset must have at least two columns (one descriptor and one response)."
        )

    # Identify the response column
    if response_column_name:
        if response_column_name not in df.columns:
            raise DataImportError(f"Response column '{response_column_name}' not found in the dataset.")
        y_series = df[response_column_name]
        temp_df = df.drop(columns=[response_column_name])
    else:
        y_series = df.iloc[:, -1]
        temp_df = df.iloc[:, :-1]

    # Identify IDs and X_df
    if description_column_name:
        if description_column_name not in temp_df.columns:
            raise DataImportError(f"Description column '{description_column_name}' not found in the dataset.")
        ids_series = temp_df[description_column_name]
        X_df = temp_df.drop(columns=[description_column_name]) 
    else:
        ids_series = pd.Series(range(len(temp_df)), name='ID')
        X_df = temp_df
    X_df.columns = [f"Descriptor_{i+1}" for i in range(X_df.shape[1])]

    # Validate data types
    if not pd.api.types.is_numeric_dtype(y_series):
        raise DataImportError("The response variable (last column) must be numeric.")

    # Validate data types
    if not pd.api.types.is_numeric_dtype(y_series):
        raise DataImportError("The response variable (last column) must be numeric.")

    non_numeric_descriptors = ~X_df.apply(pd.api.types.is_numeric_dtype)

    print(f"[DEBUG] X_df dtypes: {X_df.dtypes}")
    print(f"[DEBUG] non_numeric_descriptors: {non_numeric_descriptors}")
    if non_numeric_descriptors.any():
        bad_cols = X_df.columns[non_numeric_descriptors].tolist()
        raise DataImportError(
            f"All descriptor columns must be numeric. Found non-numeric data in: {bad_cols}"
        )

    return X_df, y_series
