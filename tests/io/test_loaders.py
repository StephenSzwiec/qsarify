"""
Unit tests for the data loading utilities.
"""
import pytest
import pandas as pd
import os
from qsarify.io import loaders
from qsarify.exceptions import DataImportError

@pytest.fixture
def temp_csv_dir(tmp_path):
    """Create a temporary directory with various CSV files for testing."""
    dir_path = tmp_path / "csv_data"
    dir_path.mkdir()

    # 1. Valid CSV with IDs
    df1 = pd.DataFrame({
        'ID': ['A', 'B', 'C'],
        'Desc1': [1.1, 2.2, 3.3],
        'Desc2': [4.4, 5.5, 6.6],
        'Response': [10, 20, 30]
    })
    df1.to_csv(dir_path / "valid_with_ids.csv", index=False)

    # 2. Valid CSV without IDs
    df2 = pd.DataFrame({
        'Desc1': [1.1, 2.2, 3.3],
        'Desc2': [4.4, 5.5, 6.6],
        'Response': [10, 20, 30]
    })
    df2.to_csv(dir_path / "valid_no_ids.csv", index=False)

    # 3. Invalid CSV - Non-numeric response
    df3 = pd.DataFrame({
        'Desc1': [1, 2, 3],
        'Response': ['a', 'b', 'c']
    })
    df3.to_csv(dir_path / "invalid_response.csv", index=False)

    # 4. Invalid CSV - Non-numeric descriptor
    df4 = pd.DataFrame({
        'Desc1': ["foo", "bar", "baz"],
        'Response': [10, 20, 30]
    })
    df4.to_csv(dir_path / "invalid_descriptor.csv", index=False)

    # 5. Invalid CSV - Too few columns
    df5 = pd.DataFrame({'Response': [10, 20, 30]})
    df5.to_csv(dir_path / "too_few_columns.csv", index=False)

    # 6. Empty file
    (dir_path / "empty.csv").touch()

    return dir_path

def test_load_csv_valid_with_ids(temp_csv_dir):
    """Test loading a valid CSV file with an ID column."""
    file_path = temp_csv_dir / "valid_with_ids.csv"
    X_df, y_series, ids_series = loaders.load_csv_dataset(str(file_path))

    assert isinstance(X_df, pd.DataFrame)
    assert isinstance(y_series, pd.Series)
    assert isinstance(ids_series, pd.Series)
    assert X_df.shape == (3, 2)
    assert y_series.shape == (3,)
    assert ids_series.shape == (3,)
    assert list(ids_series) == ['A', 'B', 'C']
    assert y_series.name == 'Response'

def test_load_csv_valid_no_ids(temp_csv_dir):
    """Test loading a valid CSV file without an ID column."""
    file_path = temp_csv_dir / "valid_no_ids.csv"
    X_df, y_series, ids_series = loaders.load_csv_dataset(str(file_path))

    assert isinstance(X_df, pd.DataFrame)
    assert isinstance(y_series, pd.Series)
    assert ids_series is None
    assert X_df.shape == (3, 2)
    assert y_series.shape == (3,)

def test_load_csv_file_not_found():
    """Test loading a non-existent file."""
    with pytest.raises(DataImportError, match="not found"):
        loaders.load_csv_dataset("non_existent_file.csv")

def test_load_csv_empty_file(temp_csv_dir):
    """Test loading an empty CSV file."""
    file_path = temp_csv_dir / "empty.csv"
    with pytest.raises(DataImportError, match="empty"):
        loaders.load_csv_dataset(str(file_path))

def test_load_csv_too_few_columns(temp_csv_dir):
    """Test loading a CSV with insufficient columns."""
    file_path = temp_csv_dir / "too_few_columns.csv"
    with pytest.raises(DataImportError, match="at least two columns"):
        loaders.load_csv_dataset(str(file_path))

def test_load_csv_invalid_response(temp_csv_dir):
    """Test loading a CSV with a non-numeric response column."""
    file_path = temp_csv_dir / "invalid_response.csv"
    with pytest.raises(DataImportError, match="must be numeric"):
        loaders.load_csv_dataset(str(file_path))

def test_load_csv_invalid_descriptor(temp_csv_dir):
    """Test loading a CSV with a non-numeric descriptor column."""
    file_path = temp_csv_dir / "invalid_descriptor.csv"
    loaders.load_csv_dataset(str(file_path))

def test_load_csv_real_data():
    """Test loading the real-world 1-451_CL_fixed.csv dataset."""
    file_path = "/Users/stephen.szwiec/current_work/qsarify_work/qsarify/tests/1-451_CL_fixed.csv"
    X_df, y_series, ids_series = loaders.load_csv_dataset(
        file_path,
        description_column_name='SMILES',
        response_column_name='Experimental data'
    )

    assert isinstance(X_df, pd.DataFrame)
    assert isinstance(y_series, pd.Series)
    assert isinstance(ids_series, pd.Series)

    # Based on inspection of the file: 74 rows, 2807 total columns - 1 (SMILES) - 1 (Experimental data) = 2805 descriptors
    assert X_df.shape == (74, 2805)
    assert y_series.shape == (74,)
    assert ids_series.shape == (74,)

    assert y_series.name == 'Experimental data'
    assert ids_series.name == 'SMILES'

    # Check a few data types
    assert pd.api.types.is_numeric_dtype(y_series)
    for col in X_df.columns:
        assert pd.api.types.is_numeric_dtype(X_df[col])
