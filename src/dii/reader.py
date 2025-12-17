"""
Data reader module for the DII Calculator.

This module handles loading and validating nutrient intake data from CSV files.
"""

from pathlib import Path
from typing import Union

import pandas as pd

from .reference import get_available_nutrients


def validate_input_file(filepath: Union[str, Path]) -> None:
    """
    Validate that an input file exists and is readable.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the input file.
    
    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file is not a CSV file.
    """
    path = Path(filepath)
    
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {filepath}")
    
    if path.suffix.lower() != ".csv":
        raise ValueError(f"Input file must be a CSV file, got: {path.suffix}")


def load_nutrient_data(
    filepath: Union[str, Path],
    encoding: str = "utf-8",
) -> pd.DataFrame:
    """
    Load nutrient intake data from a CSV file.
    
    The CSV file should contain columns for nutrient intakes. Column names
    should match the DII nutrient names (case-insensitive, whitespace-tolerant).
    
    Parameters
    ----------
    filepath : str or Path
        Path to the CSV file containing nutrient data.
    encoding : str, default "utf-8"
        File encoding.
    
    Returns
    -------
    pd.DataFrame
        DataFrame containing the nutrient intake data.
    
    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    pd.errors.EmptyDataError
        If the file is empty.
    
    Examples
    --------
    >>> df = load_nutrient_data("nutrient_intake.csv")
    >>> df.head()
    """
    path = Path(filepath)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    # Load with flexible encoding handling
    try:
        df = pd.read_csv(path, encoding=encoding)
    except UnicodeDecodeError:
        # Try alternative encodings
        for alt_encoding in ["latin-1", "cp1252", "iso-8859-1"]:
            try:
                df = pd.read_csv(path, encoding=alt_encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            raise
    
    return df


def summarize_input_data(df: pd.DataFrame) -> dict:
    """
    Generate a summary of the input nutrient data.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing nutrient intake data.
    
    Returns
    -------
    dict
        Summary statistics including:
        - n_rows: Number of participants/rows
        - n_columns: Number of columns
        - nutrients_found: List of recognized DII nutrients
        - nutrients_missing: List of DII nutrients not in data
        - coverage: Percentage of DII nutrients present
    """
    available = set(get_available_nutrients())
    
    # Normalize column names for matching
    df_columns = set(col.strip() for col in df.columns)
    
    found = available.intersection(df_columns)
    missing = available.difference(df_columns)
    
    return {
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "nutrients_found": sorted(found),
        "nutrients_missing": sorted(missing),
        "coverage": len(found) / len(available) * 100 if available else 0,
    }

