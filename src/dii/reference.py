"""
DII Reference Table Management
==============================

This module handles loading and validation of the DII reference table containing
the 45 food parameters with their inflammatory weights, global means, and
standard deviations.

The reference values are derived from:
    Shivappa N, Steck SE, Hurley TG, Hussey JR, Hébert JR. Designing and
    developing a literature-derived, population-based dietary inflammatory
    index. Public Health Nutr. 2014;17(8):1689-1696.
"""

from __future__ import annotations

import importlib.resources
from pathlib import Path
from typing import List, Optional

import pandas as pd


# Column names in the reference table
NUTRIENT_COL = "nutrient"
WEIGHT_COL = "weight"
MEAN_COL = "global_mean"
SD_COL = "global_sd"

# Required columns for validation
REQUIRED_COLUMNS = [NUTRIENT_COL, WEIGHT_COL, MEAN_COL, SD_COL]


def load_reference_table(custom_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load the DII reference table containing inflammatory weights and global statistics.

    The reference table contains 45 food parameters with their:
    - Inflammatory weight (positive = pro-inflammatory, negative = anti-inflammatory)
    - Global mean intake from the world literature
    - Global standard deviation

    Parameters
    ----------
    custom_path : str, optional
        Path to a custom reference table CSV file. If not provided, uses the
        bundled reference data from the package.

    Returns
    -------
    pd.DataFrame
        Reference table with columns: nutrient, weight, global_mean, global_sd

    Raises
    ------
    FileNotFoundError
        If the specified custom_path does not exist.
    ValueError
        If the reference table is missing required columns.

    Examples
    --------
    >>> ref = load_reference_table()
    >>> ref.head()
           nutrient  weight  global_mean  global_sd
    0       Alcohol  -0.278        13.98       3.72
    1   vitamin B12   0.106         5.15       2.70
    2    vitamin B6  -0.365         1.47       0.74

    >>> # Use a custom reference table
    >>> ref = load_reference_table("my_custom_reference.csv")
    """
    if custom_path is not None:
        path = Path(custom_path)
        if not path.exists():
            raise FileNotFoundError(f"Reference table not found: {custom_path}")
        ref_df = pd.read_csv(path)
    else:
        # Load bundled reference data using importlib.resources
        try:
            # Python 3.9+
            ref_path = importlib.resources.files("dii.data").joinpath("dii_reference.csv")
            ref_df = pd.read_csv(ref_path)
        except (AttributeError, TypeError):
            # Python 3.8 fallback
            import pkg_resources
            ref_path = pkg_resources.resource_filename("dii", "data/dii_reference.csv")
            ref_df = pd.read_csv(ref_path)

    # Validate required columns
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in ref_df.columns]
    if missing_cols:
        raise ValueError(
            f"Reference table missing required columns: {missing_cols}. "
            f"Expected columns: {REQUIRED_COLUMNS}"
        )

    # Clean nutrient names (strip whitespace)
    ref_df[NUTRIENT_COL] = ref_df[NUTRIENT_COL].str.strip()

    return ref_df


def get_available_nutrients(reference_df: Optional[pd.DataFrame] = None) -> List[str]:
    """
    Get the list of all available nutrients in the DII reference table.

    Parameters
    ----------
    reference_df : pd.DataFrame, optional
        A pre-loaded reference table. If not provided, loads the default table.

    Returns
    -------
    List[str]
        Sorted list of nutrient names that can be used for DII calculation.

    Examples
    --------
    >>> nutrients = get_available_nutrients()
    >>> len(nutrients)
    45
    >>> "Fiber" in nutrients
    True
    >>> "Vitamin C" in nutrients
    True
    """
    if reference_df is None:
        reference_df = load_reference_table()

    return sorted(reference_df[NUTRIENT_COL].tolist())


def validate_nutrient_columns(
    data_columns: List[str],
    reference_df: Optional[pd.DataFrame] = None,
    verbose: bool = True,
) -> tuple[List[str], List[str]]:
    """
    Validate which nutrients from the input data are available in the reference table.

    Parameters
    ----------
    data_columns : List[str]
        Column names from the input nutrient data.
    reference_df : pd.DataFrame, optional
        A pre-loaded reference table. If not provided, loads the default table.
    verbose : bool, default True
        If True, prints information about matched and missing nutrients.

    Returns
    -------
    tuple[List[str], List[str]]
        A tuple of (matched_nutrients, missing_nutrients):
        - matched_nutrients: Nutrients found in both input data and reference table
        - missing_nutrients: Reference nutrients not found in input data

    Examples
    --------
    >>> matched, missing = validate_nutrient_columns(["Fiber", "Alcohol", "SEQN"])
    Found 2 of 45 DII nutrients in input data
    Missing nutrients: Anthocyanidins, Beta-carotene, ...
    """
    if reference_df is None:
        reference_df = load_reference_table()

    available = set(reference_df[NUTRIENT_COL].tolist())
    input_cols = set(data_columns)

    matched = sorted(available & input_cols)
    missing = sorted(available - input_cols)

    if verbose:
        print(f"Found {len(matched)} of {len(available)} DII nutrients in input data")
        if missing:
            # Show first 5 missing nutrients
            missing_preview = ", ".join(missing[:5])
            if len(missing) > 5:
                missing_preview += f", ... (+{len(missing) - 5} more)"
            print(f"Missing nutrients: {missing_preview}")

    return matched, missing

