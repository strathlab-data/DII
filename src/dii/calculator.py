"""
DII Calculator - Core Computation Module
=========================================

This module contains the main functions for calculating the Dietary Inflammatory
Index (DII) from nutrient intake data.

The DII calculation follows these steps:
1. For each nutrient, compute a z-score: (intake - global_mean) / global_sd
2. Convert z-score to a centered percentile: 2 * CDF(z) - 1 (range: -1 to +1)
3. Multiply by the inflammatory weight for that nutrient
4. Sum all nutrient contributions to get the total DII score

Interpretation:
- Negative scores indicate an anti-inflammatory diet
- Positive scores indicate a pro-inflammatory diet
- Scores typically range from about -8 to +8

References
----------
Shivappa N, Steck SE, Hurley TG, Hussey JR, Hébert JR. Designing and
developing a literature-derived, population-based dietary inflammatory
index. Public Health Nutr. 2014;17(8):1689-1696.
"""

from __future__ import annotations

from typing import List, Optional, Union

import numpy as np
import pandas as pd
from scipy.stats import norm

from .reference import (
    MEAN_COL,
    NUTRIENT_COL,
    SD_COL,
    WEIGHT_COL,
    load_reference_table,
    validate_nutrient_columns,
)


def calculate_dii(
    nutrient_data: pd.DataFrame,
    reference_df: Optional[pd.DataFrame] = None,
    id_column: Optional[str] = None,
    detailed: bool = False,
) -> pd.DataFrame:
    """
    Calculate the Dietary Inflammatory Index (DII) for each participant.

    This function computes DII scores using the standardized methodology:
    z-score → centered percentile → weighted sum across all available nutrients.

    Parameters
    ----------
    nutrient_data : pd.DataFrame
        DataFrame containing nutrient intake values. Columns should be named
        to match the DII reference nutrients (e.g., "Fiber", "Alcohol", "Vitamin C").
        Can include non-nutrient columns (e.g., participant IDs) which will be ignored.

    reference_df : pd.DataFrame, optional
        Custom reference table with columns: nutrient, weight, global_mean, global_sd.
        If not provided, uses the bundled default reference table.

    id_column : str, optional
        Name of the column containing participant/row identifiers. If provided,
        this column will be included in the output. Common values: "SEQN", "ID".

    detailed : bool, default False
        If True, returns detailed output including per-nutrient z-scores,
        centered percentiles, and DII contributions. If False, returns only
        the final DII score.

    Returns
    -------
    pd.DataFrame
        DataFrame with DII scores. If detailed=False, contains columns:
        - [id_column] (if provided): Participant identifier
        - DII_score: The calculated Dietary Inflammatory Index

        If detailed=True, also includes for each matched nutrient:
        - {nutrient}: Original intake value
        - {nutrient}_zscore: Standardized z-score
        - {nutrient}_percentile: Centered percentile (-1 to +1)
        - {nutrient}_contribution: Weighted contribution to DII

    Raises
    ------
    TypeError
        If nutrient_data is not a pandas DataFrame.
    ValueError
        If no DII nutrients are found in the input data columns.

    Notes
    -----
    - Missing nutrient values (NaN) are excluded from the calculation
    - The function uses vectorized operations for efficient computation
    - Nutrients not present in the input data are simply excluded from the sum

    Examples
    --------
    >>> import pandas as pd
    >>> from dii import calculate_dii

    >>> # Simple usage
    >>> nutrients = pd.DataFrame({
    ...     "SEQN": [1, 2, 3],
    ...     "Fiber": [18.8, 25.0, 12.0],
    ...     "Alcohol": [13.98, 0.0, 30.0],
    ... })
    >>> scores = calculate_dii(nutrients, id_column="SEQN")
    >>> print(scores)
       SEQN  DII_score
    0     1  -0.000000
    1     2  -0.632145
    2     3   0.284523

    >>> # Detailed output
    >>> detailed = calculate_dii(nutrients, id_column="SEQN", detailed=True)
    >>> print(detailed.columns.tolist())
    ['SEQN', 'Fiber', 'Fiber_zscore', 'Fiber_percentile', 'Fiber_contribution', ...]
    """
    # === INPUT VALIDATION ===
    # Type check: must be a DataFrame
    if not isinstance(nutrient_data, pd.DataFrame):
        raise TypeError(
            f"nutrient_data must be a pandas DataFrame, got {type(nutrient_data).__name__}. "
            "Example: calculate_dii(pd.DataFrame({'Fiber': [18.8], 'Alcohol': [13.98]}))"
        )
    
    if reference_df is None:
        reference_df = load_reference_table()

    # Clean column names (strip whitespace) to handle common data issues
    nutrient_data = nutrient_data.copy()
    nutrient_data.columns = nutrient_data.columns.str.strip()

    # Validate and find matching nutrients
    matched_nutrients, _ = validate_nutrient_columns(
        nutrient_data.columns.tolist(), reference_df, verbose=True
    )

    if not matched_nutrients:
        raise ValueError(
            "No DII nutrients found in input data. "
            "Ensure column names match the reference nutrients "
            "(e.g., 'Fiber', 'Alcohol', 'Vitamin C'). "
            "Use get_available_nutrients() to see the full list."
        )
    
    # === NUMERIC VALIDATION ===
    # Ensure nutrient columns contain numeric data
    non_numeric_cols = []
    for col in matched_nutrients:
        if col in nutrient_data.columns:
            if not pd.api.types.is_numeric_dtype(nutrient_data[col]):
                # Try to convert to numeric
                try:
                    nutrient_data[col] = pd.to_numeric(nutrient_data[col], errors='coerce')
                except Exception:
                    non_numeric_cols.append(col)
    
    if non_numeric_cols:
        import warnings
        warnings.warn(
            f"Non-numeric data found in columns {non_numeric_cols}. "
            "Values were coerced to numeric (non-convertible values become NaN).",
            UserWarning
        )
    
    # === COVERAGE WARNING ===
    # Warn if nutrient coverage is low (< 25%)
    total_nutrients = len(reference_df)
    coverage_pct = len(matched_nutrients) / total_nutrients * 100
    if coverage_pct < 25:
        import warnings
        warnings.warn(
            f"Low nutrient coverage: only {len(matched_nutrients)}/{total_nutrients} "
            f"DII nutrients found ({coverage_pct:.1f}%). "
            "DII scores may be less reliable with limited nutrients. "
            "Consider adding more nutrient columns if available.",
            UserWarning
        )

    # Filter reference to matched nutrients only
    ref_matched = reference_df[reference_df[NUTRIENT_COL].isin(matched_nutrients)].copy()

    if detailed:
        return calculate_dii_detailed(nutrient_data, ref_matched, id_column)
    else:
        return _calculate_dii_simple(nutrient_data, ref_matched, id_column)


def _calculate_dii_simple(
    nutrient_data: pd.DataFrame,
    reference_df: pd.DataFrame,
    id_column: Optional[str] = None,
) -> pd.DataFrame:
    """
    Calculate DII scores efficiently using vectorized operations (simple output).

    This internal function computes only the final DII score without
    intermediate values.
    """
    # Initialize result DataFrame
    if id_column and id_column in nutrient_data.columns:
        result = nutrient_data[[id_column]].copy()
    else:
        result = pd.DataFrame(index=nutrient_data.index)

    # Vectorized DII calculation
    total_scores = np.zeros(len(nutrient_data))

    for _, ref_row in reference_df.iterrows():
        nutrient = ref_row[NUTRIENT_COL]
        mean = ref_row[MEAN_COL]
        sd = ref_row[SD_COL]
        weight = ref_row[WEIGHT_COL]

        if nutrient not in nutrient_data.columns:
            continue

        # Get nutrient values
        values = nutrient_data[nutrient].values.astype(float)

        # Compute z-score
        z_scores = (values - mean) / sd

        # Convert to centered percentile: 2 * Phi(z) - 1
        percentiles = 2 * norm.cdf(z_scores) - 1

        # Compute contribution (weight * percentile)
        contributions = percentiles * weight

        # Add to total (handling NaN)
        total_scores = np.nansum([total_scores, contributions], axis=0)

    result["DII_score"] = total_scores
    return result


def calculate_dii_detailed(
    nutrient_data: pd.DataFrame,
    reference_df: Optional[pd.DataFrame] = None,
    id_column: Optional[str] = None,
) -> pd.DataFrame:
    """
    Calculate DII with detailed per-nutrient breakdown.

    This function provides full transparency into the DII calculation by
    returning all intermediate values for each nutrient.

    Parameters
    ----------
    nutrient_data : pd.DataFrame
        DataFrame containing nutrient intake values.
    reference_df : pd.DataFrame, optional
        Reference table. If not provided, loads the default table.
    id_column : str, optional
        Name of the identifier column to include in output.

    Returns
    -------
    pd.DataFrame
        Detailed output with columns for each nutrient:
        - {nutrient}: Original intake value
        - {nutrient}_zscore: Standardized z-score
        - {nutrient}_percentile: Centered percentile (-1 to +1)
        - {nutrient}_contribution: Weighted DII contribution
        - DII_score: Total DII score (sum of all contributions)

    Examples
    --------
    >>> detailed = calculate_dii_detailed(nutrients, id_column="SEQN")
    >>> # View contribution of each nutrient
    >>> contrib_cols = [c for c in detailed.columns if c.endswith("_contribution")]
    >>> detailed[contrib_cols].describe()
    """
    if reference_df is None:
        reference_df = load_reference_table()
        matched_nutrients, _ = validate_nutrient_columns(
            nutrient_data.columns.tolist(), reference_df, verbose=False
        )
        reference_df = reference_df[reference_df[NUTRIENT_COL].isin(matched_nutrients)]

    # Initialize result with ID column if provided
    result_data = {}
    if id_column and id_column in nutrient_data.columns:
        result_data[id_column] = nutrient_data[id_column].values

    # Pre-allocate arrays for all calculations
    n_rows = len(nutrient_data)
    total_scores = np.zeros(n_rows)

    # Calculate for each nutrient
    for _, ref_row in reference_df.iterrows():
        nutrient = ref_row[NUTRIENT_COL]
        mean = ref_row[MEAN_COL]
        sd = ref_row[SD_COL]
        weight = ref_row[WEIGHT_COL]

        if nutrient not in nutrient_data.columns:
            continue

        # Get values and compute all metrics
        values = nutrient_data[nutrient].values.astype(float)
        z_scores = (values - mean) / sd
        percentiles = 2 * norm.cdf(z_scores) - 1
        contributions = percentiles * weight

        # Store in result dictionary
        result_data[nutrient] = values
        result_data[f"{nutrient}_zscore"] = z_scores
        result_data[f"{nutrient}_percentile"] = percentiles
        result_data[f"{nutrient}_contribution"] = contributions

        # Accumulate total (handling NaN)
        total_scores = np.nansum([total_scores, contributions], axis=0)

    result_data["DII_score"] = total_scores

    return pd.DataFrame(result_data)


def compute_zscore(value: float, mean: float, sd: float) -> float:
    """
    Compute z-score for a single value.

    Parameters
    ----------
    value : float
        The observed nutrient intake value.
    mean : float
        Global mean from the reference table.
    sd : float
        Global standard deviation from the reference table.

    Returns
    -------
    float
        The z-score: (value - mean) / sd
    """
    return (value - mean) / sd


def compute_centered_percentile(z_score: float) -> float:
    """
    Convert z-score to centered percentile.

    The centered percentile transformation maps z-scores to the range [-1, +1],
    where 0 represents the global mean and ±1 represent extreme values.

    Parameters
    ----------
    z_score : float
        The standardized z-score.

    Returns
    -------
    float
        Centered percentile in range [-1, +1], calculated as 2*Phi(z) - 1
        where Phi is the standard normal CDF.

    Notes
    -----
    This transformation ensures that:
    - A z-score of 0 → percentile of 0 (at the mean)
    - Positive z-scores → positive percentiles (above mean)
    - Negative z-scores → negative percentiles (below mean)
    """
    return 2 * norm.cdf(z_score) - 1

