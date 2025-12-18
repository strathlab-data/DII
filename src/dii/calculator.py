"""
DII Calculator - Core Computation Module
=========================================

This module contains the main functions for calculating the Dietary Inflammatory
Index (DII) from nutrient intake data.

Algorithm
---------
The DII calculation follows these steps for each nutrient:

1. **Z-score**: z = (intake - global_mean) / global_sd
2. **Centered percentile**: p = 2 × Φ(z) - 1, where Φ is the standard normal CDF
3. **Contribution**: c = p × inflammatory_weight
4. **Total DII**: sum of all nutrient contributions

Mathematical Properties
-----------------------
- Centered percentile maps z-scores to range [-1, +1]
- At z=0 (intake equals global mean), percentile = 0
- At z=±∞, percentile approaches ±1
- DII range depends on available nutrients; theoretical max ~±8 with all 45 nutrients

Interpretation
--------------
- Negative scores indicate an anti-inflammatory diet
- Positive scores indicate a pro-inflammatory diet
- Scores typically range from about -8 to +8

Precision
---------
- All calculations use numpy.float64 for reproducibility
- Validation tolerance: 1e-10 (matches R implementation)
- Infinity values from division are converted to NaN

References
----------
Shivappa N, Steck SE, Hurley TG, Hussey JR, Hébert JR. Designing and
developing a literature-derived, population-based dietary inflammatory
index. Public Health Nutr. 2014;17(8):1689-1696.
doi:10.1017/S1368980013002115
"""

from __future__ import annotations

import warnings
from typing import List, Optional, Tuple, Union

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

# =============================================================================
# CONSTANTS FOR SCIENTIFIC REPRODUCIBILITY
# =============================================================================

#: NumPy dtype for all floating-point calculations (IEEE 754 double precision)
FLOAT_DTYPE = np.float64

#: Validation tolerance for comparing calculated vs expected DII scores
#: Set to 1e-9 to account for floating-point representation differences
VALIDATION_TOLERANCE = 1e-9

#: Threshold (in standard deviations) for flagging potentially erroneous values
EXTREME_VALUE_THRESHOLD = 10.0

#: Minimum coverage (fraction) below which a warning is issued
MIN_COVERAGE_FRACTION = 0.25


# =============================================================================
# INPUT VALIDATION UTILITIES
# =============================================================================

def _validate_nutrient_bounds(
    values: np.ndarray,
    nutrient: str,
    mean: float,
    sd: float,
    threshold_sd: float = EXTREME_VALUE_THRESHOLD,
) -> np.ndarray:
    """
    Check for biologically implausible nutrient values.

    Values exceeding the threshold (default: 10 SD from global mean) trigger
    a warning but do not fail. This helps catch common unit errors
    (e.g., mg vs g for caffeine).

    Parameters
    ----------
    values : np.ndarray
        Nutrient intake values (dtype: float64).
    nutrient : str
        Name of the nutrient (for warning message).
    mean : float
        Global mean from reference table.
    sd : float
        Global standard deviation from reference table.
    threshold_sd : float, default 10.0
        Number of standard deviations beyond which values are flagged.

    Returns
    -------
    np.ndarray
        The input values unchanged (validation only issues warnings).

    Warns
    -----
    UserWarning
        If any values exceed the threshold.
    """
    # Skip if all NaN
    if np.all(np.isnan(values)):
        return values

    # Calculate how many SD from mean
    deviations = np.abs(values - mean) / sd
    extreme_mask = deviations > threshold_sd

    # Count extreme values (excluding NaN)
    n_extreme = np.sum(extreme_mask & ~np.isnan(values))

    if n_extreme > 0:
        max_deviation = np.nanmax(deviations)
        warnings.warn(
            f"Nutrient '{nutrient}' has {n_extreme} value(s) exceeding {threshold_sd} SD "
            f"from global mean (max: {max_deviation:.1f} SD). "
            f"Check for unit errors (e.g., mg vs g, µg vs mg). "
            f"Global mean: {mean}, SD: {sd}.",
            UserWarning,
            stacklevel=4,
        )

    return values


def _ensure_float64(
    values: np.ndarray,
    handle_inf: bool = True,
) -> np.ndarray:
    """
    Ensure array is float64 and optionally handle infinity values.

    Parameters
    ----------
    values : np.ndarray
        Input array.
    handle_inf : bool, default True
        If True, replace ±inf with NaN.

    Returns
    -------
    np.ndarray
        Array with dtype float64, optionally with inf replaced by NaN.
    """
    result = np.asarray(values, dtype=FLOAT_DTYPE)

    if handle_inf:
        result = np.where(np.isinf(result), np.nan, result)

    return result


# =============================================================================
# CORE DII CALCULATION FUNCTIONS
# =============================================================================

def calculate_dii(
    nutrient_data: pd.DataFrame,
    reference_df: Optional[pd.DataFrame] = None,
    id_column: Optional[str] = None,
    detailed: bool = False,
    validate_bounds: bool = True,
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
        All nutrient values should be numeric (float64 recommended).

    reference_df : pd.DataFrame, optional
        Custom reference table with columns: nutrient, weight, global_mean, global_sd.
        If not provided, uses the bundled default reference table from Shivappa et al. (2014).

    id_column : str, optional
        Name of the column containing participant/row identifiers. If provided,
        this column will be included in the output. Common values: "SEQN", "ID".

    detailed : bool, default False
        If True, returns detailed output including per-nutrient z-scores,
        centered percentiles, and DII contributions. If False, returns only
        the final DII score.

    validate_bounds : bool, default True
        If True, checks for values exceeding 10 SD from the global mean and
        issues warnings. This helps catch unit conversion errors.

    Returns
    -------
    pd.DataFrame
        DataFrame with DII scores (dtype: float64). If detailed=False, contains columns:

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

    Warns
    -----
    UserWarning
        - If nutrient coverage is below 25% (11 of 45 nutrients)
        - If any nutrient has values >10 SD from global mean
        - If non-numeric data is coerced to numeric

    Notes
    -----
    - All calculations use float64 precision for reproducibility
    - Missing nutrient values (NaN) are excluded from the sum
    - Infinity values (from edge cases) are converted to NaN
    - Nutrients not present in the input data are simply excluded
    - Validated against dietaryindex R package with tolerance < 1e-10

    Examples
    --------
    >>> import pandas as pd
    >>> from dii import calculate_dii

    Basic usage:

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

    Detailed output with per-nutrient breakdown:

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
    coerced_cols = []
    for col in matched_nutrients:
        if col in nutrient_data.columns:
            if not pd.api.types.is_numeric_dtype(nutrient_data[col]):
                # Try to convert to numeric
                original_non_null = nutrient_data[col].notna().sum()
                nutrient_data[col] = pd.to_numeric(nutrient_data[col], errors="coerce")
                new_non_null = nutrient_data[col].notna().sum()
                if new_non_null < original_non_null:
                    coerced_cols.append(col)

    if coerced_cols:
        warnings.warn(
            f"Non-numeric data found in columns {coerced_cols}. "
            "Values were coerced to numeric (non-convertible values become NaN).",
            UserWarning,
        )

    # === COVERAGE WARNING ===
    # Warn if nutrient coverage is low (< 25%)
    total_nutrients = len(reference_df)
    coverage_pct = len(matched_nutrients) / total_nutrients * 100
    if coverage_pct < (MIN_COVERAGE_FRACTION * 100):
        warnings.warn(
            f"Low nutrient coverage: only {len(matched_nutrients)}/{total_nutrients} "
            f"DII nutrients found ({coverage_pct:.1f}%). "
            "DII scores may be less reliable with limited nutrients. "
            "Consider adding more nutrient columns if available.",
            UserWarning,
        )

    # Filter reference to matched nutrients only
    ref_matched = reference_df[reference_df[NUTRIENT_COL].isin(matched_nutrients)].copy()

    if detailed:
        return _calculate_dii_detailed(
            nutrient_data, ref_matched, id_column, validate_bounds
        )
    else:
        return _calculate_dii_simple(
            nutrient_data, ref_matched, id_column, validate_bounds
        )


def _calculate_dii_simple(
    nutrient_data: pd.DataFrame,
    reference_df: pd.DataFrame,
    id_column: Optional[str] = None,
    validate_bounds: bool = True,
) -> pd.DataFrame:
    """
    Calculate DII scores efficiently using vectorized operations (simple output).

    This internal function computes only the final DII score without
    intermediate values. All calculations use float64 precision.

    Parameters
    ----------
    nutrient_data : pd.DataFrame
        Input nutrient data.
    reference_df : pd.DataFrame
        Reference table (pre-filtered to matched nutrients).
    id_column : str, optional
        Identifier column name.
    validate_bounds : bool
        Whether to check for extreme values.

    Returns
    -------
    pd.DataFrame
        DataFrame with id_column (if provided) and DII_score.
    """
    # Initialize result DataFrame
    if id_column and id_column in nutrient_data.columns:
        result = nutrient_data[[id_column]].copy()
    else:
        result = pd.DataFrame(index=nutrient_data.index)

    # Vectorized DII calculation with explicit float64
    total_scores = np.zeros(len(nutrient_data), dtype=FLOAT_DTYPE)

    for _, ref_row in reference_df.iterrows():
        nutrient = ref_row[NUTRIENT_COL]
        mean = FLOAT_DTYPE(ref_row[MEAN_COL])
        sd = FLOAT_DTYPE(ref_row[SD_COL])
        weight = FLOAT_DTYPE(ref_row[WEIGHT_COL])

        if nutrient not in nutrient_data.columns:
            continue

        # Get nutrient values as float64
        values = _ensure_float64(nutrient_data[nutrient].values)

        # Validate bounds if requested
        if validate_bounds:
            _validate_nutrient_bounds(values, nutrient, mean, sd)

        # Compute z-score (float64)
        z_scores = (values - mean) / sd
        z_scores = _ensure_float64(z_scores)  # Handle potential inf

        # Convert to centered percentile: 2 * Φ(z) - 1
        percentiles = np.asarray(2 * norm.cdf(z_scores) - 1, dtype=FLOAT_DTYPE)

        # Compute contribution (weight × percentile)
        contributions = percentiles * weight

        # Add to total (handling NaN)
        total_scores = np.nansum(
            np.stack([total_scores, contributions]), axis=0
        ).astype(FLOAT_DTYPE)

    result["DII_score"] = total_scores
    return result


def _calculate_dii_detailed(
    nutrient_data: pd.DataFrame,
    reference_df: pd.DataFrame,
    id_column: Optional[str] = None,
    validate_bounds: bool = True,
) -> pd.DataFrame:
    """
    Calculate DII with detailed per-nutrient breakdown.

    This function provides full transparency into the DII calculation by
    returning all intermediate values for each nutrient.

    Parameters
    ----------
    nutrient_data : pd.DataFrame
        DataFrame containing nutrient intake values.
    reference_df : pd.DataFrame
        Reference table (pre-filtered to matched nutrients).
    id_column : str, optional
        Name of the identifier column to include in output.
    validate_bounds : bool
        Whether to check for extreme values.

    Returns
    -------
    pd.DataFrame
        Detailed output with columns for each nutrient:

        - {nutrient}: Original intake value (float64)
        - {nutrient}_zscore: Standardized z-score (float64)
        - {nutrient}_percentile: Centered percentile, range [-1, +1] (float64)
        - {nutrient}_contribution: Weighted DII contribution (float64)
        - DII_score: Total DII score (sum of all contributions) (float64)

    Examples
    --------
    >>> detailed = calculate_dii(nutrients, id_column="SEQN", detailed=True)
    >>> # View contribution of each nutrient
    >>> contrib_cols = [c for c in detailed.columns if c.endswith("_contribution")]
    >>> detailed[contrib_cols].describe()
    """
    # Initialize result with ID column if provided
    result_data = {}
    if id_column and id_column in nutrient_data.columns:
        result_data[id_column] = nutrient_data[id_column].values

    # Pre-allocate arrays for all calculations (float64)
    n_rows = len(nutrient_data)
    total_scores = np.zeros(n_rows, dtype=FLOAT_DTYPE)

    # Calculate for each nutrient
    for _, ref_row in reference_df.iterrows():
        nutrient = ref_row[NUTRIENT_COL]
        mean = FLOAT_DTYPE(ref_row[MEAN_COL])
        sd = FLOAT_DTYPE(ref_row[SD_COL])
        weight = FLOAT_DTYPE(ref_row[WEIGHT_COL])

        if nutrient not in nutrient_data.columns:
            continue

        # Get values as float64 and validate
        values = _ensure_float64(nutrient_data[nutrient].values)

        if validate_bounds:
            _validate_nutrient_bounds(values, nutrient, mean, sd)

        # Compute all metrics (float64)
        z_scores = _ensure_float64((values - mean) / sd)
        percentiles = np.asarray(2 * norm.cdf(z_scores) - 1, dtype=FLOAT_DTYPE)
        contributions = percentiles * weight

        # Store in result dictionary
        result_data[nutrient] = values
        result_data[f"{nutrient}_zscore"] = z_scores
        result_data[f"{nutrient}_percentile"] = percentiles
        result_data[f"{nutrient}_contribution"] = contributions

        # Accumulate total (handling NaN)
        total_scores = np.nansum(
            np.stack([total_scores, contributions]), axis=0
        ).astype(FLOAT_DTYPE)

    result_data["DII_score"] = total_scores

    return pd.DataFrame(result_data)


# =============================================================================
# LEGACY WRAPPER (for backward compatibility)
# =============================================================================

def calculate_dii_detailed(
    nutrient_data: pd.DataFrame,
    reference_df: Optional[pd.DataFrame] = None,
    id_column: Optional[str] = None,
) -> pd.DataFrame:
    """
    Calculate DII with detailed per-nutrient breakdown.

    .. deprecated::
        Use ``calculate_dii(nutrient_data, detailed=True)`` instead.
        This function is retained for backward compatibility.

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
    """
    if reference_df is None:
        reference_df = load_reference_table()
        matched_nutrients, _ = validate_nutrient_columns(
            nutrient_data.columns.tolist(), reference_df, verbose=False
        )
        reference_df = reference_df[reference_df[NUTRIENT_COL].isin(matched_nutrients)]

    return _calculate_dii_detailed(
        nutrient_data, reference_df, id_column, validate_bounds=True
    )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def compute_zscore(
    value: Union[float, np.ndarray],
    mean: float,
    sd: float,
) -> Union[float, np.ndarray]:
    """
    Compute z-score for a single value or array.

    The z-score represents how many standard deviations a value is from
    the global mean.

    Parameters
    ----------
    value : float or np.ndarray
        The observed nutrient intake value(s).
    mean : float
        Global mean from the reference table.
    sd : float
        Global standard deviation from the reference table.

    Returns
    -------
    float or np.ndarray
        The z-score: (value - mean) / sd

    Examples
    --------
    >>> compute_zscore(25.0, 18.8, 4.9)  # Fiber: 25g, mean=18.8, sd=4.9
    1.2653061224489797

    >>> compute_zscore(np.array([10, 18.8, 30]), 18.8, 4.9)
    array([-1.79591837,  0.        ,  2.28571429])
    """
    result = (value - mean) / sd
    if isinstance(result, np.ndarray):
        return _ensure_float64(result)
    return FLOAT_DTYPE(result) if not np.isinf(result) else np.nan


def compute_centered_percentile(
    z_score: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """
    Convert z-score to centered percentile.

    The centered percentile transformation maps z-scores to the range [-1, +1],
    where 0 represents the global mean and ±1 represent extreme values.

    Parameters
    ----------
    z_score : float or np.ndarray
        The standardized z-score(s).

    Returns
    -------
    float or np.ndarray
        Centered percentile in range [-1, +1], calculated as 2×Φ(z) - 1
        where Φ is the standard normal CDF.

    Notes
    -----
    This transformation ensures that:

    - A z-score of 0 → percentile of 0 (at the mean)
    - Positive z-scores → positive percentiles (above mean)
    - Negative z-scores → negative percentiles (below mean)
    - As z → +∞, percentile → +1
    - As z → -∞, percentile → -1

    Examples
    --------
    >>> compute_centered_percentile(0)  # At the mean
    0.0

    >>> compute_centered_percentile(1.96)  # 97.5th percentile
    0.9500042097035593

    >>> compute_centered_percentile(-1.96)  # 2.5th percentile
    -0.9500042097035593
    """
    result = 2 * norm.cdf(z_score) - 1
    if isinstance(result, np.ndarray):
        return np.asarray(result, dtype=FLOAT_DTYPE)
    return FLOAT_DTYPE(result)


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Core functions
    "calculate_dii",
    "calculate_dii_detailed",
    # Utility functions
    "compute_zscore",
    "compute_centered_percentile",
    # Constants
    "FLOAT_DTYPE",
    "VALIDATION_TOLERANCE",
]
