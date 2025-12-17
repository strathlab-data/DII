"""
Output viewer module for the DII Calculator.

This module handles formatting and displaying DII calculation results.
"""

from pathlib import Path
from typing import Optional, Union

import pandas as pd

from .reference import load_reference_table, get_available_nutrients


def display_results(
    results: pd.DataFrame,
    output_file: Optional[Union[str, Path]] = None,
    detailed: bool = False,
    quiet: bool = False,
) -> None:
    """
    Display or save DII calculation results.
    
    Parameters
    ----------
    results : pd.DataFrame
        DataFrame containing DII scores and optionally detailed calculations.
    output_file : str or Path, optional
        If provided, save results to this CSV file. Otherwise, print to console.
    detailed : bool, default False
        Whether the results include detailed breakdown.
    quiet : bool, default False
        If True, suppress informational messages.
    """
    if output_file is not None:
        # Save to file
        output_path = Path(output_file)
        results.to_csv(output_path, index=False)
        if not quiet:
            print(f"Results saved to: {output_path}")
            print(f"  - {len(results)} rows")
            if "DII_score" in results.columns:
                scores = results["DII_score"]
                print(f"  - DII range: {scores.min():.2f} to {scores.max():.2f}")
    else:
        # Print to console
        _print_summary(results, detailed)


def _print_summary(results: pd.DataFrame, detailed: bool = False) -> None:
    """Print a formatted summary of DII results to the console."""
    print("\n" + "=" * 60)
    print("DII CALCULATION RESULTS")
    print("=" * 60)
    
    if "DII_score" not in results.columns:
        print("No DII scores found in results.")
        return
    
    scores = results["DII_score"]
    n = len(scores)
    
    print(f"\nParticipants: {n}")
    print(f"\nDII Score Summary:")
    print(f"  Mean:   {scores.mean():>8.3f}")
    print(f"  Std:    {scores.std():>8.3f}")
    print(f"  Min:    {scores.min():>8.3f}")
    print(f"  Max:    {scores.max():>8.3f}")
    print(f"  Median: {scores.median():>8.3f}")
    
    # Interpretation breakdown
    print(f"\nScore Distribution:")
    anti_inflammatory = (scores < -1).sum()
    neutral = ((scores >= -1) & (scores <= 1)).sum()
    pro_inflammatory = (scores > 1).sum()
    
    print(f"  Anti-inflammatory (< -1):  {anti_inflammatory:>4} ({anti_inflammatory/n*100:>5.1f}%)")
    print(f"  Neutral (-1 to 1):         {neutral:>4} ({neutral/n*100:>5.1f}%)")
    print(f"  Pro-inflammatory (> 1):    {pro_inflammatory:>4} ({pro_inflammatory/n*100:>5.1f}%)")
    
    if detailed and n <= 20:
        print(f"\nIndividual Scores:")
        for i, score in enumerate(scores):
            interpretation = _interpret_score(score)
            print(f"  Row {i+1:>3}: {score:>8.3f}  ({interpretation})")
    
    print("\n" + "=" * 60)


def _interpret_score(score: float) -> str:
    """Return interpretation string for a DII score."""
    if pd.isna(score):
        return "N/A"
    elif score < -4:
        return "strongly anti-inflammatory"
    elif score < -1:
        return "anti-inflammatory"
    elif score <= 1:
        return "neutral"
    elif score <= 4:
        return "pro-inflammatory"
    else:
        return "strongly pro-inflammatory"


def display_nutrients_table() -> None:
    """Display a formatted table of all supported DII nutrients."""
    print("\n" + "=" * 70)
    print("SUPPORTED DII NUTRIENTS")
    print("=" * 70)
    
    ref = load_reference_table()
    
    print(f"\n{'Nutrient':<20} {'Weight':>10} {'Global Mean':>14} {'Global SD':>12}")
    print("-" * 70)
    
    for _, row in ref.iterrows():
        print(
            f"{row['nutrient']:<20} "
            f"{row['weight']:>10.3f} "
            f"{row['global_mean']:>14.3f} "
            f"{row['global_sd']:>12.3f}"
        )
    
    print("-" * 70)
    print(f"Total nutrients: {len(ref)}")
    print("\nUnits: See README.md for detailed unit requirements.")
    print("Reference: Shivappa et al. (2014) Public Health Nutrition")
    print("=" * 70 + "\n")


def format_detailed_output(
    results: pd.DataFrame,
    include_intermediates: bool = True,
) -> pd.DataFrame:
    """
    Format detailed DII output for export.
    
    Parameters
    ----------
    results : pd.DataFrame
        Raw detailed results from calculate_dii_detailed.
    include_intermediates : bool, default True
        Whether to include z-scores and centered percentiles.
    
    Returns
    -------
    pd.DataFrame
        Formatted DataFrame ready for export.
    """
    if not include_intermediates:
        # Keep only final DII columns
        dii_cols = [col for col in results.columns if col.endswith("_dii") or col == "DII_score"]
        return results[dii_cols].copy()
    
    return results.copy()

