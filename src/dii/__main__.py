#!/usr/bin/env python3
"""
Command-line interface for the DII Calculator.

Usage:
    python -m dii                           # Show help
    python -m dii input.csv                 # Calculate DII, print to console
    python -m dii input.csv -o output.csv   # Calculate DII, save to file
    python -m dii input.csv --detailed      # Include z-scores and contributions
    python -m dii --nutrients               # List available nutrients
"""

import argparse
import sys
from pathlib import Path

from .calculator import calculate_dii, calculate_dii_detailed
from .reader import load_nutrient_data, validate_input_file
from .viewer import display_results, display_nutrients_table
from . import __version__


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        prog="dii",
        description="Calculate Dietary Inflammatory Index (DII) scores from nutrient intake data.",
        epilog="For more information, visit: https://github.com/strathlab-data/DII",
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    
    parser.add_argument(
        "input_file",
        nargs="?",
        type=Path,
        help="Path to CSV file containing nutrient intake data",
    )
    
    parser.add_argument(
        "-o", "--output",
        type=Path,
        metavar="FILE",
        help="Save results to CSV file (default: print to console)",
    )
    
    parser.add_argument(
        "-d", "--detailed",
        action="store_true",
        help="Include z-scores, centered percentiles, and nutrient contributions",
    )
    
    parser.add_argument(
        "-n", "--nutrients",
        action="store_true",
        help="List all supported nutrients and their units",
    )
    
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress informational messages",
    )
    
    return parser


def main(args: list[str] | None = None) -> int:
    """
    Main entry point for the DII calculator CLI.
    
    Parameters
    ----------
    args : list of str, optional
        Command-line arguments. If None, uses sys.argv.
    
    Returns
    -------
    int
        Exit code (0 for success, non-zero for errors).
    """
    parser = create_parser()
    parsed_args = parser.parse_args(args)
    
    # Show nutrients list
    if parsed_args.nutrients:
        display_nutrients_table()
        return 0
    
    # Require input file for calculations
    if parsed_args.input_file is None:
        parser.print_help()
        return 0
    
    # Validate input file
    try:
        validate_input_file(parsed_args.input_file)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    # Load data
    if not parsed_args.quiet:
        print(f"Loading data from: {parsed_args.input_file}")
    
    try:
        nutrient_data = load_nutrient_data(parsed_args.input_file)
    except Exception as e:
        print(f"Error loading data: {e}", file=sys.stderr)
        return 1
    
    # Calculate DII
    if not parsed_args.quiet:
        print("Calculating DII scores...")
    
    try:
        if parsed_args.detailed:
            results = calculate_dii_detailed(nutrient_data)
        else:
            results = calculate_dii(nutrient_data)
    except Exception as e:
        print(f"Error calculating DII: {e}", file=sys.stderr)
        return 1
    
    # Output results
    display_results(
        results,
        output_file=parsed_args.output,
        detailed=parsed_args.detailed,
        quiet=parsed_args.quiet,
    )
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

