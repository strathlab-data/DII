"""
DII Calculator - Dietary Inflammatory Index for Python
======================================================

A Python implementation of the Dietary Inflammatory Index (DII) calculator
for nutritional epidemiology research.

The Dietary Inflammatory Index is a literature-derived, population-based
index that characterizes the inflammatory potential of an individual's diet.

Basic usage:

    >>> import pandas as pd
    >>> from dii import calculate_dii
    >>>
    >>> # Load your nutrient intake data
    >>> nutrients = pd.read_csv("my_nutrient_data.csv")
    >>>
    >>> # Calculate DII scores
    >>> scores = calculate_dii(nutrients)

For detailed output including per-nutrient contributions:

    >>> detailed = calculate_dii(nutrients, detailed=True)

References
----------
- Shivappa N, Steck SE, Hurley TG, Hussey JR, Hébert JR. Designing and
  developing a literature-derived, population-based dietary inflammatory
  index. Public Health Nutr. 2014;17(8):1689-1696.

- This implementation was inspired by and validated against the dietaryindex
  R package by Jiada (James) Zhan: https://github.com/jamesjiadazhan/dietaryindex

Authors
-------
- Ted Clark (tedclark94@gmail.com)
- Larissa Strath, PhD (larissastrath@ufl.edu)

University of Florida, Department of Health Outcomes and Biomedical Informatics
"""

__version__ = "1.0.5"
__author__ = "Ted Clark, Larissa Strath"
__email__ = "tedclark94@gmail.com"

# Core calculation functions
from .calculator import calculate_dii, calculate_dii_detailed

# Reference table utilities
from .reference import load_reference_table, get_available_nutrients

# Data I/O utilities
from .reader import load_nutrient_data, summarize_input_data, validate_input_file
from .viewer import display_results, display_nutrients_table

# Visualization
from .visualization import (
    plot_dii_distribution,
    plot_nutrient_contributions,
    plot_dii_categories_pie,
)

__all__ = [
    # Core
    "calculate_dii",
    "calculate_dii_detailed",
    # Reference
    "load_reference_table",
    "get_available_nutrients",
    # Reader
    "load_nutrient_data",
    "summarize_input_data",
    "validate_input_file",
    # Viewer
    "display_results",
    "display_nutrients_table",
    # Visualization
    "plot_dii_distribution",
    "plot_nutrient_contributions", 
    "plot_dii_categories_pie",
    # Metadata
    "__version__",
]

