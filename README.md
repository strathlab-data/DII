# DII Calculator

**A Python implementation of the Dietary Inflammatory Index (DII) for nutritional epidemiology research**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

The **Dietary Inflammatory Index (DII)** is a literature-derived, population-based index designed to assess the inflammatory potential of an individual's diet. This Python package provides a validated, efficient implementation for calculating DII scores from nutrient intake data.

**Key Features:**
- 🚀 Vectorized calculations for high performance on large datasets
- 📊 Support for all 45 DII food parameters
- 🔧 Graceful handling of missing nutrients
- 📈 Built-in visualization tools
- 💻 Command-line interface for quick analysis
- 📦 Bundled reference data (no external files required)
- ✅ Comprehensive validation against known test cases

### Sample Output

<p align="center">
  <img src="docs/images/dii_distribution.png" alt="DII Score Distribution" width="700">
</p>

*DII score distribution from NHANES 2017-2018 data (n=13,580). Over half of U.S. adults have pro-inflammatory diets.*

### Interpretation

| DII Score | Dietary Pattern |
|-----------|-----------------|
| Negative (e.g., -4 to -1) | Anti-inflammatory diet |
| Near zero | Neutral inflammatory potential |
| Positive (e.g., +1 to +4) | Pro-inflammatory diet |

Scores typically range from approximately **-8** (maximally anti-inflammatory) to **+8** (maximally pro-inflammatory).

---

## Installation

### Prerequisites

- Python 3.8 or higher ([Download Python](https://www.python.org/downloads/))

### Option 1: Quick Install from GitHub

```bash
pip install git+https://github.com/strathlab-data/DII.git
```

### Option 2: Full Development Setup (Recommended for Researchers)

This creates an isolated environment that won't conflict with other Python projects.

**Windows (PowerShell):**
```powershell
# Clone the repository
git clone https://github.com/strathlab-data/DII.git
cd dii-calculator

# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install package and dependencies
pip install -e .

# Verify installation
python -c "from dii import calculate_dii; print('Installation successful!')"
```

**macOS / Linux:**
```bash
# Clone the repository
git clone https://github.com/strathlab-data/DII.git
cd dii-calculator

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install package and dependencies
pip install -e .

# Verify installation
python -c "from dii import calculate_dii; print('Installation successful!')"
```

### Running Tests

```bash
# Make sure your virtual environment is activated, then:
pip install pytest
pytest tests/ -v
```

### Dependencies

Automatically installed:
- pandas ≥ 1.3.0
- numpy ≥ 1.20.0
- scipy ≥ 1.7.0

---

## Quick Start

```python
import pandas as pd
from dii import calculate_dii

# Load your nutrient intake data
nutrients = pd.read_csv("your_nutrient_data.csv")

# Calculate DII scores
results = calculate_dii(nutrients, id_column="participant_id")

print(results.head())
#    participant_id  DII_score
# 0               1      -1.23
# 1               2       0.45
# 2               3       2.87
```

### Detailed Output

For transparency in the calculation, request detailed output:

```python
detailed = calculate_dii(nutrients, id_column="participant_id", detailed=True)

# View contribution of each nutrient
print(detailed[["participant_id", "Fiber_contribution", "Alcohol_contribution", "DII_score"]])
```

---

## Input Data Format

Your input DataFrame should contain columns named to match the DII reference nutrients. The following 45 nutrients are supported:

| Nutrient | Unit | Nutrient | Unit |
|----------|------|----------|------|
| Alcohol | g/day | Magnesium | mg/day |
| Beta-carotene | μg/day | MUFA | g/day |
| Caffeine | g/day | n-3 fatty acid | g/day |
| Carbohydrate | g/day | n-6 fatty acid | g/day |
| Cholesterol | mg/day | Niacin | mg/day |
| Energy | kcal/day | Onion | g/day |
| Eugenol | mg/day | Pepper | g/day |
| Fiber | g/day | Protein | g/day |
| Flavan-3-ol | mg/day | PUFA | g/day |
| Flavones | mg/day | Riboflavin | mg/day |
| Flavonols | mg/day | Rosemary | mg/day |
| Flavonones | mg/day | Saffron | mg/day |
| Folic acid | μg/day | Saturated fat | g/day |
| Garlic | g/day | Selenium | μg/day |
| Ginger | mg/day | Thiamin | mg/day |
| Green/black tea | g/day | Thyme/oregano | mg/day |
| Iron | mg/day | Total fat | g/day |
| Isoflavones | mg/day | Trans fat | g/day |
| Anthocyanidins | mg/day | Turmeric | mg/day |
| vitamin B6 | mg/day | Vitamin A | RE/day |
| vitamin B12 | μg/day | Vitamin C | mg/day |
| Zinc | mg/day | Vitamin D | μg/day |
| | | Vitamin E | mg/day |

**Note:** Column names must match exactly (case-sensitive). Nutrients not present in your data will be excluded from the calculation.

### Example Input CSV

```csv
participant_id,Fiber,Alcohol,Vitamin C,Saturated fat,Energy
1,18.8,13.98,118.2,28.6,2056
2,25.0,0.0,150.0,22.0,1800
3,12.0,30.0,80.0,35.0,2400
```

---

## API Reference

### `calculate_dii()`

```python
calculate_dii(
    nutrient_data: pd.DataFrame,
    reference_df: pd.DataFrame = None,
    id_column: str = None,
    detailed: bool = False
) -> pd.DataFrame
```

**Parameters:**
- `nutrient_data`: DataFrame with nutrient intake values
- `reference_df`: Custom reference table (optional, uses bundled data by default)
- `id_column`: Name of participant identifier column
- `detailed`: If True, returns per-nutrient breakdown

**Returns:** DataFrame with DII scores

### `get_available_nutrients()`

```python
get_available_nutrients() -> List[str]
```

Returns a list of all 45 supported nutrient names.

### `load_reference_table()`

```python
load_reference_table(custom_path: str = None) -> pd.DataFrame
```

Load the DII reference table with inflammatory weights and global statistics.

---

## Visualization

The package includes built-in visualization tools (requires `matplotlib`):

```bash
pip install matplotlib
```

```python
from dii import calculate_dii, calculate_dii_detailed
from dii import plot_dii_distribution, plot_nutrient_contributions

# Calculate DII
results = calculate_dii(nutrients)
detailed = calculate_dii_detailed(nutrients)

# Generate plots
plot_dii_distribution(results, save_path="dii_distribution.png")
plot_nutrient_contributions(detailed, save_path="nutrient_contributions.png")
```

### Nutrient Contributions Analysis

<p align="center">
  <img src="docs/images/nutrient_contributions.png" alt="Nutrient Contributions" width="700">
</p>

*Average nutrient contributions to DII in NHANES data. Green bars indicate anti-inflammatory contributions; red bars indicate pro-inflammatory contributions.*

---

## Command-Line Interface

Run DII calculations directly from the terminal:

```bash
# Calculate DII scores
python -m dii input_data.csv -o results.csv

# Include detailed breakdown
python -m dii input_data.csv -o results.csv --detailed

# List supported nutrients
python -m dii --nutrients

# Show help
python -m dii --help
```

---

## Methodology

The DII calculation follows the standardized approach from Shivappa et al. (2014):

1. **Z-score standardization**: For each nutrient, compute:
   ```
   z = (intake - global_mean) / global_sd
   ```

2. **Centered percentile conversion**: Transform z-score to a value between -1 and +1:
   ```
   percentile = 2 × Φ(z) - 1
   ```
   where Φ is the standard normal CDF.

3. **Weighted summation**: Multiply each percentile by the nutrient's inflammatory weight and sum:
   ```
   DII = Σ (percentile_i × weight_i)
   ```

Positive weights indicate pro-inflammatory effects; negative weights indicate anti-inflammatory effects.

---

## Validation

This implementation has been validated against:
- Synthetic test cases with mathematically verifiable results (SEQN 1, 2, 3)
- The [dietaryindex R package](https://github.com/jamesjiadazhan/dietaryindex) methodology
- Original R implementation from PANDA-1 study statistician
- Independent verification by Jiyan Aslan Ceylan (University of Florida)

**Validation result:** 13,580 NHANES participants tested with perfect match to original calculations.

Run the test suite:

```bash
pytest tests/ -v
```

---

## Limitations

- **Nutrient availability**: DII accuracy depends on having data for as many of the 45 parameters as possible. Studies with limited nutrient data will have less precise scores.
- **Reference population**: Global means and SDs are derived from world literature circa 2014. Dietary patterns may have shifted.
- **Food-level data**: This package calculates DII from nutrient totals, not individual foods. Some DII parameters (e.g., garlic, turmeric) may be underestimated in datasets that don't capture specific foods.
- **Supplement data**: The DII was designed for dietary intake. Including supplement data may affect score interpretation.

For detailed information about data sources, see [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md).

---

## Citation

If you use this package in your research, please cite:

```bibtex
@software{clark2025dii,
  author = {Clark, Ted and Strath, Larissa},
  title = {dii-calculator: Dietary Inflammatory Index Calculator for Python},
  year = {2025},
  url = {https://github.com/strathlab-data/DII}
}
```

### Original DII Methodology

Please also cite the original DII development paper:

> Shivappa N, Steck SE, Hurley TG, Hussey JR, Hébert JR. Designing and developing a literature-derived, population-based dietary inflammatory index. *Public Health Nutr*. 2014;17(8):1689-1696. doi:10.1017/S1368980013002115

### dietaryindex R Package

This implementation was inspired by and validated against the [dietaryindex R package](https://github.com/jamesjiadazhan/dietaryindex). If you use this Python package, please also cite their excellent work:

> Zhan JJ, Hodge RA, Dunlop AL, et al. Dietaryindex: a user-friendly and versatile R package for standardizing dietary pattern analysis in epidemiological and clinical studies. *Am J Clin Nutr*. 2024. doi:10.1016/j.ajcnut.2024.08.021

```bibtex
@article{zhan2024dietaryindex,
  author = {Zhan, Jiada James and Hodge, Rebecca A and Dunlop, Anne L and Lee, Matthew M and Bui, Linh and Liang, Donghai and Ferranti, Erin P},
  title = {Dietaryindex: a user-friendly and versatile R package for standardizing dietary pattern analysis in epidemiological and clinical studies},
  journal = {American Journal of Clinical Nutrition},
  year = {2024},
  doi = {10.1016/j.ajcnut.2024.08.021},
  url = {https://github.com/jamesjiadazhan/dietaryindex}
}
```

---

## Authors & Acknowledgments

**Authors:**
- **Ted Clark** — Data Analyst, University of Florida ([tedclark94@gmail.com](mailto:tedclark94@gmail.com))
- **Larissa Strath, PhD** — Assistant Professor, Department of Health Outcomes and Biomedical Informatics, University of Florida ([larissastrath@ufl.edu](mailto:larissastrath@ufl.edu))

**Affiliation:**  
University of Florida, College of Medicine  
Department of Health Outcomes and Biomedical Informatics

This package was developed as part of the PANDA-1 study at the University of Florida Pain Research & Intervention Center of Excellence.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request
