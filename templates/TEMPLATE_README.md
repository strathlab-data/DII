# DII Input Data Template

This folder contains templates to help you prepare your nutrient intake data for DII calculation.

## Quick Start

1. Copy `input_template.csv` to your working directory
2. Fill in your nutrient intake values (see units below)
3. Run: `python -m dii your_data.csv -o results.csv`

## Required Units

| Nutrient | Unit | Notes |
|----------|------|-------|
| Alcohol | g/day | Ethanol equivalent |
| vitamin B12 | µg/day | |
| vitamin B6 | mg/day | |
| Beta-carotene | µg/day | |
| Caffeine | g/day | **Note: grams, not mg** |
| Carbohydrate | g/day | |
| Cholesterol | mg/day | |
| Energy | kcal/day | |
| Total fat | g/day | |
| Fiber | g/day | |
| Folic acid | µg/day | |
| Iron | mg/day | |
| Magnesium | mg/day | |
| MUFA | g/day | Monounsaturated fatty acids |
| Niacin | mg/day | |
| n-3 fatty acid | g/day | Omega-3 |
| n-6 fatty acid | g/day | Omega-6 |
| Protein | g/day | |
| PUFA | g/day | Polyunsaturated fatty acids |
| Riboflavin | mg/day | Vitamin B2 |
| Saturated fat | g/day | |
| Selenium | µg/day | |
| Thiamin | mg/day | Vitamin B1 |
| Trans fat | g/day | |
| Vitamin A | RE/day | Retinol equivalents |
| Vitamin C | mg/day | |
| Vitamin D | µg/day | |
| Vitamin E | mg/day | |
| Zinc | mg/day | |

## Optional Nutrients (if available)

These nutrients are part of the full 45-parameter DII but are less commonly available:

| Nutrient | Unit |
|----------|------|
| Eugenol | mg/day |
| Garlic | g/day |
| Ginger | g/day |
| Onion | g/day |
| Saffron | g/day |
| Turmeric | mg/day |
| Green/black tea | g/day |
| Flavan-3-ol | mg/day |
| Flavones | mg/day |
| Flavonols | mg/day |
| Flavonones | mg/day |
| Anthocyanidins | mg/day |
| Isoflavones | mg/day |
| Pepper | g/day |
| Thyme/oregano | mg/day |
| Rosemary | mg/day |

## Missing Data

- Leave cells blank or use empty values for missing nutrients
- The DII calculator will use only available nutrients
- More nutrients = more accurate DII score
- Minimum recommended: Energy + macronutrients + key vitamins

## Example

```csv
participant_id,Energy,Carbohydrate,Protein,Total fat,Fiber,Vitamin C
P001,2100,280,85,70,22,95
P002,1850,240,72,65,18,78
P003,2400,310,95,82,28,120
```

## Citation

If you use this tool in your research, please cite:

```
Clark, T., & Strath, L. (2025). DII Calculator: A Python Implementation 
of the Dietary Inflammatory Index. GitHub. 
https://github.com/strathlab-data/DII
```

And the original DII methodology:

```
Shivappa, N., Steck, S. E., Hurley, T. G., Hussey, J. R., & Hébert, J. R. (2014). 
Designing and developing a literature-derived, population-based dietary 
inflammatory index. Public Health Nutrition, 17(8), 1689-1696.
```

