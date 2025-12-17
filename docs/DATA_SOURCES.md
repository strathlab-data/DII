# Data Sources and Provenance

This document describes the sources of data used in this package.

## DII Reference Table

The DII reference table (`dii/data/dii_reference.csv`) contains the 45 food parameters used to calculate the Dietary Inflammatory Index. Each parameter includes:

- **nutrient**: Name of the food parameter
- **weight**: Inflammatory effect score (negative = anti-inflammatory, positive = pro-inflammatory)
- **global_mean**: Global daily intake mean from world literature
- **global_sd**: Global daily intake standard deviation

### Source

These values are derived from the original DII development paper:

> Shivappa N, Steck SE, Hurley TG, Hussey JR, Hébert JR. Designing and developing a literature-derived, population-based dietary inflammatory index. *Public Health Nutr*. 2014;17(8):1689-1696. doi:10.1017/S1368980013002115

The inflammatory weights were calculated based on a comprehensive literature review of 1,943 articles examining the relationship between diet and inflammatory biomarkers (IL-1β, IL-4, IL-6, IL-10, TNF-α, and CRP).

The global means and standard deviations were derived from 11 datasets from around the world representing diverse dietary patterns.

## Sample Data

The sample data (`data/sample_input.csv`) contains nutrient intake data from the National Health and Nutrition Examination Survey (NHANES) 2017-2018 cycle.

### Source

- **Dataset**: NHANES 2017-2018
- **Files**: Dietary Interview - Total Nutrient Intakes (DR1TOT_J, DR2TOT_J)
- **Processing**: Two-day average of nutrient intakes
- **Sample**: Participants meeting study inclusion criteria (n=13,577 + 3 validation rows)

### NHANES Citation

> Centers for Disease Control and Prevention (CDC). National Center for Health Statistics (NCHS). National Health and Nutrition Examination Survey Data. Hyattsville, MD: U.S. Department of Health and Human Services, Centers for Disease Control and Prevention, 2017-2018. https://wwwn.cdc.gov/nchs/nhanes/

### Validation Rows

The first three rows (SEQN 1, 2, 3) are synthetic validation cases:

| SEQN | Description | Expected DII |
|------|-------------|--------------|
| 1 | All nutrients at global mean | 0.0 |
| 2 | Maximally anti-inflammatory profile | -7.004394 |
| 3 | Maximally pro-inflammatory profile | +7.004394 |

These were constructed by setting nutrient values to:
- SEQN 1: Exactly the global mean for each nutrient
- SEQN 2: Values designed to minimize DII (favorable direction for each weight)
- SEQN 3: Values designed to maximize DII (opposite of SEQN 2)

## Validation

This implementation was validated against:

1. **Original R code** provided by the study statistician (Jeanette M. Andrade, PhD, RDN)
2. **Independent verification** by Jiyan Aslan Ceylan (University of Florida)
3. **Cross-validation** with the [dietaryindex R package](https://github.com/jamesjiadazhan/dietaryindex) by Jiada (James) Zhan

All 13,580 rows produce identical results (within floating-point precision) to the original implementation.

