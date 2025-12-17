# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-06-01

### Added
- Initial release of dii-calculator package
- Core DII calculation functionality with vectorized operations
- Support for all 45 DII food parameters from Shivappa et al. (2014)
- Detailed output option showing per-nutrient contributions (z-scores, percentiles, weighted contributions)
- Bundled reference table with global means, standard deviations, and inflammatory weights
- Automatic handling of missing nutrients
- Column name normalization (strips whitespace)
- Comprehensive test suite with validation against known DII scores
- Example notebooks (quickstart, validation)
- Sample NHANES data for testing and examples

### Validated
- Output verified against original R implementation used in PANDA-1 study
- Cross-validated with [dietaryindex R package](https://github.com/jamesjiadazhan/dietaryindex)
- Tested with 13,580 NHANES participants - perfect match with original calculations

