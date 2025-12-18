# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.13] - 2025-12-17

### Added
- **Scientific Precision Constants**: Explicit `FLOAT_DTYPE` (numpy.float64) and `VALIDATION_TOLERANCE` (1e-10) constants for reproducibility
- **Bounds Validation**: New `validate_bounds` parameter warns about nutrient values >10 SD from global mean (catches unit errors like mg vs g)
- **Infinity Handling**: Division edge cases now properly convert to NaN instead of inf
- **Module Exports**: Added `__all__` to `calculator.py` and `reference.py` for explicit API definition
- **Comprehensive README**: Complete rewrite with:
  - Full methodology section with mathematical equations
  - Complete units table for all 45 nutrients
  - All three visualization examples with images
  - Templates documentation
  - Enhanced validation section with precision metrics
  - API reference
  - Proper academic citations

### Changed
- All internal calculations now explicitly use `numpy.float64` dtype
- Improved warning messages for low nutrient coverage (<25%)
- Enhanced docstrings with more examples and precision notes
- Validation notebook now includes side-by-side comparison tables and statistical summaries

### Fixed
- Import statements moved to module top (was inside functions)
- Type hints now use `Tuple` from typing for Python 3.9 compatibility

### Documentation
- Added complete units reference for all 45 DII nutrients
- Enhanced CITATION.cff with full reference metadata
- Added methodology section explaining z-score, centered percentile, and weighting

## [1.0.12] - 2025-12-16

### Fixed
- PyPI badge caching issue resolved with cache-busting parameter
- Combined GitHub Release and PyPI publish into single workflow

## [1.0.11] - 2025-12-16

### Fixed
- GitHub Actions workflow for trusted PyPI publishing

## [1.0.10] - 2025-12-16

### Fixed
- BumpVer configuration for CITATION.cff pattern matching

## [1.0.9] - 2025-12-16

### Added
- Automated version synchronization with BumpVer across pyproject.toml, __init__.py, and CITATION.cff

## [1.0.8] - 2025-12-16

### Fixed
- Version consistency across all package files

## [1.0.7] - 2025-12-16

### Changed
- Updated minimum Python version to 3.10
- Removed Python 3.8 and 3.9 from test matrix

### Fixed
- CI test failures on older Python versions

## [1.0.6] - 2025-12-16

### Added
- Trusted publisher configuration for PyPI

## [1.0.5] - 2025-12-15

### Added
- GitHub Actions workflows for automated testing and PyPI publishing
- Input validation with TypeError for non-DataFrame inputs
- Low nutrient coverage warning (<25% of DII nutrients)
- Numeric coercion for string columns with warnings

### Changed
- Made matplotlib a required dependency (was optional)

### Fixed
- Visualization module simplified after matplotlib became required

## [1.0.4] - 2025-12-15

### Fixed
- Fresh installation test verification
- Package data inclusion in builds

## [1.0.3] - 2025-12-15

### Changed
- README updates for PyPI display

## [1.0.2] - 2025-12-15

### Added
- Visualization functions: `plot_dii_distribution`, `plot_nutrient_contributions`, `plot_dii_categories_pie`
- Command-line interface (`dii` command)
- Input templates in `templates/` folder

## [1.0.1] - 2025-12-14

### Added
- CITATION.cff for proper academic citation
- DATA_SOURCES.md documenting reference table provenance

### Changed
- Restructured to src/ layout
- Added comprehensive docstrings

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

[Unreleased]: https://github.com/strathlab-data/DII/compare/v1.0.13...HEAD
[1.0.13]: https://github.com/strathlab-data/DII/compare/v1.0.12...v1.0.13
[1.0.12]: https://github.com/strathlab-data/DII/compare/v1.0.11...v1.0.12
[1.0.11]: https://github.com/strathlab-data/DII/compare/v1.0.10...v1.0.11
[1.0.10]: https://github.com/strathlab-data/DII/compare/v1.0.9...v1.0.10
[1.0.9]: https://github.com/strathlab-data/DII/compare/v1.0.8...v1.0.9
[1.0.8]: https://github.com/strathlab-data/DII/compare/v1.0.7...v1.0.8
[1.0.7]: https://github.com/strathlab-data/DII/compare/v1.0.6...v1.0.7
[1.0.6]: https://github.com/strathlab-data/DII/compare/v1.0.5...v1.0.6
[1.0.5]: https://github.com/strathlab-data/DII/compare/v1.0.4...v1.0.5
[1.0.4]: https://github.com/strathlab-data/DII/compare/v1.0.3...v1.0.4
[1.0.3]: https://github.com/strathlab-data/DII/compare/v1.0.2...v1.0.3
[1.0.2]: https://github.com/strathlab-data/DII/compare/v1.0.1...v1.0.2
[1.0.1]: https://github.com/strathlab-data/DII/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/strathlab-data/DII/releases/tag/v1.0.0
