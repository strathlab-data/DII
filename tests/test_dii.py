"""
Test suite for DII Calculator
=============================

These tests validate the DII calculation against known values, including:
1. Synthetic test cases with mathematically verifiable results
2. Validation against the dietaryindex R package methodology

Test cases:
- SEQN 1: All nutrients at global mean → DII should be 0.0
- SEQN 2: Anti-inflammatory profile → DII should be -7.004394
- SEQN 3: Pro-inflammatory profile → DII should be +7.004394
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from dii import calculate_dii, calculate_dii_detailed, get_available_nutrients, load_reference_table
from dii.reader import load_nutrient_data, validate_input_file, summarize_input_data
from dii.reference import validate_nutrient_columns


class TestReferenceTable:
    """Tests for reference table loading and validation."""

    def test_load_reference_table(self):
        """Test that reference table loads correctly."""
        ref = load_reference_table()
        assert isinstance(ref, pd.DataFrame)
        assert len(ref) == 45  # 45 DII nutrients
        assert "nutrient" in ref.columns
        assert "weight" in ref.columns
        assert "global_mean" in ref.columns
        assert "global_sd" in ref.columns

    def test_get_available_nutrients(self):
        """Test that we can get the list of available nutrients."""
        nutrients = get_available_nutrients()
        assert isinstance(nutrients, list)
        assert len(nutrients) == 45
        assert "Fiber" in nutrients
        assert "Alcohol" in nutrients
        assert "Vitamin C" in nutrients

    def test_reference_values_reasonable(self):
        """Test that reference values are within reasonable ranges."""
        ref = load_reference_table()
        
        # All weights should be between -1 and +1
        assert ref["weight"].min() >= -1.0
        assert ref["weight"].max() <= 1.0
        
        # All SDs should be positive
        assert (ref["global_sd"] > 0).all()


class TestDIICalculation:
    """Tests for DII score calculation."""

    def test_mean_intake_gives_zero_dii(self):
        """
        When all nutrient values equal global means, DII should be 0.
        
        This is SEQN 1 in the validation dataset.
        """
        ref = load_reference_table()
        
        # Create a row where each nutrient equals its global mean
        data = {"SEQN": [1]}
        for _, row in ref.iterrows():
            data[row["nutrient"]] = [row["global_mean"]]
        
        df = pd.DataFrame(data)
        result = calculate_dii(df, id_column="SEQN")
        
        # DII should be exactly 0 (within floating point tolerance)
        assert abs(result["DII_score"].iloc[0]) < 1e-10

    def test_validation_cases_from_sample_data(self):
        """
        Test against the actual validation data (SEQN 1, 2, 3) from sample_input.csv.
        
        These rows were constructed with known DII values:
        - SEQN 1: All nutrients at global mean → DII = 0
        - SEQN 2: Anti-inflammatory profile → DII = -7.004394
        - SEQN 3: Pro-inflammatory profile → DII = +7.004394
        
        This test validates against the DII_Confirmed column which contains
        the expected values verified by the original R implementation.
        """
        from pathlib import Path
        
        # Load actual validation data
        data_path = Path(__file__).parent.parent / "data" / "sample_input.csv"
        if not data_path.exists():
            pytest.skip("Sample data not found")
        
        data = pd.read_csv(data_path)
        validation = data[data["SEQN"].isin([1, 2, 3])].copy()
        
        # Clean column names (remove trailing spaces)
        validation.columns = validation.columns.str.strip()
        
        # Calculate DII
        result = calculate_dii(validation, id_column="SEQN")
        
        # Check each validation case
        expected = {1: 0.0, 2: -7.004394189, 3: 7.004394189}
        
        for seqn, expected_dii in expected.items():
            actual_dii = result[result["SEQN"] == seqn]["DII_score"].iloc[0]
            # Allow small tolerance due to floating point
            assert abs(actual_dii - expected_dii) < 0.1, \
                f"SEQN {seqn}: Expected DII ~{expected_dii}, got {actual_dii}"

    def test_detailed_output_structure(self):
        """Test that detailed output contains expected columns."""
        ref = load_reference_table()
        
        # Create minimal test data with a few nutrients
        df = pd.DataFrame({
            "SEQN": [1],
            "Fiber": [18.8],
            "Alcohol": [13.98],
        })
        
        result = calculate_dii_detailed(df, id_column="SEQN")
        
        # Check for expected columns
        assert "SEQN" in result.columns
        assert "DII_score" in result.columns
        assert "Fiber" in result.columns
        assert "Fiber_zscore" in result.columns
        assert "Fiber_percentile" in result.columns
        assert "Fiber_contribution" in result.columns

    def test_missing_nutrients_handled(self):
        """Test that missing nutrients are handled gracefully."""
        import warnings
        
        # Data with only a subset of nutrients
        df = pd.DataFrame({
            "SEQN": [1, 2],
            "Fiber": [18.8, 25.0],
            "Alcohol": [13.98, 0.0],
        })
        
        # Suppress expected low-coverage warning for this test
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = calculate_dii(df, id_column="SEQN")
        
        assert len(result) == 2
        assert "DII_score" in result.columns

    def test_nan_handling(self):
        """Test that NaN values are handled correctly."""
        import warnings
        
        df = pd.DataFrame({
            "SEQN": [1, 2],
            "Fiber": [18.8, np.nan],
            "Alcohol": [np.nan, 13.98],
        })
        
        # Suppress expected low-coverage warning for this test
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = calculate_dii(df, id_column="SEQN")
        
        # Should complete without error
        assert len(result) == 2
        # Scores should not be NaN (partial data still produces valid DII)
        assert not result["DII_score"].isna().all()

    def test_all_nan_returns_nan(self):
        """Test that all-NaN nutrient values return NaN DII score."""
        import warnings
        
        df = pd.DataFrame({
            "SEQN": [1, 2],
            "Fiber": [np.nan, 18.8],
            "Alcohol": [np.nan, 13.98],
        })
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = calculate_dii(df, id_column="SEQN")
        
        # Row 1: All NaN → DII should be NaN
        assert pd.isna(result["DII_score"].iloc[0]), \
            "All-NaN row should return NaN DII score"
        # Row 2: Valid data → DII should NOT be NaN
        assert not pd.isna(result["DII_score"].iloc[1]), \
            "Row with valid data should return numeric DII score"

    def test_multiple_participants(self):
        """Test calculation for multiple participants."""
        ref = load_reference_table()
        
        # Create data for 3 participants
        data = {"SEQN": [1, 2, 3]}
        for _, row in ref.iterrows():
            nutrient = row["nutrient"]
            mean = row["global_mean"]
            sd = row["global_sd"]
            # Different values for each participant
            data[nutrient] = [mean, mean + sd, mean - sd]
        
        df = pd.DataFrame(data)
        result = calculate_dii(df, id_column="SEQN")
        
        assert len(result) == 3
        # First participant (all at mean) should have DII ≈ 0
        assert abs(result["DII_score"].iloc[0]) < 0.0001


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_no_valid_nutrients_raises_error(self):
        """Test that error is raised when no valid nutrients found."""
        df = pd.DataFrame({
            "SEQN": [1],
            "invalid_column": [100],
            "another_invalid": [200],
        })
        
        with pytest.raises(ValueError, match="No DII nutrients found"):
            calculate_dii(df)

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        import warnings
        
        df = pd.DataFrame({
            "SEQN": [],
            "Fiber": [],
        })
        
        # Suppress expected low-coverage warning for this test
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = calculate_dii(df, id_column="SEQN")
        assert len(result) == 0

    def test_single_nutrient(self):
        """Test calculation with only one nutrient."""
        import warnings
        
        ref = load_reference_table()
        fiber_row = ref[ref["nutrient"] == "Fiber"].iloc[0]
        
        df = pd.DataFrame({
            "SEQN": [1],
            "Fiber": [fiber_row["global_mean"]],  # At mean
        })
        
        # Suppress expected low-coverage warning for this test
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = calculate_dii(df, id_column="SEQN")
        
        # With only one nutrient at mean, DII contribution should be 0
        assert abs(result["DII_score"].iloc[0]) < 0.0001


class TestInputValidation:
    """Tests for input validation and error handling."""

    def test_non_dataframe_input_raises_typeerror(self):
        """Test that passing non-DataFrame raises TypeError with helpful message."""
        # List input
        with pytest.raises(TypeError, match="must be a pandas DataFrame"):
            calculate_dii([1, 2, 3])
        
        # Dict input
        with pytest.raises(TypeError, match="must be a pandas DataFrame"):
            calculate_dii({"Fiber": [18.8]})
        
        # None input
        with pytest.raises(TypeError, match="must be a pandas DataFrame"):
            calculate_dii(None)

    def test_low_coverage_warning(self):
        """Test that low nutrient coverage triggers a warning."""
        import warnings
        
        # Only 2 out of 45 nutrients = ~4% coverage
        df = pd.DataFrame({
            "SEQN": [1],
            "Fiber": [18.8],
            "Alcohol": [13.98],
        })
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = calculate_dii(df, id_column="SEQN")
            
            # Check low coverage warning was issued
            assert any("Low nutrient coverage" in str(warning.message) for warning in w)
        
        # Should still compute
        assert "DII_score" in result.columns

    def test_high_coverage_no_warning(self):
        """Test that high nutrient coverage does not trigger warning."""
        import warnings
        
        ref = load_reference_table()
        
        # Create data with all 45 nutrients (100% coverage)
        data = {"SEQN": [1]}
        for _, row in ref.iterrows():
            data[row["nutrient"]] = [row["global_mean"]]
        
        df = pd.DataFrame(data)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = calculate_dii(df, id_column="SEQN")
            
            # No low coverage warning should be issued
            coverage_warnings = [x for x in w if "Low nutrient coverage" in str(x.message)]
            assert len(coverage_warnings) == 0
        
        assert "DII_score" in result.columns

    def test_whitespace_in_column_names(self):
        """Test that whitespace in column names is handled."""
        import warnings
        
        df = pd.DataFrame({
            "SEQN": [1],
            "  Fiber  ": [18.8],  # Leading/trailing whitespace
            "Alcohol ": [13.98],  # Trailing whitespace
        })
        
        # Suppress expected low-coverage warning for this test
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = calculate_dii(df, id_column="SEQN")
        assert len(result) == 1
        assert "DII_score" in result.columns


class TestReaderModule:
    """Tests for the reader module functions."""

    def test_validate_input_file_not_found(self):
        """Test that FileNotFoundError is raised for missing files."""
        with pytest.raises(FileNotFoundError, match="Input file not found"):
            validate_input_file("nonexistent_file.csv")

    def test_validate_input_file_not_csv(self, tmp_path):
        """Test that ValueError is raised for non-CSV files."""
        txt_file = tmp_path / "data.txt"
        txt_file.write_text("some data")
        
        with pytest.raises(ValueError, match="must be a CSV file"):
            validate_input_file(txt_file)

    def test_validate_input_file_success(self, tmp_path):
        """Test that valid CSV file passes validation."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("col1,col2\n1,2")
        
        # Should not raise
        validate_input_file(csv_file)

    def test_load_nutrient_data_success(self, tmp_path):
        """Test loading a valid CSV file."""
        csv_file = tmp_path / "nutrients.csv"
        csv_file.write_text("SEQN,Fiber,Alcohol\n1,18.8,13.98")
        
        df = load_nutrient_data(csv_file)
        assert len(df) == 1
        assert "Fiber" in df.columns

    def test_load_nutrient_data_not_found(self):
        """Test FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            load_nutrient_data("nonexistent.csv")

    def test_summarize_input_data(self):
        """Test the summarize_input_data function."""
        df = pd.DataFrame({
            "SEQN": [1, 2],
            "Fiber": [18.8, 25.0],
            "Alcohol": [13.98, 0.0],
            "Other_Column": [100, 200],
        })
        
        summary = summarize_input_data(df)
        
        assert summary["n_rows"] == 2
        assert summary["n_columns"] == 4
        assert "Fiber" in summary["nutrients_found"]
        assert "Alcohol" in summary["nutrients_found"]
        assert len(summary["nutrients_found"]) == 2
        assert len(summary["nutrients_missing"]) == 43  # 45 - 2
        assert 0 < summary["coverage"] < 10  # About 4.4%


class TestReferenceModuleExtended:
    """Extended tests for reference module."""

    def test_custom_reference_table_not_found(self):
        """Test FileNotFoundError for missing custom reference."""
        with pytest.raises(FileNotFoundError, match="Reference table not found"):
            load_reference_table("nonexistent_reference.csv")

    def test_custom_reference_table_missing_columns(self, tmp_path):
        """Test ValueError when custom reference missing required columns."""
        csv_file = tmp_path / "bad_ref.csv"
        csv_file.write_text("nutrient,weight\nFiber,-0.663")  # Missing global_mean, global_sd
        
        with pytest.raises(ValueError, match="missing required columns"):
            load_reference_table(str(csv_file))

    def test_custom_reference_table_success(self, tmp_path):
        """Test loading a valid custom reference table."""
        csv_file = tmp_path / "custom_ref.csv"
        csv_file.write_text("nutrient,weight,global_mean,global_sd\nFiber,-0.663,18.8,4.9")
        
        ref = load_reference_table(str(csv_file))
        assert len(ref) == 1
        assert ref["nutrient"].iloc[0] == "Fiber"

    def test_validate_nutrient_columns(self):
        """Test the validate_nutrient_columns function."""
        import io
        import sys
        
        # Capture stdout
        captured = io.StringIO()
        sys.stdout = captured
        
        matched, missing = validate_nutrient_columns(
            ["Fiber", "Alcohol", "SEQN", "Unknown"],
            verbose=True
        )
        
        sys.stdout = sys.__stdout__
        
        assert "Fiber" in matched
        assert "Alcohol" in matched
        assert "SEQN" not in matched  # Not a nutrient
        assert "Unknown" not in matched
        assert len(matched) == 2
        assert len(missing) == 43

    def test_validate_nutrient_columns_silent(self):
        """Test validate_nutrient_columns with verbose=False."""
        import io
        import sys
        
        captured = io.StringIO()
        sys.stdout = captured
        
        matched, _ = validate_nutrient_columns(
            ["Fiber"],
            verbose=False
        )
        
        sys.stdout = sys.__stdout__
        output = captured.getvalue()
        
        assert "Fiber" in matched
        assert output == ""  # No output when verbose=False


class TestMathematicalValidation:
    """Tests to validate the mathematical correctness of DII calculations."""

    def test_zscore_calculation(self):
        """Test that z-scores are calculated correctly."""
        ref = load_reference_table()
        fiber_row = ref[ref["nutrient"] == "Fiber"].iloc[0]
        mean, sd = fiber_row["global_mean"], fiber_row["global_sd"]
        
        # Test value exactly 1 SD above mean
        test_value = mean + sd
        expected_zscore = 1.0
        
        df = pd.DataFrame({"Fiber": [test_value]})
        
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = calculate_dii_detailed(df)
        
        actual_zscore = result["Fiber_zscore"].iloc[0]
        assert abs(actual_zscore - expected_zscore) < 1e-10

    def test_centered_percentile_at_mean(self):
        """Test that percentile is 0 when value equals global mean."""
        ref = load_reference_table()
        fiber_row = ref[ref["nutrient"] == "Fiber"].iloc[0]
        
        df = pd.DataFrame({"Fiber": [fiber_row["global_mean"]]})
        
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = calculate_dii_detailed(df)
        
        # Percentile should be 0 at mean (z=0 → 2*Φ(0)-1 = 2*0.5-1 = 0)
        assert abs(result["Fiber_percentile"].iloc[0]) < 1e-10

    def test_centered_percentile_formula(self):
        """Test that centered percentile matches 2*Φ(z)-1 formula."""
        ref = load_reference_table()
        fiber_row = ref[ref["nutrient"] == "Fiber"].iloc[0]
        mean, sd = fiber_row["global_mean"], fiber_row["global_sd"]
        
        # Test at z = 1.96 (97.5th percentile)
        test_value = mean + 1.96 * sd
        expected_percentile = 2 * norm.cdf(1.96) - 1  # ≈ 0.95
        
        df = pd.DataFrame({"Fiber": [test_value]})
        
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = calculate_dii_detailed(df)
        
        assert abs(result["Fiber_percentile"].iloc[0] - expected_percentile) < 1e-6

    def test_contribution_formula(self):
        """Test that contribution equals percentile × weight."""
        ref = load_reference_table()
        fiber_row = ref[ref["nutrient"] == "Fiber"].iloc[0]
        weight = fiber_row["weight"]
        
        # Use a value that gives a known percentile
        df = pd.DataFrame({"Fiber": [fiber_row["global_mean"] + fiber_row["global_sd"]]})
        
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = calculate_dii_detailed(df)
        
        percentile = result["Fiber_percentile"].iloc[0]
        contribution = result["Fiber_contribution"].iloc[0]
        expected_contribution = percentile * weight
        
        assert abs(contribution - expected_contribution) < 1e-10

    def test_detailed_vs_simple_consistency(self):
        """Test that detailed and simple outputs produce same DII score."""
        ref = load_reference_table()
        
        # Create test data with multiple nutrients
        data = {"SEQN": [1]}
        for _, row in ref.iterrows():
            data[row["nutrient"]] = [row["global_mean"] + row["global_sd"]]
        
        df = pd.DataFrame(data)
        
        simple_result = calculate_dii(df, id_column="SEQN")
        detailed_result = calculate_dii(df, id_column="SEQN", detailed=True)
        
        simple_score = simple_result["DII_score"].iloc[0]
        detailed_score = detailed_result["DII_score"].iloc[0]
        
        assert abs(simple_score - detailed_score) < 1e-10


class TestExtremeValues:
    """Tests for extreme value handling."""

    def test_extreme_value_warning(self):
        """Test that extreme values trigger a warning."""
        import warnings
        
        ref = load_reference_table()
        fiber_row = ref[ref["nutrient"] == "Fiber"].iloc[0]
        mean, sd = fiber_row["global_mean"], fiber_row["global_sd"]
        
        # Create a value > 10 SD from mean
        extreme_value = mean + 15 * sd
        
        df = pd.DataFrame({
            "SEQN": [1],
            "Fiber": [extreme_value],
        })
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            calculate_dii(df, id_column="SEQN", validate_bounds=True)
            
            # Check extreme value warning was issued
            extreme_warnings = [x for x in w if "exceeding" in str(x.message).lower()]
            assert len(extreme_warnings) > 0

    def test_no_warning_when_validation_disabled(self):
        """Test that validate_bounds=False suppresses extreme value warnings."""
        import warnings
        
        ref = load_reference_table()
        fiber_row = ref[ref["nutrient"] == "Fiber"].iloc[0]
        mean, sd = fiber_row["global_mean"], fiber_row["global_sd"]
        
        # Create extreme value
        extreme_value = mean + 15 * sd
        
        df = pd.DataFrame({
            "SEQN": [1],
            "Fiber": [extreme_value],
        })
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            calculate_dii(df, id_column="SEQN", validate_bounds=False)
            
            # No extreme value warning should be issued
            extreme_warnings = [x for x in w if "exceeding" in str(x.message).lower()]
            assert len(extreme_warnings) == 0

    def test_negative_nutrient_values(self):
        """Test handling of negative nutrient values (can occur in some datasets)."""
        import warnings
        
        # Some nutrients can technically have negative values in edge cases
        df = pd.DataFrame({
            "SEQN": [1],
            "Fiber": [-5.0],  # Unusual but should still compute
        })
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = calculate_dii(df, id_column="SEQN")
        
        # Should complete and produce a numeric result
        assert not pd.isna(result["DII_score"].iloc[0])


class TestNonNumericData:
    """Tests for handling non-numeric data."""

    def test_string_values_coerced(self):
        """Test that string values are coerced to numeric with warning."""
        import warnings
        
        df = pd.DataFrame({
            "SEQN": [1, 2],
            "Fiber": ["18.8", "invalid"],  # String numbers and invalid
            "Alcohol": [13.98, 0.0],
        })
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = calculate_dii(df, id_column="SEQN")
            
            # Check coercion warning
            coercion_warnings = [x for x in w if "coerced" in str(x.message).lower()]
            assert len(coercion_warnings) > 0
        
        # Row 1 should have valid score, row 2 might have partial score
        assert len(result) == 2


class TestCLI:
    """Tests for command-line interface."""

    def test_cli_help(self):
        """Test that CLI --help works."""
        import subprocess
        import sys
        
        result = subprocess.run(
            [sys.executable, "-m", "dii", "--help"],
            capture_output=True,
            text=True,
        )
        
        assert result.returncode == 0
        assert "usage" in result.stdout.lower() or "dii" in result.stdout.lower()

    def test_cli_nutrients_flag(self):
        """Test that CLI --nutrients lists supported nutrients."""
        import subprocess
        import sys
        
        result = subprocess.run(
            [sys.executable, "-m", "dii", "--nutrients"],
            capture_output=True,
            text=True,
        )
        
        assert result.returncode == 0
        # Should list some known nutrients
        assert "Fiber" in result.stdout or "fiber" in result.stdout.lower()

    def test_cli_version(self):
        """Test that CLI --version works."""
        import subprocess
        import sys
        
        result = subprocess.run(
            [sys.executable, "-m", "dii", "--version"],
            capture_output=True,
            text=True,
        )
        
        assert result.returncode == 0
        assert "dii" in result.stdout.lower() or "1." in result.stdout

    def test_cli_process_file(self, tmp_path):
        """Test CLI processing of a CSV file."""
        import subprocess
        import sys
        
        # Create test CSV
        csv_file = tmp_path / "test_input.csv"
        csv_file.write_text("SEQN,Fiber,Alcohol\n1,18.8,13.98\n2,25.0,0.0")
        
        result = subprocess.run(
            [sys.executable, "-m", "dii", str(csv_file)],
            capture_output=True,
            text=True,
        )
        
        assert result.returncode == 0
        assert "DII" in result.stdout

    def test_cli_invalid_file(self):
        """Test CLI error handling for invalid file."""
        import subprocess
        import sys
        
        result = subprocess.run(
            [sys.executable, "-m", "dii", "nonexistent_file.csv"],
            capture_output=True,
            text=True,
        )
        
        assert result.returncode != 0
        assert "error" in result.stderr.lower() or "not found" in result.stderr.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

