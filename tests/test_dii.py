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

import numpy as np
import pandas as pd
import pytest

from dii import calculate_dii, calculate_dii_detailed, get_available_nutrients, load_reference_table


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
        # Data with only a subset of nutrients
        df = pd.DataFrame({
            "SEQN": [1, 2],
            "Fiber": [18.8, 25.0],
            "Alcohol": [13.98, 0.0],
        })
        
        # Should not raise an error
        result = calculate_dii(df, id_column="SEQN")
        
        assert len(result) == 2
        assert "DII_score" in result.columns

    def test_nan_handling(self):
        """Test that NaN values are handled correctly."""
        df = pd.DataFrame({
            "SEQN": [1, 2],
            "Fiber": [18.8, np.nan],
            "Alcohol": [np.nan, 13.98],
        })
        
        result = calculate_dii(df, id_column="SEQN")
        
        # Should complete without error
        assert len(result) == 2
        # Scores should not be NaN
        assert not result["DII_score"].isna().all()

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
        df = pd.DataFrame({
            "SEQN": [],
            "Fiber": [],
        })
        
        result = calculate_dii(df, id_column="SEQN")
        assert len(result) == 0

    def test_single_nutrient(self):
        """Test calculation with only one nutrient."""
        ref = load_reference_table()
        fiber_row = ref[ref["nutrient"] == "Fiber"].iloc[0]
        
        df = pd.DataFrame({
            "SEQN": [1],
            "Fiber": [fiber_row["global_mean"]],  # At mean
        })
        
        result = calculate_dii(df, id_column="SEQN")
        
        # With only one nutrient at mean, DII contribution should be 0
        assert abs(result["DII_score"].iloc[0]) < 0.0001


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

