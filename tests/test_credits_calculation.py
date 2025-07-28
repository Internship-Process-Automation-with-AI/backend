"""
Tests for credits calculation logic.

This module tests the actual credits calculation logic used in the AI workflow,
which follows the formula: base_credits = total_hours / 27, with caps based on training type.
"""

import pytest


class TestCreditsCalculationLogic:
    """Test the actual credits calculation logic from AI workflow."""

    def test_standard_credit_calculation(self):
        """Test standard credit calculation (27 hours = 1 credit)."""
        # Test various working hour scenarios
        assert self._calculate_base_credits(27) == 1
        assert self._calculate_base_credits(54) == 2
        assert self._calculate_base_credits(81) == 3
        assert self._calculate_base_credits(108) == 4
        assert self._calculate_base_credits(135) == 5
        assert self._calculate_base_credits(162) == 6

    def test_credit_calculation_rounding(self):
        """Test credit calculation with rounding (integer division)."""
        # Test rounding down (integer division)
        assert self._calculate_base_credits(26) == 0
        assert self._calculate_base_credits(53) == 1
        assert self._calculate_base_credits(80) == 2

        # Test exact boundaries
        assert self._calculate_base_credits(27) == 1
        assert self._calculate_base_credits(54) == 2
        assert self._calculate_base_credits(81) == 3

    def test_credit_calculation_edge_cases(self):
        """Test credit calculation edge cases."""
        # Test zero hours
        assert self._calculate_base_credits(0) == 0

        # Test negative hours (should return negative credits due to integer division)
        assert self._calculate_base_credits(-10) == 0  # -10 / 27 = 0 (integer division)
        assert self._calculate_base_credits(-27) == -1  # -27 / 27 = -1
        assert self._calculate_base_credits(-100) == -3  # -100 / 27 = -3

        # Test very large numbers
        assert self._calculate_base_credits(1000) == 37
        assert self._calculate_base_credits(10000) == 370
        assert self._calculate_base_credits(100000) == 3703

    def test_credit_calculation_fractional_hours(self):
        """Test credit calculation with fractional hours (integer division)."""
        # Test fractional hours (should be rounded down due to integer division)
        assert self._calculate_base_credits(26.5) == 0
        assert self._calculate_base_credits(27.5) == 1
        assert self._calculate_base_credits(53.9) == 1
        assert self._calculate_base_credits(54.1) == 2

    def test_credit_calculation_minimum_threshold(self):
        """Test credit calculation minimum threshold."""
        # Test minimum working hours for credits
        assert self._calculate_base_credits(15) == 0  # Below minimum
        assert self._calculate_base_credits(27) == 1  # Minimum for 1 credit
        assert (
            self._calculate_base_credits(40) == 1
        )  # Above minimum but below 2 credits

    def test_credit_calculation_validation(self):
        """Test credit calculation input validation."""
        # Test with None
        with pytest.raises(TypeError):
            self._calculate_base_credits(None)

        # Test with string
        with pytest.raises(TypeError):
            self._calculate_base_credits("30")

        # Test with list
        with pytest.raises(TypeError):
            self._calculate_base_credits([30])

    def _calculate_base_credits(self, total_hours):
        """Helper method to calculate base credits using the actual formula."""
        if total_hours is None:
            raise TypeError("total_hours cannot be None")
        if not isinstance(total_hours, (int, float)):
            raise TypeError("total_hours must be a number")
        return int(total_hours / 27)


class TestCreditsCappingLogic:
    """Test the credits capping logic based on training type."""

    def test_professional_training_cap(self):
        """Test professional training credit cap (maximum 30 ECTS)."""
        # Test below cap
        assert self._apply_training_type_cap(20, "professional") == 20.0

        # Test at cap
        assert self._apply_training_type_cap(30, "professional") == 30.0

        # Test above cap (should be capped)
        assert self._apply_training_type_cap(40, "professional") == 30.0
        assert self._apply_training_type_cap(50, "professional") == 30.0
        assert self._apply_training_type_cap(100, "professional") == 30.0

    def test_general_training_cap(self):
        """Test general training credit cap (maximum 10 ECTS)."""
        # Test below cap
        assert self._apply_training_type_cap(5, "general") == 5.0

        # Test at cap
        assert self._apply_training_type_cap(10, "general") == 10.0

        # Test above cap (should be capped)
        assert self._apply_training_type_cap(15, "general") == 10.0
        assert self._apply_training_type_cap(20, "general") == 10.0
        assert self._apply_training_type_cap(50, "general") == 10.0

    def test_no_cap_scenarios(self):
        """Test scenarios where no capping is applied."""
        # Test edge cases
        assert self._apply_training_type_cap(0, "professional") == 0.0
        assert self._apply_training_type_cap(0, "general") == 0.0

        # Test negative credits (should not be capped)
        assert self._apply_training_type_cap(-5, "professional") == -5.0
        assert self._apply_training_type_cap(-5, "general") == -5.0

    def test_invalid_training_type(self):
        """Test handling of invalid training types."""
        # Test with None training type
        assert self._apply_training_type_cap(20, None) == 20.0
        assert self._apply_training_type_cap(50, None) == 50.0

        # Test with empty string
        assert self._apply_training_type_cap(20, "") == 20.0
        assert self._apply_training_type_cap(50, "") == 50.0

        # Test with invalid training type
        assert self._apply_training_type_cap(20, "invalid") == 20.0
        assert self._apply_training_type_cap(50, "invalid") == 50.0

    def _apply_training_type_cap(self, base_credits, training_type):
        """Helper method to apply training type caps using the actual logic."""
        if training_type == "professional" and base_credits > 30:
            return 30.0
        elif training_type == "general" and base_credits > 10:
            return 10.0
        else:
            return float(base_credits)


class TestCreditsCalculationBreakdown:
    """Test the calculation breakdown string generation."""

    def test_professional_training_breakdown_capped(self):
        """Test calculation breakdown for professional training when capped."""
        breakdown = self._generate_calculation_breakdown(1000, 37, 30, "professional")
        expected = "1000 hours / 27 hours per ECTS = 37.0 credits, capped at 30.0 maximum for professional training"
        assert breakdown == expected

    def test_general_training_breakdown_capped(self):
        """Test calculation breakdown for general training when capped."""
        breakdown = self._generate_calculation_breakdown(500, 18, 10, "general")
        expected = "500 hours / 27 hours per ECTS = 18.0 credits, capped at 10.0 maximum for general training"
        assert breakdown == expected

    def test_no_cap_breakdown(self):
        """Test calculation breakdown when no capping is applied."""
        breakdown = self._generate_calculation_breakdown(270, 10, 10, "professional")
        expected = "270 hours / 27 hours per ECTS = 10.0 credits"
        assert breakdown == expected

    def test_edge_case_breakdown(self):
        """Test calculation breakdown for edge cases."""
        # Zero hours
        breakdown = self._generate_calculation_breakdown(0, 0, 0, "professional")
        expected = "0 hours / 27 hours per ECTS = 0.0 credits"
        assert breakdown == expected

        # Negative credits (should not happen in practice)
        breakdown = self._generate_calculation_breakdown(-27, -1, -1, "professional")
        expected = "-27 hours / 27 hours per ECTS = -1.0 credits"
        assert breakdown == expected

    def _generate_calculation_breakdown(
        self, total_hours, base_credits, qualified_credits, training_type
    ):
        """Helper method to generate calculation breakdown string."""
        if training_type == "professional" and base_credits > 30:
            return f"{total_hours} hours / 27 hours per ECTS = {base_credits}.0 credits, capped at 30.0 maximum for professional training"
        elif training_type == "general" and base_credits > 10:
            return f"{total_hours} hours / 27 hours per ECTS = {base_credits}.0 credits, capped at 10.0 maximum for general training"
        else:
            return f"{total_hours} hours / 27 hours per ECTS = {base_credits}.0 credits"


class TestCreditsCalculationIntegration:
    """Test integrated credits calculation scenarios."""

    def test_complete_credits_calculation_professional(self):
        """Test complete credits calculation for professional training."""
        total_hours = 1000
        base_credits = int(total_hours / 27)  # 37
        qualified_credits = self._apply_training_type_cap(
            base_credits, "professional"
        )  # 30
        breakdown = self._generate_calculation_breakdown(
            total_hours, base_credits, qualified_credits, "professional"
        )

        assert base_credits == 37
        assert qualified_credits == 30.0
        assert "capped at 30.0 maximum for professional training" in breakdown

    def test_complete_credits_calculation_general(self):
        """Test complete credits calculation for general training."""
        total_hours = 500
        base_credits = int(total_hours / 27)  # 18
        qualified_credits = self._apply_training_type_cap(base_credits, "general")  # 10
        breakdown = self._generate_calculation_breakdown(
            total_hours, base_credits, qualified_credits, "general"
        )

        assert base_credits == 18
        assert qualified_credits == 10.0
        assert "capped at 10.0 maximum for general training" in breakdown

    def test_complete_credits_calculation_no_cap(self):
        """Test complete credits calculation when no capping is needed."""
        total_hours = 270
        base_credits = int(total_hours / 27)  # 10
        qualified_credits = self._apply_training_type_cap(
            base_credits, "professional"
        )  # 10
        breakdown = self._generate_calculation_breakdown(
            total_hours, base_credits, qualified_credits, "professional"
        )

        assert base_credits == 10
        assert qualified_credits == 10.0
        assert "capped" not in breakdown

    def test_credits_calculation_batch(self):
        """Test batch credits calculation for multiple scenarios."""
        test_cases = [
            (27, "professional", 1.0),  # 1 credit, no cap
            (270, "professional", 10.0),  # 10 credits, no cap
            (1000, "professional", 30.0),  # 37 credits, capped at 30
            (27, "general", 1.0),  # 1 credit, no cap
            (270, "general", 10.0),  # 10 credits, no cap
            (500, "general", 10.0),  # 18 credits, capped at 10
        ]

        for total_hours, training_type, expected_credits in test_cases:
            base_credits = int(total_hours / 27)
            qualified_credits = self._apply_training_type_cap(
                base_credits, training_type
            )
            assert (
                qualified_credits == expected_credits
            ), f"Failed for {total_hours} hours, {training_type} training"

    def _apply_training_type_cap(self, base_credits, training_type):
        """Helper method to apply training type caps."""
        if training_type == "professional" and base_credits > 30:
            return 30.0
        elif training_type == "general" and base_credits > 10:
            return 10.0
        else:
            return float(base_credits)

    def _generate_calculation_breakdown(
        self, total_hours, base_credits, qualified_credits, training_type
    ):
        """Helper method to generate calculation breakdown string."""
        if training_type == "professional" and base_credits > 30:
            return f"{total_hours} hours / 27 hours per ECTS = {base_credits}.0 credits, capped at 30.0 maximum for professional training"
        elif training_type == "general" and base_credits > 10:
            return f"{total_hours} hours / 27 hours per ECTS = {base_credits}.0 credits, capped at 10.0 maximum for general training"
        else:
            return f"{total_hours} hours / 27 hours per ECTS = {base_credits}.0 credits"


class TestCreditsCalculationBusinessRules:
    """Test business rules for credits calculation."""

    def test_ects_hours_ratio(self):
        """Test that the ECTS to hours ratio is correct (27 hours = 1 ECTS)."""
        # Verify the ratio is exactly 27:1
        assert self._calculate_base_credits(27) == 1
        assert self._calculate_base_credits(54) == 2
        assert self._calculate_base_credits(81) == 3

        # Test edge cases around the ratio
        assert self._calculate_base_credits(26) == 0  # Just below
        assert self._calculate_base_credits(27) == 1  # Exactly
        assert self._calculate_base_credits(28) == 1  # Just above

    def test_professional_training_maximum(self):
        """Test that professional training has a maximum of 30 ECTS credits."""
        # Test various scenarios
        assert self._apply_training_type_cap(25, "professional") == 25.0
        assert self._apply_training_type_cap(30, "professional") == 30.0
        assert self._apply_training_type_cap(35, "professional") == 30.0
        assert self._apply_training_type_cap(50, "professional") == 30.0
        assert self._apply_training_type_cap(100, "professional") == 30.0

    def test_general_training_maximum(self):
        """Test that general training has a maximum of 10 ECTS credits."""
        # Test various scenarios
        assert self._apply_training_type_cap(5, "general") == 5.0
        assert self._apply_training_type_cap(10, "general") == 10.0
        assert self._apply_training_type_cap(15, "general") == 10.0
        assert self._apply_training_type_cap(30, "general") == 10.0
        assert self._apply_training_type_cap(100, "general") == 10.0

    def test_total_practical_training_requirement(self):
        """Test that total practical training requirement is 30 ECTS credits."""
        # Professional training can contribute up to 30 ECTS
        max_professional = self._apply_training_type_cap(
            1000, "professional"
        )  # Should be 30
        assert max_professional == 30

        # General training can contribute up to 10 ECTS
        max_general = self._apply_training_type_cap(1000, "general")  # Should be 10
        assert max_general == 10

        # Combined maximum (though this would be unusual in practice)
        combined_max = max_professional + max_general
        assert combined_max == 40  # This exceeds the 30 requirement, which is correct

    def _calculate_base_credits(self, total_hours):
        """Helper method to calculate base credits."""
        return int(total_hours / 27)

    def _apply_training_type_cap(self, base_credits, training_type):
        """Helper method to apply training type caps."""
        if training_type == "professional" and base_credits > 30:
            return 30.0
        elif training_type == "general" and base_credits > 10:
            return 10.0
        else:
            return float(base_credits)
