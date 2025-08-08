"""
Pydantic models for structural validation of LLM extraction and evaluation results.
Provides fast, reliable validation for dates, timelines, and business rules.
"""

import logging
from datetime import date, datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, computed_field, validator

logger = logging.getLogger(__name__)


class Position(BaseModel):
    """Model for individual employment positions."""

    model_config = ConfigDict(use_enum_values=True)

    title: str
    employer: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    duration: Optional[str] = None
    responsibilities: Optional[str] = None

    @validator("start_date", "end_date")
    def validate_dates(cls, v):
        """Validate that dates are not in the future."""
        if v:
            try:
                date_obj = datetime.strptime(v, "%Y-%m-%d").date()
                if date_obj > date.today():
                    logger.warning(
                        f"Future date detected: {v}. This may indicate extraction errors but processing will continue."
                    )
            except ValueError:
                logger.warning(
                    f"Invalid date format: {v}. Expected YYYY-MM-DD. Processing will continue."
                )
        return v

    @validator("end_date")
    def validate_end_after_start(cls, v, values):
        """Validate that end date is after start date."""
        if v and "start_date" in values and values["start_date"]:
            try:
                end_date = datetime.strptime(v, "%Y-%m-%d").date()
                start_date = datetime.strptime(values["start_date"], "%Y-%m-%d").date()
                if end_date < start_date:
                    logger.warning(
                        f"End date {v} is before start date {values['start_date']}. This may indicate extraction errors but processing will continue."
                    )
            except ValueError as e:
                # If date parsing fails, skip this validation
                logger.warning(f"Date parsing failed during end/start validation: {e}")
                pass
        return v

    @computed_field
    @property
    def duration_days(self) -> Optional[int]:
        """Calculate duration in days if both dates are available."""
        if self.start_date and self.end_date:
            try:
                start_date = datetime.strptime(self.start_date, "%Y-%m-%d").date()
                end_date = datetime.strptime(self.end_date, "%Y-%m-%d").date()
                return (end_date - start_date).days
            except ValueError:
                return None
        return None

    @computed_field
    @property
    def duration_years(self) -> Optional[float]:
        """Calculate duration in years if both dates are available."""
        if self.duration_days:
            return self.duration_days / 365.25
        return None


class ExtractionResults(BaseModel):
    """Model for LLM extraction results with structural validation."""

    employee_name: str
    employer: Optional[str] = None
    certificate_issue_date: Optional[str] = None
    positions: List[Position] = Field(default_factory=list)
    total_employment_period: Optional[str] = None
    document_language: str = Field(default="en")
    confidence_level: Optional[str] = None

    @validator("certificate_issue_date")
    def validate_certificate_date(cls, v):
        """Validate certificate issue date is not in the future."""
        if v:
            try:
                date_obj = datetime.strptime(v, "%Y-%m-%d").date()
                if date_obj > date.today():
                    logger.warning(
                        f"Certificate issue date is in the future: {v}. This may indicate extraction errors but processing will continue."
                    )
            except ValueError:
                logger.warning(
                    f"Invalid certificate issue date format: {v}. Expected YYYY-MM-DD. Processing will continue."
                )
        return v

    @validator("positions")
    def validate_employment_sequence(cls, v):
        """Validate employment timeline consistency."""
        if len(v) < 2:
            return v

        # Sort positions by start date
        sorted_positions = sorted(
            [p for p in v if p.start_date], key=lambda x: x.start_date
        )

        if len(sorted_positions) < 2:
            return v

        # Check for overlapping or impossible sequences
        for i in range(1, len(sorted_positions)):
            prev_end = sorted_positions[i - 1].end_date
            curr_start = sorted_positions[i].start_date

            if prev_end and curr_start:
                try:
                    prev_end_date = datetime.strptime(prev_end, "%Y-%m-%d").date()
                    curr_start_date = datetime.strptime(curr_start, "%Y-%m-%d").date()

                    if prev_end_date > curr_start_date:
                        # Log warning instead of raising error
                        logger.warning(
                            f"Employment periods overlap or are in wrong sequence: "
                            f"Position {i - 1} ends {prev_end} after position {i} starts {curr_start}. "
                            f"This may indicate extraction errors but processing will continue."
                        )

                    # Check for suspicious gaps (more than 10 years between positions)
                    gap_days = (curr_start_date - prev_end_date).days
                    if gap_days > 3650:  # 10 years
                        logger.warning(
                            f"Large gap detected between employment periods: "
                            f"{gap_days} days between {prev_end} and {curr_start}"
                        )
                except ValueError as e:
                    # If date parsing fails, skip this validation
                    logger.warning(
                        f"Date parsing failed during employment sequence validation: {e}"
                    )
                    pass

        return v

    @validator("positions")
    def validate_employer_consistency(cls, v):
        """Validate employer consistency across positions."""
        if len(v) < 2:
            return v

        employers = [p.employer for p in v if p.employer]
        if len(set(employers)) == 1:
            # All positions have the same employer
            return v

        # Check for positions with null employers when others have employers
        has_employer = any(p.employer for p in v)
        null_employers = [p for p in v if p.employer is None]

        if has_employer and null_employers:
            logger.warning(
                f"Inconsistent employer information: {len(null_employers)} positions "
                f"have null employers while others specify employers"
            )

        return v

    @computed_field
    @property
    def total_employment_days(self) -> int:
        """Calculate total employment days across all positions."""
        total = 0
        for position in self.positions:
            if position.duration_days:
                total += position.duration_days
        return total

    @computed_field
    @property
    def total_employment_years(self) -> float:
        """Calculate total employment years across all positions."""
        return (
            self.total_employment_days / 365.25
            if self.total_employment_days > 0
            else 0.0
        )


class EvaluationResults(BaseModel):
    """Model for LLM evaluation results with business rule validation."""

    total_working_hours: Optional[int] = None
    requested_training_type: str = Field(..., pattern="^(general|professional)$")
    credits_calculated: Optional[float] = None
    credits_qualified: float
    degree_relevance: str = Field(..., pattern="^(high|medium|low)$")
    relevance_explanation: str
    calculation_breakdown: str
    summary_justification: str
    decision: str = Field(..., pattern="^(ACCEPTED|REJECTED)$")
    justification: str
    recommendation: Optional[str] = None
    confidence_level: Optional[str] = None

    @validator("total_working_hours")
    def validate_working_hours(cls, v):
        """Validate working hours are reasonable."""
        if v is not None:
            if v < 0:
                raise ValueError(f"Working hours cannot be negative: {v}")
            if v > 100000:  # More than ~50 years of full-time work
                raise ValueError(f"Working hours seem unreasonably high: {v}")
        return v

    @validator("credits_qualified")
    def validate_credits(cls, v):
        """Validate credit calculations."""
        if v < 0:
            raise ValueError(f"Credits cannot be negative: {v}")
        if v > 30:
            raise ValueError(f"Credits exceed maximum allowed: {v}")
        return v

    @validator("requested_training_type")
    def validate_training_type_consistency(cls, v, values):
        """Validate training type consistency with degree relevance."""
        if "degree_relevance" in values:
            relevance = values["degree_relevance"]
            if relevance in ["high", "medium"] and v == "general":
                raise ValueError(
                    f"Inconsistent classification: degree relevance is '{relevance}' "
                    f"but requested training type is '{v}'. Should be 'professional'."
                )
            if relevance == "low" and v == "professional":
                raise ValueError(
                    f"Inconsistent classification: degree relevance is '{relevance}' "
                    f"but requested training type is '{v}'. Should be 'general'."
                )
        return v

    @validator("credits_qualified")
    def validate_credit_limits(cls, v, values):
        """Validate credit limits based on training type."""
        if "requested_training_type" in values:
            training_type = values["requested_training_type"]
            if training_type == "general" and v > 10:
                raise ValueError(f"General training credits exceed maximum: {v} > 10")
            if training_type == "professional" and v > 30:
                raise ValueError(
                    f"Professional training credits exceed maximum: {v} > 30"
                )
        return v

    @computed_field
    @property
    def hours_per_credit(self) -> Optional[float]:
        """Calculate hours per credit if both values are available."""
        if self.total_working_hours and self.credits_qualified:
            return self.total_working_hours / self.credits_qualified
        return None


class ValidationIssue(BaseModel):
    """Model for validation issues found during structural validation."""

    type: str = Field(
        ...,
        pattern="^(date_validation|timeline_consistency|employer_consistency|business_rule|data_type)$",
    )
    severity: str = Field(..., pattern="^(low|medium|high|critical)$")
    description: str
    field_affected: str
    suggestion: str
    position_index: Optional[int] = None  # For position-specific issues


class StructuralValidationResult(BaseModel):
    """Model for structural validation results."""

    validation_passed: bool
    issues_found: List[ValidationIssue] = Field(default_factory=list)
    extraction_valid: bool = True
    evaluation_valid: bool = True
    summary: str = ""


def validate_extraction_results(data: Dict[str, Any]) -> StructuralValidationResult:
    """
    Validate extraction results using Pydantic models.

    Args:
        data: Raw extraction results dictionary

    Returns:
        StructuralValidationResult with validation findings
    """
    issues = []

    try:
        # Validate with Pydantic model (now handles string dates directly)
        _ = ExtractionResults(**data)  # Validate but don't store

    except Exception as e:
        issues.append(
            ValidationIssue(
                type="data_type",
                severity="high",
                description=f"Validation error: {str(e)}",
                field_affected="extraction_results",
                suggestion="Review data structure and format",
            )
        )
        return StructuralValidationResult(
            validation_passed=False,
            issues_found=issues,
            extraction_valid=False,
            summary=f"Extraction validation failed: {str(e)}",
        )

    return StructuralValidationResult(
        validation_passed=len(issues) == 0,
        issues_found=issues,
        extraction_valid=True,
        summary="Extraction validation passed"
        if len(issues) == 0
        else f"Found {len(issues)} issues",
    )


def validate_evaluation_results(data: Dict[str, Any]) -> StructuralValidationResult:
    """
    Validate evaluation results using Pydantic models.

    Args:
        data: Raw evaluation results dictionary

    Returns:
        StructuralValidationResult with validation findings
    """
    issues = []

    try:
        # Validate with Pydantic model
        _ = EvaluationResults(**data)  # Validate but don't store

    except Exception as e:
        issues.append(
            ValidationIssue(
                type="business_rule",
                severity="high",
                description=f"Validation error: {str(e)}",
                field_affected="evaluation_results",
                suggestion="Review evaluation logic and business rules",
            )
        )
        return StructuralValidationResult(
            validation_passed=False,
            issues_found=issues,
            evaluation_valid=False,
            summary=f"Evaluation validation failed: {str(e)}",
        )

    return StructuralValidationResult(
        validation_passed=len(issues) == 0,
        issues_found=issues,
        evaluation_valid=True,
        summary="Evaluation validation passed"
        if len(issues) == 0
        else f"Found {len(issues)} issues",
    )
