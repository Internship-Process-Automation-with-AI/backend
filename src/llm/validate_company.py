"""
Simple company validation module using pattern analysis and free APIs.
Detects suspicious company names and provides basic validation status.
"""

import logging
import re
import time
from typing import Dict, List, Optional

import dns.resolver
import phonenumbers
import requests

from .models import CompanyValidationResult

logger = logging.getLogger(__name__)


class CompanyValidator:
    """Simple company validator that marks companies as suspicious or not."""

    def __init__(self):
        """Initialize the company validator."""
        self._initialize_patterns()

    def _initialize_patterns(self):
        """Initialize suspicious pattern detection rules."""

        # Simple suspicious patterns
        self.suspicious_patterns = [
            r"test",
            r"sample",
            r"example",
            r"fake",
            r"dummy",
            r"placeholder",
            r"mock",
            r"demo",
            r"trial",
            r"temp",
            r"not.*real",
            r"real.*not",
            r"super.*mega",
            r"mega.*super",
            r"ultra.*company",
            r"company.*ultra",
            r"amazing",
            r"fantastic",
            r"incredible",
            r"awesome",
            r"wonderful",
            r"brilliant",
            r"excellent",
            r"perfect",
            # Finnish suspicious patterns
            r"testi",
            r"esimerkki",
            r"vale",
            r"näyte",
            r"harjoitus",
            r"kokeilu",
            # Unrealistic patterns
            r"^\d{5,}.*company",
            r"^[a-z]{1,3}\d{3,}.*corp",
            r"^[a-z]{10,}.*inc",
            r"^[a-z]{1,2}[a-z]{1,2}[a-z]{1,2}.*ltd",
        ]

        # Compile patterns for efficiency
        self.compiled_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.suspicious_patterns
        ]

    def validate_company(
        self,
        company_name: str,
        address: Optional[str] = None,
        business_id: Optional[str] = None,
        contact_info: Optional[Dict[str, str]] = None,
    ) -> CompanyValidationResult:
        """
        Simple company validation - just marks as suspicious or not.

        Args:
            company_name: The company name to validate
            address: Optional company address for validation
            business_id: Optional business ID for validation
            contact_info: Optional dict with 'phone' and 'email' keys

        Returns:
            CompanyValidationResult with simple validation status
        """
        if not company_name or not isinstance(company_name, str):
            return CompanyValidationResult(
                company_name=company_name or "",
                is_suspicious=True,
                risk_level="very_high",
                confidence_score=0.0,
                suspicious_patterns=["Invalid company name"],
                validation_notes="Company name is invalid or empty",
                requires_manual_review=True,
            )

        # 1. Check if name is suspicious
        is_suspicious = self._is_name_suspicious(company_name)
        suspicious_patterns = self._get_suspicious_patterns(company_name)

        # 2. Address validation
        address_status = "Address not confirmed"
        if address:
            if self._validate_address_openstreetmap(address).get("is_valid"):
                address_status = "Address confirmed"

        # 3. Business ID validation
        business_id_status = "Business ID not provided"
        if business_id:
            if self._validate_business_id(business_id).get("is_valid"):
                business_id_status = "Business ID format valid"
            else:
                business_id_status = "Business ID format invalid"

        # 4. Contact validation
        contact_status = "Contact info not provided"
        if contact_info:
            phone = contact_info.get("phone")
            email = contact_info.get("email")

            phone_valid = phone and self._validate_phone_number(phone).get("is_valid")
            email_valid = email and self._validate_email_address(email).get("is_valid")

            if phone_valid or email_valid:
                contact_status = "Phone number or email is verified"
            else:
                contact_status = "Contact info not verified"

        # Generate simple validation notes
        validation_notes = self._generate_simple_notes(
            is_suspicious,
            suspicious_patterns,
            address_status,
            business_id_status,
            contact_status,
        )

        # Determine risk level (simple mapping)
        risk_level = "very_high" if is_suspicious else "very_low"
        confidence_score = 0.0 if is_suspicious else 1.0

        return CompanyValidationResult(
            company_name=company_name,
            is_suspicious=is_suspicious,
            risk_level=risk_level,
            confidence_score=confidence_score,
            suspicious_patterns=suspicious_patterns,
            validation_notes=validation_notes,
            requires_manual_review=is_suspicious,
            address_validation={"status": address_status},
            business_id_validation={"status": business_id_status},
            contact_validation={"status": contact_status},
        )

    def validate_company_info(self, extracted_info: Dict[str, any]) -> Dict[str, any]:
        """
        Validate company information from extracted employment data.

        Args:
            extracted_info: Dictionary containing extracted employment information

        Returns:
            Validation result dictionary with company validation details
        """
        if not extracted_info or not isinstance(extracted_info, dict):
            return self._create_error_response("Invalid extracted information format")

        positions = extracted_info.get("positions", [])
        if not positions:
            return self._create_success_response("No company information to validate")

        # Extract all company names
        company_names = []
        valid_positions = []

        for i, position in enumerate(positions):
            if isinstance(position, dict) and position.get("employer"):
                company_names.append(position["employer"])
                valid_positions.append((i, position))

        if not company_names:
            return self._create_success_response("No valid company names found")

        # Validate each company
        company_validation_results = []
        issues_found = []
        suspicious_companies = 0

        for i, (position_idx, position) in enumerate(valid_positions):
            company_name = position["employer"]
            validation_result = self.validate_company(company_name)

            company_validation_results.append(
                {
                    "position_index": position_idx,
                    "company_name": company_name,
                    "validation_result": validation_result,
                }
            )

            # Track suspicious companies
            if validation_result.is_suspicious:
                suspicious_companies += 1
                issues_found.append(
                    {
                        "type": "company_validation_error",
                        "severity": "high",
                        "description": f"Suspicious company detected: {company_name}",
                        "field_affected": "company_validation",
                        "suggestion": f"Review company '{company_name}' for legitimacy",
                    }
                )

        # Determine if validation passed
        validation_passed = suspicious_companies == 0

        # Generate summary
        if suspicious_companies == 0:
            summary = "All companies passed validation"
        elif suspicious_companies == 1:
            summary = "1 suspicious company detected requiring review"
        else:
            summary = (
                f"{suspicious_companies} suspicious companies detected requiring review"
            )

        return {
            "validation_passed": validation_passed,
            "summary": summary,
            "issues_found": issues_found,
            "company_validation": {
                "company_name_legitimate": validation_passed,
                "risk_level": "very_high" if suspicious_companies > 0 else "very_low",
                "confidence_score": 0.0 if suspicious_companies > 0 else 1.0,
                "suspicious_patterns_detected": [
                    pattern
                    for result in company_validation_results
                    if result["validation_result"].suspicious_patterns
                    for pattern in result["validation_result"].suspicious_patterns
                ],
                "requires_manual_review": suspicious_companies > 0,
                "validation_notes": f"Validated {len(company_validation_results)} companies. {suspicious_companies} require manual review.",
                "detailed_results": company_validation_results,
            },
        }

    def _create_error_response(self, error_message: str) -> Dict[str, any]:
        """Create standardized error response."""
        return {
            "validation_passed": False,
            "summary": error_message,
            "issues_found": [
                {
                    "type": "company_validation_error",
                    "severity": "critical",
                    "description": error_message,
                    "field_affected": "company_validation",
                    "suggestion": "Check extraction results format",
                }
            ],
            "company_validation": {
                "company_name_legitimate": False,
                "risk_level": "very_high",
                "confidence_score": 0.0,
                "suspicious_patterns_detected": ["Invalid data format"],
                "requires_manual_review": True,
                "validation_notes": f"Cannot validate company information: {error_message}",
            },
        }

    def _create_success_response(self, message: str) -> Dict[str, any]:
        """Create standardized success response."""
        return {
            "validation_passed": True,
            "summary": message,
            "issues_found": [],
            "company_validation": {
                "company_name_legitimate": True,
                "risk_level": "very_low",
                "confidence_score": 1.0,
                "suspicious_patterns_detected": [],
                "requires_manual_review": False,
                "validation_notes": message,
            },
        }

    def _is_name_suspicious(self, company_name: str) -> bool:
        """Check if company name contains suspicious patterns."""
        normalized_name = self._normalize_company_name(company_name)

        for pattern in self.compiled_patterns:
            if pattern.search(normalized_name):
                return True

        return False

    def _get_suspicious_patterns(self, company_name: str) -> List[str]:
        """Get list of suspicious patterns found in company name."""
        normalized_name = self._normalize_company_name(company_name)
        found_patterns = []

        for pattern in self.compiled_patterns:
            if pattern.search(normalized_name):
                # Extract the actual matched text
                matches = pattern.findall(normalized_name)
                found_patterns.extend(matches)

        return found_patterns

    def _normalize_company_name(self, company_name: str) -> str:
        """Normalize company name for pattern matching."""
        # Convert to lowercase
        normalized = company_name.lower().strip()

        # Remove common punctuation
        normalized = re.sub(r'[.,;:!?()[\]{}"\']', " ", normalized)

        # Remove extra whitespace
        normalized = re.sub(r"\s+", " ", normalized)

        # Remove common business suffixes for pattern matching
        normalized = re.sub(
            r"\b(inc|ltd|corp|corporation|oy|ab|gmbh|llc|plc|sa|nv|bv|as)\b",
            "",
            normalized,
        )

        return normalized.strip()

    def _validate_address_openstreetmap(self, address: str) -> Dict[str, any]:
        """
        Validate address using OpenStreetMap Nominatim API (free).

        Args:
            address: Address string to validate

        Returns:
            Validation result dictionary
        """
        # Rate limiting: 1 request per second
        time.sleep(1)

        try:
            url = "https://nominatim.openstreetmap.org/search"
            params = {"q": address, "format": "json", "limit": 1, "addressdetails": 1}

            response = requests.get(url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()

                if data:
                    # Address found
                    return {"is_valid": True, "validation_method": "openstreetmap"}
                else:
                    # Address not found
                    return {"is_valid": False, "validation_method": "openstreetmap"}
            else:
                return {"is_valid": False, "validation_method": "openstreetmap"}

        except Exception:
            return {"is_valid": False, "validation_method": "openstreetmap"}

    def _validate_business_id(self, business_id: str) -> Dict[str, any]:
        """
        Validate business ID format.

        Args:
            business_id: Business ID to validate

        Returns:
            Validation result dictionary
        """
        # Finnish Y-tunnus validation (8 digits + hyphen + 1 digit)
        finnish_pattern = r"^\d{7}-\d$"

        # US EIN validation (XX-XXXXXXX format)
        us_ein_pattern = r"^\d{2}-\d{7}$"

        # UK Companies House number (8 digits)
        uk_pattern = r"^\d{8}$"

        if re.match(finnish_pattern, business_id):
            return {
                "is_valid": True,
                "format": "finnish_ytunnus",
                "country": "FI",
                "validation_method": "format_check",
            }
        elif re.match(us_ein_pattern, business_id):
            return {
                "is_valid": True,
                "format": "us_ein",
                "country": "US",
                "validation_method": "format_check",
            }
        elif re.match(uk_pattern, business_id):
            return {
                "is_valid": True,
                "format": "uk_companies_house",
                "country": "UK",
                "validation_method": "format_check",
            }
        else:
            return {"is_valid": False, "validation_method": "format_check"}

    def _validate_phone_number(self, phone: str) -> Dict[str, any]:
        """Validate phone number format."""

        # Remove all non-digit characters except + and -
        cleaned = re.sub(r"[^\d+\-\(\)\s]", "", phone)

        try:
            # Parse with phonenumbers library
            parsed = phonenumbers.parse(cleaned, None)

            # Check if valid format
            is_valid = phonenumbers.is_valid_number(parsed)

            return {
                "is_valid": is_valid,
                "validation_method": "format_check",
                "raw_number": phone,
            }

        except Exception:
            return {
                "is_valid": False,
                "validation_method": "format_check",
                "raw_number": phone,
            }

    def _validate_email_address(self, email: str) -> Dict[str, any]:
        """Validate email format and domain."""

        # Basic email regex pattern
        email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"

        if not re.match(email_pattern, email):
            return {
                "is_valid": False,
                "validation_method": "format_check",
                "raw_email": email,
            }

        # Split email into local and domain parts
        local_part, domain = email.split("@")

        # Check lengths
        if len(local_part) > 64 or len(domain) > 253:
            return {
                "is_valid": False,
                "validation_method": "format_check",
                "raw_email": email,
            }

        # Domain validation
        domain_result = self._validate_email_domain(domain)

        return {
            "is_valid": domain_result.get("is_valid", False),
            "validation_method": "format_and_dns_check",
            "raw_email": email,
        }

    def _validate_email_domain(self, domain: str) -> Dict[str, any]:
        """Validate email domain exists and has mail servers."""

        try:
            # Check if domain has MX records (mail servers)
            dns.resolver.resolve(domain, "MX")

            # Check if domain has A records (web servers)
            dns.resolver.resolve(domain, "A")

            return {"is_valid": True, "validation_method": "dns_check"}

        except Exception:
            return {"is_valid": False, "validation_method": "dns_check"}

    def _generate_simple_notes(
        self,
        is_suspicious: bool,
        suspicious_patterns: List[str],
        address_status: str,
        business_id_status: str,
        contact_status: str,
    ) -> str:
        """Generate simple validation notes."""
        notes_parts = []

        if is_suspicious:
            patterns = ", ".join(suspicious_patterns)
            notes_parts.append(
                f"Company name is suspicious due to patterns: {patterns}"
            )
        else:
            notes_parts.append("Company name appears legitimate")

        notes_parts.append(f"Address: {address_status}")
        notes_parts.append(f"Business ID: {business_id_status}")
        notes_parts.append(f"Contact: {contact_status}")

        return ". ".join(notes_parts)


# Example usage and testing
if __name__ == "__main__":
    # Test the validator
    validator = CompanyValidator()

    test_companies = [
        "Nokia Oyj",  # Should be legitimate
        "Test Company Inc",  # Should be suspicious
        "Sample Business Ltd",  # Should be suspicious
        "Fake Corp",  # Should be suspicious
        "Microsoft Corporation",  # Should be legitimate
        "12345 Company",  # Should be suspicious
        "Super Mega Ultra Corp",  # Should be suspicious
        "Nordic Software Solutions Oy",  # Should be legitimate
        "Testi Yritys Ab",  # Should be suspicious (Finnish)
        "Example Company GmbH",  # Should be suspicious
    ]

    print("Simple Company Validation Test Results:")
    print("=" * 50)

    # Test individual validation
    for company in test_companies:
        result = validator.validate_company(company)
        status = "✅ LEGITIMATE" if not result.is_suspicious else "❌ SUSPICIOUS"
        print(f"{company:<30} | {status:<15} | {result.validation_notes}")

    print("\n" + "=" * 50)
    print("Available validation methods:")
    print("1. Pattern analysis (always available)")
    print("2. Address validation (OpenStreetMap - free)")
    print("3. Business ID format validation (always available)")
    print("4. Phone and email validation (always available)")
    print("5. Simple suspicious/not suspicious classification")
