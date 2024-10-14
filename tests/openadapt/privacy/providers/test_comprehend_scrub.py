"""Module to test ComprehendScrubbingProvider."""

import re

from botocore.exceptions import NoRegionError
import pytest

from openadapt.privacy.providers.aws_comprehend import ComprehendScrubbingProvider

scrub = ComprehendScrubbingProvider()

try:
    scrub.scrub_text("hello James Mark")
except NoRegionError:
    msg = (
        "AWS Config Files not setup correctly. Please see "
        "https://boto3.amazonaws.com/v1/documentation/api/latest/guide/quickstart.html#configuration"  # noqa: E501
    )
    pytestmark = pytest.mark.skip(reason=msg)


def test_empty_string() -> None:
    """Test empty string input for scrub function returns empty string."""
    text = ""
    expected_output = ""
    assert scrub.scrub_text(text) == expected_output


def test_no_scrub_string() -> None:
    """Test that the same string is returned."""
    text = "This string doesn't have anything to scrub."
    expected_output = "This string doesn't have anything to scrub."
    assert scrub.scrub_text(text) == expected_output


def test_scrub_email() -> None:
    """Test that the email address is scrubbed."""
    assert (
        scrub.scrub_text("My email is john.doe@example.com.") == "My email is <EMAIL>."
    )


def test_scrub_phone_number() -> None:
    """Test that the phone number is scrubbed."""
    assert (
        scrub.scrub_text("My phone number is 123-456-7890.")
        == "My phone number is <PHONE>."
    )


def test_scrub_credit_card() -> None:
    """Test that the credit card number is scrubbed."""
    assert (
        scrub.scrub_text("My credit card number is 4234-5678-9012-3456 and ")
    ) == "My credit card number is <CREDIT_DEBIT_NUMBER> and "


def test_scrub_date_of_birth() -> None:
    """Test that the date of birth is scrubbed."""
    assert (
        scrub.scrub_text("My date of birth is 01/01/2000.")
        == "My date of birth is <DATE_TIME>."
    )


def test_scrub_address() -> None:
    """Test that the address is scrubbed."""
    scrubbed_text = scrub.scrub_text("My address is 123 Main St, Toronto, On, CAN.")
    # Patterns to match the different acceptable scrubbed results
    acceptable_patterns = [
        "My address is <ADDRESS>.",
        "My address is 123 Main St, <LOCATION>, On, <LOCATION>.",
        "My address is 123 Main St, <LOCATION>, <LOCATION>, <LOCATION>.",
    ]

    # Check if scrubbed_text matches any of the acceptable patterns
    match_found = any(
        re.match(pattern, scrubbed_text) for pattern in acceptable_patterns
    )

    assert (
        match_found
    ), f"Scrubbed text '{scrubbed_text}' did not match any expected patterns."


def test_scrub_ssn() -> None:
    """Test that the social security number is scrubbed."""
    # Test scrubbing of social security number
    assert (
        scrub.scrub_text("My social security number is 923-45-6789")
        == "My social security number is <SSN>"
    )


def test_scrub_dl() -> None:
    """Test that the driver's license number is scrubbed."""
    assert (
        scrub.scrub_text("My driver's license number is A123-456-789-012")
        == "My driver's license number is <DRIVER_ID>"
    )


def test_scrub_passport() -> None:
    """Test that the passport number is scrubbed."""
    assert (
        scrub.scrub_text("My passport number is A1234567.")
        == "My passport number is <PASSPORT_NUMBER>."
    )


def test_scrub_national_id() -> None:
    """Test that the national ID number is scrubbed."""
    scrubbed_text = scrub.scrub_text("My national ID number is 1234567890123.")
    expected_outcomes = [
        "My national ID number is <PHONE>.",
        "My national ID number is <DATA>.",
        "My national ID number is <SSN>.",  # Adding <SSN> to expected outcomes
    ]
    assert (
        scrubbed_text in expected_outcomes
    ), f"Scrubbed text '{scrubbed_text}' did not match any expected outcomes."


def test_scrub_routing_number() -> None:
    """Test that the bank routing number is scrubbed."""
    assert (
        scrub.scrub_text("My bank routing number is 123456789.")
        == "My bank routing number is <BANK_ROUTING>."
    )


def test_scrub_bank_account() -> None:
    """Test that the bank account number is scrubbed."""
    assert (
        scrub.scrub_text("My bank account number is 635526789012.")
        == "My bank account number is <BANK_ACCOUNT_NUMBER>."
    )


def test_scrub_all_together() -> None:
    """Test that all PII/PHI types are scrubbed."""
    text_with_pii_phi = (
        "John Smith's email is johnsmith@example.com and"
        " his phone number is 555-123-4567."
        "His credit card number is 4831-5538-2996-5651 and"
        " his social security number is 923-45-6789."
        " He was born on 01/01/1980."
    )
    assert (
        scrub.scrub_text(text_with_pii_phi)
        == "<NAME>'s email is <EMAIL> and"
        " his phone number is <PHONE>"
        " credit card number is <CREDIT_DEBIT_NUMBER> and"
        " his social security number is <SSN>."
        " He was born on <DATE_TIME>."
    )


def test_scrub_dict() -> None:
    """Test that the scrub_dict function works."""
    text_with_pii_phi = {"title": "hi my name is Bob Smith."}

    expected_output = {"title": "hi my name is <NAME>."}

    assert scrub.scrub_dict(text_with_pii_phi) == expected_output
