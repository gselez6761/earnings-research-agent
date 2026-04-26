"""EDGAR identity validation.

The SEC requires all programmatic EDGAR access to identify the requester
via the User-Agent header in the format "Full Name email@example.com".
EdgarTools reads this from the EDGAR_IDENTITY environment variable.

This module validates the format at startup so the process fails fast
with a clear error rather than getting blocked by SEC rate-limiting
mid-run with an opaque HTTP 403.

Reference: https://www.sec.gov/os/accessing-edgar-data
"""
from __future__ import annotations

import re

_IDENTITY_PATTERN = re.compile(
    r"^[A-Za-z]+(?: [A-Za-z]+)+ [A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$"
)


def validate_edgar_identity(identity: str) -> None:
    """Raise ValueError if identity doesn't match 'Full Name email@domain' format.

    Args:
        identity: The value of the EDGAR_IDENTITY environment variable.

    Raises:
        ValueError: If the format is wrong. Message includes the required format.
    """
    if not identity or not _IDENTITY_PATTERN.match(identity.strip()):
        raise ValueError(
            f"EDGAR_IDENTITY must be 'First Last email@example.com' "
            f"(SEC fair-access requirement). Got: '{identity}'"
        )
