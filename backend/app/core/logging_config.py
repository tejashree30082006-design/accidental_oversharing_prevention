"""
logging_config.py — Secure logging setup for the hotel document analysis pipeline.

Rules enforced here:
  - Sensitive fields (text, card number, phone, email, etc.) must NEVER appear
    in log output.
  - Use Finding.to_safe_dict() before logging any Finding object.
  - Log level DEBUG is disabled in production (see LOG_LEVEL env var).

Usage:
    from app.core.logging_config import configure_logging
    configure_logging()  # call once at application startup
"""

from __future__ import annotations

import logging
import os
import sys

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"


class SensitiveDataFilter(logging.Filter):
    """Drop log records that accidentally include sensitive field markers.

    This is a last-resort guard. The code should never log raw PII in the
    first place, but this filter acts as a safety net.
    """

    _FORBIDDEN_PATTERNS = [
        # Patterns that should never appear verbatim in logs
        "text=",         # direct Finding.text attribute
        "card_no",
        "passport_no",
        "email=",
    ]

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage().lower()
        for pat in self._FORBIDDEN_PATTERNS:
            if pat.lower() in msg:
                # Replace the sensitive content rather than drop the record
                record.msg = (
                    "[SENSITIVE DATA FILTERED] Original message contained "
                    f"forbidden pattern: {pat!r}"
                )
                record.args = ()
                break
        return True


def configure_logging() -> None:
    """Configure root logger with secure defaults."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    handler.addFilter(SensitiveDataFilter())

    root = logging.getLogger()
    root.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    if not root.handlers:
        root.addHandler(handler)
