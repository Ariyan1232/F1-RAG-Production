"""
config.py

Loads and validates environment variables (API keys) once, at import time,
so failures happen with a clear message here rather than a cryptic
traceback three layers deep inside langchain_google_genai.

Note: Google is transitioning Gemini API keys from the old "AIza" format
(Standard keys) to a newer "AQ." format (Auth keys) through 2026. Both are
accepted here since either may be issued depending on when/how the key
was created.
"""

import os
from dotenv import load_dotenv

load_dotenv()


def _get_required_env(var_name: str, expected_prefixes: tuple[str, ...] | None = None) -> str:
    """Fetch an environment variable, failing loudly and clearly if it's
    missing or obviously the wrong kind of value."""

    value = os.getenv(var_name)

    if not value:
        raise EnvironmentError(
            f"\n\n"
            f"Missing environment variable: {var_name}\n"
            f"Add it to your .env file in the project root, e.g.:\n"
            f"    {var_name}=your_actual_key_here\n"
        )

    if expected_prefixes and not any(value.startswith(p) for p in expected_prefixes):
        prefixes_str = "' or '".join(expected_prefixes)
        raise EnvironmentError(
            f"\n\n"
            f"{var_name} doesn't look like a valid key.\n"
            f"Expected it to start with '{prefixes_str}', but got: "
            f"'{value[:10]}...'\n"
            f"Double check you copied the key from the right place "
            f"(Google AI Studio: https://aistudio.google.com/apikey).\n"
        )

    return value


# Gemini API keys may be either the older "AIza" (Standard) format
# or the newer "AQ." (Auth) format, depending on when they were issued.
GOOGLE_API_KEY = _get_required_env("GOOGLE_API_KEY", expected_prefixes=("AIza", "AQ."))