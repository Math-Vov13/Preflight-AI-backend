"""CORS allowlist parsing (BE-PR23).

Pins the env-var → list contract so a misconfigured deployment can't
silently fall back to credentialed wildcards (which browsers reject).
"""
from __future__ import annotations

import pytest

from server import _parse_cors_origins


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, []),
        ("", []),
        ("   ", []),
        ("https://app.preflight.dev", ["https://app.preflight.dev"]),
        (
            "http://localhost:3000,https://app.preflight.dev",
            ["http://localhost:3000", "https://app.preflight.dev"],
        ),
        # Tolerates the human-friendly cases: trailing comma, whitespace
        # around entries, repeated commas.
        (
            "  http://localhost:3000 , https://app.preflight.dev ,  ",
            ["http://localhost:3000", "https://app.preflight.dev"],
        ),
    ],
)
def test_parse_cors_origins(raw: str | None, expected: list[str]) -> None:
    assert _parse_cors_origins(raw) == expected
