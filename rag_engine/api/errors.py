"""Stable error payloads."""

from __future__ import annotations

from typing import Any


def api_error(code: str, message: str, **extra: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {"code": code, "message": message}
    for key, value in extra.items():
        if value is not None:
            payload[key] = value
    return payload
