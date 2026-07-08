"""Shared validation helpers for planner ``params:`` blocks.

Planner param dataclasses read a fixed key set with defaults; without these
checks a typo (``zigzag_spaceing``) or a nonsensical value (negative velocity)
is silently swallowed. Mirrors the fail-loud policy enforced at the config
document level (``runtime_config._unknown_keys``) and on sensor parameters
(``BaseSensor.validate_parameters``).
"""

from __future__ import annotations

from typing import Iterable, Mapping

from src.robot_config import ConfigError


def reject_unknown_keys(
    params: Mapping[str, object], allowed: Iterable[str], ctx: str
) -> None:
    """Raise ``ConfigError`` if ``params`` holds keys outside ``allowed``."""
    unknown = sorted(set(params) - set(allowed))
    if unknown:
        raise ConfigError(
            f"{ctx}: unknown key(s): {', '.join(unknown)}. "
            f"Allowed: {', '.join(sorted(allowed))}."
        )


def require_positive(
    params: Mapping[str, object], keys: Iterable[str], ctx: str
) -> None:
    """Raise ``ConfigError`` if any present ``keys`` is non-numeric or <= 0."""
    for key in keys:
        if key not in params:
            continue
        value = params[key]
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
            raise ConfigError(
                f"{ctx}: '{key}' must be a positive number (got {value!r})."
            )
