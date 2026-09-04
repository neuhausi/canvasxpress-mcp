"""
Figure validation / lint for the canvasxpress-mcp server (agent-native #9b).

Runs a formal JSON Schema check (the published cxfigure schema) AND the existing
markdown-parsed parameter validation, returning structured, fixable results so an agent can
correct a spec before rendering. Pure functions; no server state.
"""

import json
from pathlib import Path

from jsonschema import Draft202012Validator

import cx_knowledge

_SCHEMA_PATH = Path(__file__).parent.parent / "data" / "schema" / "cxfigure-1.0.schema.json"

# Load + compile once at import; a bad/missing schema degrades to param-only validation
# rather than crashing the server.
try:
    _SCHEMA = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    _VALIDATOR = Draft202012Validator(_SCHEMA)
except Exception:  # noqa: BLE001 - never let schema loading take down the server
    _SCHEMA = None
    _VALIDATOR = None


def lint_figure(figure: dict) -> dict:
    """Validate a CanvasXpress figure.

    Args:
        figure: {"data": ..., "config": ...} (schemaVersion optional).

    Returns:
        {
          "ok": bool,                # True when there are no hard schema errors
          "schema_errors": [ {path, message} ],   # hard: structural contract violations
          "param_warnings": ...,     # soft: value-level checks from cx_knowledge
        }
    """
    figure = figure or {}
    schema_errors = []
    if _VALIDATOR is not None:
        for err in sorted(_VALIDATOR.iter_errors(figure), key=lambda e: list(e.absolute_path)):
            schema_errors.append(
                {"path": list(err.absolute_path), "message": err.message}
            )

    config = figure.get("config", {}) if isinstance(figure, dict) else {}
    try:
        param_warnings = cx_knowledge.validate_param_values(config)
    except Exception as exc:  # noqa: BLE001 - param validation is best-effort
        param_warnings = {"error": str(exc)}

    return {
        "ok": len(schema_errors) == 0,
        "schema_errors": schema_errors,
        "param_warnings": param_warnings,
        "schema_available": _VALIDATOR is not None,
    }
