"""Bundled sample datasets and table normalisation for the canvasxpress-mcp server.

Two jobs, both feeding ``suggest_canvasxpress_dashboard``:

* ``load_dataset(name)`` — the CSVs under ``data/datasets`` (penguins, iris, mtcars) as a
  CanvasXpress data object, so a request like "use the penguins data" needs no fetching.
* ``table_to_data(rows)`` — a headers-plus-rows array (first row = headers, the shape that
  ``generate_canvasxpress_config`` and ``compose_canvasxpress_dashboard`` use) as the same
  data object ``{y: {vars, smps, data}, x: {...}}``.

Numeric columns become ``y`` variables (missing → ``None``); factor columns become ``x``
annotations (missing → ``"NA"``). A column is numeric when more than 90% of its non-missing
values parse as numbers. ``year`` is always a factor. Pure functions; no server state.
"""

from __future__ import annotations

import csv
import os
import re
from typing import Any, Optional

_DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "datasets")

_NUMERIC_SHARE = 0.9
_MISSING_RE = re.compile(r"^(na|n/a|nan|null|none|-)?$", re.I)
_NUMBER_RE = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")
_ALWAYS_FACTOR_RE = re.compile(r"^year$", re.I)

# name -> file stem, columns that are categories though they hold numbers, and the column
# that names each row (it becomes the sample id, not a variable).
_DATASETS: dict = {
    "penguins": {"file": "penguins", "factors": [], "id": None,
                 "aliases": ["palmer penguins", "palmerpenguins", "palmer-penguins", "palmer_penguins",
                             "penguin"]},
    "iris": {"file": "iris", "factors": [], "id": None, "aliases": ["fisher iris", "iris flowers"]},
    "mtcars": {"file": "mtcars", "factors": ["cyl", "vs", "am", "gear", "carb"], "id": "model",
               "aliases": ["motor trend cars", "mtcar", "mt cars"]},
}


def available_datasets() -> list:
    return sorted(_DATASETS)


def _canonical_name(name: str) -> Optional[str]:
    key = re.sub(r"\s+", " ", str(name or "").strip().lower())
    key = re.sub(r"\.csv$", "", key)
    if key in _DATASETS:
        return key
    for canonical, spec in _DATASETS.items():
        if key in spec["aliases"]:
            return canonical
    return None


def _is_missing(value: Any) -> bool:
    return value is None or bool(_MISSING_RE.match(str(value).strip()))


def _to_number(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value) if value == value else None
    s = str(value).strip()
    return float(s) if _NUMBER_RE.match(s) else None


def _tidy(number: float) -> Any:
    return int(number) if number.is_integer() else number


def _is_numeric_column(name: str, values: list, force_factor: bool) -> bool:
    if force_factor or _ALWAYS_FACTOR_RE.match(str(name).strip()):
        return False
    present = [v for v in values if not _is_missing(v)]
    if not present:
        return False
    return sum(1 for v in present if _to_number(v) is not None) / len(present) > _NUMERIC_SHARE


def _factor_text(value: Any) -> str:
    if _is_missing(value):
        return "NA"
    number = _to_number(value)
    return str(_tidy(number)) if number is not None and not isinstance(value, str) else str(value).strip()


def table_to_data(rows: list, factors: Optional[list] = None, id_column: Optional[str] = None) -> dict:
    """A headers-plus-rows table as a CanvasXpress data object.

    Args:
        rows: ``[[header, ...], [cell, ...], ...]`` — the first row is the headers.
        factors: column names to treat as factors even though they hold numbers.
        id_column: a column whose values name the rows (it is not a variable or annotation).

    Raises:
        ValueError: the table has no header row, no data rows, or no usable columns.
    """
    if not isinstance(rows, list) or len(rows) < 2 or not all(isinstance(r, (list, tuple)) for r in rows):
        raise ValueError("data must be an object {y, x} or a headers+rows array with at least one data row")
    headers = [str(h).strip() for h in rows[0]]
    if len(set(headers)) != len(headers) or "" in headers:
        raise ValueError("the header row must have unique, non-empty column names")
    body = [list(r) + [None] * (len(headers) - len(r)) for r in rows[1:]]
    forced = {str(f) for f in (factors or [])}
    id_index = headers.index(id_column) if id_column in headers else -1

    variables: list = []
    columns: list = []
    annotations: dict = {}
    for c, name in enumerate(headers):
        if c == id_index:
            continue
        cells = [row[c] for row in body]
        if _is_numeric_column(name, cells, name in forced):
            variables.append(name)
            columns.append([None if _is_missing(v) or _to_number(v) is None else _tidy(_to_number(v))
                            for v in cells])
        else:
            annotations[name] = [_factor_text(v) for v in cells]
    if not variables:
        raise ValueError("the table has no numeric columns")
    if id_index >= 0:
        samples = [str(row[id_index]).strip() for row in body]
    else:
        samples = [str(i + 1) for i in range(len(body))]
    data: dict = {"y": {"vars": variables, "smps": samples, "data": columns}}
    if annotations:
        data["x"] = annotations
    return data


def load_dataset(name: str) -> dict:
    """A bundled dataset as a data object. Raises ``KeyError`` with the available names."""
    canonical = _canonical_name(name)
    if canonical is None:
        raise KeyError("Unknown dataset %r. Available datasets: %s." % (name, ", ".join(available_datasets())))
    spec = _DATASETS[canonical]
    with open(os.path.join(_DATASET_DIR, spec["file"] + ".csv"), newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    return table_to_data(rows, factors=spec["factors"], id_column=spec["id"])
