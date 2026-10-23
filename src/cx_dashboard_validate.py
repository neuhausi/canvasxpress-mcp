"""Dashboard-spec validation (vendored).

VENDORED from canvasxpress-dashboards ``server/src/cxd_server/validate_spec.py``
(spec format 1.3) — a dependency-free Python port of the browser-side
``src/validateSpec.js``. The canonical contract for both is
``schema/dashboard.schema.json`` in that repo. When canvasxpress-dashboards
releases a new spec format, refresh this file from that repo.
(``validate_bindings`` from the original is omitted — it needs that server's
dataset catalogue.) Pure functions; no server state.
"""

from __future__ import annotations

import math
import re
from typing import Any, Optional

_FIT = ("contain", "cover", "fill", "none", "scale-down")
_STYLE = ("auto", "dropdown", "radio", "buttons", "search", "slider")
_ALIGN = ("left", "center", "right")
_VALIGN = ("top", "middle", "bottom")
_PARAM_TYPES = ("string", "number", "boolean")
_DATA_KINDS = ("inline", "connector", "dataset", "join", "function", "live")
_FUNCTION_LANGUAGES = ("python", "r")
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_JOIN_TYPES = ("inner", "left", "right", "outer")
_AXES = ("smps", "vars")
_MARKING_MODES = ("focus", "highlight", "ghost", "filter")
_FIELD_KINDS = ("values", "range", "search")
_CALC_TARGETS = ("variable", "sampleAnnotation", "variableAnnotation")
_BIN_METHODS = ("equalWidth", "quantile", "percentile", "custom")
# The dashboard spec format version this validator reads — keep in sync with
# DASHBOARD_SCHEMA_VERSION in canvasxpress-dashboards src/spec.js.
DASHBOARD_SCHEMA_VERSION = "1.3"
_VERSION_RE = re.compile(r"^(\d+)\.(\d+)$")


def _is_int(value: Any) -> bool:
    """Integer check that excludes bool (``True`` is an int in Python)."""
    return isinstance(value, int) and not isinstance(value, bool)


def _is_obj(value: Any) -> bool:
    """True for a JSON object (dict), not a list and not None."""
    return isinstance(value, dict)


def _is_str(value: Any) -> bool:
    return isinstance(value, str)


def validate_spec(spec: Any) -> dict:
    """Validate a dashboard spec's structure and internal references.

    Mirrors ``validateSpec.js`` error-for-error so client and server agree.

    :param spec: the dashboard spec to check.
    :returns: ``{"valid": bool, "errors": [str, ...]}``.
    """
    errors: list = []

    if not _is_obj(spec):
        return {"valid": False, "errors": ["spec must be an object"]}

    if not _is_str(spec.get("id")) or not spec.get("id"):
        errors.append("spec.id is required and must be a non-empty string")

    version = spec.get("version")
    if version is not None and not (_is_int(version) and version >= 1):
        errors.append("spec.version must be an integer >= 1")

    # Format version (schemaVersion "MAJOR.MINOR"; absent = the legacy 1.0).
    warnings: list = []
    status, found = spec_compatibility(spec)
    if spec.get("$schema") is not None and not _is_str(spec.get("$schema")):
        errors.append("spec.$schema must be a URL string")
    if status == "invalid":
        errors.append('spec.schemaVersion must be "MAJOR.MINOR" (e.g. "%s")'
                      % DASHBOARD_SCHEMA_VERSION)
    elif status == "newer-major":
        errors.append("spec.schemaVersion %s needs a newer canvasxpress-dashboards "
                      "(this one reads %s.x)" % (found, DASHBOARD_SCHEMA_VERSION.split(".")[0]))

    panels = spec.get("panels")
    data = spec.get("data")
    params = spec.get("params")

    _check_layout(spec, panels, errors)
    _check_panels(panels, data, params, errors)
    _check_data(data, params, errors, warnings if status == "newer-minor" else None)
    _check_relationships(spec.get("relationships"), data, errors)
    if spec.get("markingMode") is not None and spec.get("markingMode") not in _MARKING_MODES:
        errors.append("spec.markingMode must be one of: " + ", ".join(_MARKING_MODES))
    _check_insights(spec.get("insights"), data, errors)
    _check_filter_schemes(spec.get("filterSchemes"), data, errors)
    _check_params(params, errors)
    _check_controls(spec.get("controls"), data, errors)

    result = {"valid": not errors, "errors": errors}
    if warnings:
        result["warnings"] = warnings
    return result


def spec_compatibility(spec: dict):
    """How a spec's format version relates to this validator.

    Mirrors ``specCompatibility`` in src/spec.js.

    :returns: ``(status, version)`` — status is ``current``, ``older``,
        ``newer-minor``, ``newer-major`` or ``invalid``.
    """
    raw = spec.get("schemaVersion")
    if raw is None:
        major, minor = 1, 0
    else:
        match = _VERSION_RE.fullmatch(raw) if _is_str(raw) else None
        if not match:
            return "invalid", str(raw)
        major, minor = int(match.group(1)), int(match.group(2))
    cur_major, cur_minor = (int(p) for p in DASHBOARD_SCHEMA_VERSION.split("."))
    version = "%d.%d" % (major, minor)
    if major > cur_major:
        return "newer-major", version
    if major < cur_major or minor < cur_minor:
        return "older", version
    if minor > cur_minor:
        return "newer-minor", version
    return "current", version


def _check_layout(spec: dict, panels: Any, errors: list) -> None:
    """Validate spec.layout and that every item references a real panel."""
    layout = spec.get("layout")
    if not _is_obj(layout):
        errors.append("spec.layout is required and must be an object")
        return

    items = layout.get("items")
    if not isinstance(items, list):
        errors.append("spec.layout.items is required and must be an array")
    else:
        for i, item in enumerate(items):
            at = "spec.layout.items[%d]" % i
            if not _is_obj(item):
                errors.append(at + " must be an object")
                continue
            panel = item.get("panel")
            if not _is_str(panel):
                errors.append(at + ".panel is required and must be a string")
            elif not _is_obj(panels) or panel not in panels:
                errors.append(at + '.panel "%s" has no matching entry in spec.panels' % panel)
            for key in ("x", "y", "w", "h"):
                if not _is_int(item.get(key)):
                    errors.append(at + "." + key + " is required and must be an integer")
            for key, floor in (("w", 1), ("h", 1), ("x", 0), ("y", 0)):
                value = item.get(key)
                if _is_int(value) and value < floor:
                    errors.append(at + ".%s must be >= %d" % (key, floor))

    cols = layout.get("cols")
    if cols is not None and not (_is_int(cols) and cols >= 1):
        errors.append("spec.layout.cols must be an integer >= 1")

    # Narrow-screen stacking breakpoint: a width in px, or false to disable.
    stack = layout.get("stackBelow")
    if stack is not None and stack is not False and not (
            isinstance(stack, (int, float)) and not isinstance(stack, bool)
            and math.isfinite(stack) and stack >= 0):
        errors.append("spec.layout.stackBelow must be a width in px (a number >= 0) or false")


def _check_panels(panels: Any, data: Any, params: Any, errors: list) -> None:
    """Validate every panel: data requirement, control wiring, references."""
    if not _is_obj(panels):
        errors.append("spec.panels is required and must be an object map")
        return

    for key, panel in panels.items():
        at = 'spec.panels["%s"]' % key
        if not _is_obj(panel):
            errors.append(at + " must be an object")
            continue

        ptype = panel.get("type")
        # A param control sourcing choices statically (`options`), from another
        # dataset (`optionsFrom`), or as free text (`style:"search"`) drives a
        # backend query and needs no data of its own — exempt from the data
        # requirement. A config control likewise drives a target panel.
        param_with_options = ptype == "control" and panel.get("mode") == "param" and (
            isinstance(panel.get("options"), list)
            or panel.get("optionsFrom") is not None
            or panel.get("style") == "search")
        is_config_control = ptype == "control" and panel.get("mode") == "config"

        if (ptype not in ("text", "image", "filters") and not param_with_options
                and not is_config_control
                and panel.get("dataRef") is None and panel.get("data") is None):
            errors.append(at + " must have either a dataRef or inline data")

        if ptype == "image":
            if panel.get("src") is not None and not _is_str(panel.get("src")):
                errors.append(at + ".src must be a string (URL or data: URI)")
            if panel.get("fit") is not None and panel.get("fit") not in _FIT:
                errors.append(at + '.fit must be "contain", "cover", "fill", "none", or "scale-down"')

        if ptype == "filters":
            _check_filters_panel(panel, at, data, errors)
        if panel.get("transpose") is not None and not isinstance(panel.get("transpose"), bool):
            errors.append(at + ".transpose must be true or false")
        if ptype == "control":
            _check_control_panel(panel, panels, data, params, at, errors)

        if panel.get("align") is not None and panel.get("align") not in _ALIGN:
            errors.append(at + '.align must be "left", "center", or "right"')
        if panel.get("valign") is not None and panel.get("valign") not in _VALIGN:
            errors.append(at + '.valign must be "top", "middle", or "bottom"')

        data_ref = panel.get("dataRef")
        if data_ref is not None and (not _is_obj(data) or data_ref not in data):
            errors.append(at + '.dataRef "%s" has no matching entry in spec.data' % data_ref)

        # Chart-click cross-filter: a clicked mark sets this parameter.
        click_param = panel.get("clickParam")
        if click_param is not None:
            if not _is_str(click_param):
                errors.append(at + ".clickParam must be a string")
            elif not _is_obj(params) or click_param not in params:
                errors.append(at + '.clickParam "%s" has no matching entry in spec.params' % click_param)


def _check_control_panel(panel: dict, panels: Any, data: Any, params: Any,
                         at: str, errors: list) -> None:
    """Validate a control panel's compartment/style/mode and mode wiring."""
    compartment = panel.get("compartment")
    if compartment is not None and compartment not in ("x", "z"):
        errors.append(at + '.compartment must be "x" (samples) or "z" (variables)')

    if panel.get("style") is not None and panel.get("style") not in _STYLE:
        errors.append(at + '.style must be "auto", "dropdown", "radio", "buttons", "search", or "slider"')

    mode = panel.get("mode")
    if mode is not None and mode not in ("filter", "param", "config"):
        errors.append(at + '.mode must be "filter", "param", or "config"')

    if mode == "config":
        target = panel.get("target")
        if not _is_str(target) or not target:
            errors.append(at + ' of mode "config" requires a target panel id string')
        elif not _is_obj(panels) or target not in panels:
            errors.append(at + '.target "%s" has no matching entry in spec.panels' % target)
        options = panel.get("options")
        if not isinstance(options, list) or not options:
            errors.append(at + ' of mode "config" requires a non-empty options array')
        else:
            for oi, opt in enumerate(options):
                if not _is_obj(opt) or not _is_obj(opt.get("config")):
                    errors.append(at + ".options[%d] must be an object with a config fragment" % oi)

    if mode == "param":
        name = panel.get("param")
        if not _is_str(name) or not name:
            errors.append(at + ' of mode "param" requires a param name string')
        elif not _is_obj(params) or name not in params:
            errors.append(at + '.param "%s" has no matching entry in spec.params' % name)
        options_from = panel.get("optionsFrom")
        if options_from is not None:
            if not _is_obj(options_from):
                errors.append(at + ".optionsFrom must be an object")
            elif not _is_str(options_from.get("dataRef")) or \
                    not _is_obj(data) or options_from.get("dataRef") not in data:
                errors.append(at + ".optionsFrom.dataRef has no matching entry in spec.data")


def _check_data(data: Any, params: Any, errors: list, lenient: Optional[list] = None) -> None:
    """Validate spec.data sources and their $param query tokens.

    :param lenient: For a newer-MINOR spec, a list collecting unknown-kind
        warnings (those sources are skipped rather than failing the spec).
    """
    if data is None:
        return
    if not _is_obj(data):
        errors.append("spec.data must be an object map")
        return

    for key, src in data.items():
        at = 'spec.data["%s"]' % key
        if not _is_obj(src):
            errors.append(at + " must be an object")
            continue
        kind = src.get("kind")
        if kind not in _DATA_KINDS:
            message = at + '.kind must be "inline", "connector", "dataset", "join", "function", or "live"'
            if lenient is not None:
                lenient.append(message + " (unknown kind from a newer format: skipped)")
            else:
                errors.append(message)
        if kind == "inline" and src.get("value") is None:
            errors.append(at + ' of kind "inline" requires a value')
        if kind == "connector" and not _is_str(src.get("url")):
            errors.append(at + ' of kind "connector" requires a url string')
        if kind == "live":
            _check_live(src, at, errors)
        if kind == "dataset" and (not _is_str(src.get("id")) or not src.get("id")):
            errors.append(at + ' of kind "dataset" requires an id string')
        if kind == "join":
            _check_join(src, at, data, errors)
        elif src.get("axis") is not None and src.get("axis") not in _AXES:
            errors.append(at + '.axis must be "smps" or "vars"')
        if kind == "function":
            _check_function(src, at, data, params, errors)
        if src.get("pushdown") is not None:
            _check_pushdown(src, at, params, errors)
        if src.get("calculatedFields") is not None:
            _check_calculated_fields(src, at, errors)

        # A `query` template maps request keys to literals or "$param" tokens;
        # every token must name a declared parameter.
        query = src.get("query")
        if query is not None:
            if not _is_obj(query):
                errors.append(at + ".query must be an object map")
            else:
                for qk, token in query.items():
                    if _is_str(token) and token.startswith("$"):
                        name = token[1:]
                        if not _is_obj(params) or name not in params:
                            errors.append(at + '.query["%s"] references undeclared param "%s"' % (qk, name))

    # A join / data function must not read itself, directly or indirectly.
    # Flag each ref on the cycle; a ref that only leads into one is not flagged.
    for key, src in data.items():
        if _is_obj(src) and src.get("kind") in ("join", "function"):
            cycle = join_cycle(key, data)
            if cycle and cycle[0] == key:
                errors.append('spec.data["%s"] %s depends on itself (%s)'
                              % (key, src.get("kind"), " -> ".join(cycle)))


def _check_live(src: dict, at: str, errors: list) -> None:
    """Validate a kind:"live" source (port of ``checkLive`` in src/validateSpec.js): a url
    string (the connectors SSE stream), an optional positive-integer ``window`` (samples
    kept), an optional ``variables`` string list, and an optional ``initial`` data object."""
    if not _is_str(src.get("url")) or not src.get("url"):
        errors.append(at + ' of kind "live" requires a url string')
    window = src.get("window")
    if window is not None and not (_is_int(window) and window > 0):
        errors.append(at + ".window must be a positive integer")
    variables = src.get("variables")
    if variables is not None and not (isinstance(variables, list)
                                      and all(_is_str(v) for v in variables)):
        errors.append(at + ".variables must be an array of strings")
    initial = src.get("initial")
    if initial is not None and not (_is_obj(initial) and initial.get("y")):
        errors.append(at + ".initial must be a CanvasXpress data object with a y block")


def _check_join(src: dict, at: str, data: dict, errors: list) -> None:
    """Validate a kind:"join" source: left/right refs, how, on, axes, suffix."""
    _check_relation(src, at, data, errors, 'of kind "join"')
    if src.get("how") is not None and src.get("how") not in _JOIN_TYPES:
        errors.append(at + '.how must be one of "%s"' % '", "'.join(_JOIN_TYPES))
    if src.get("suffix") is not None and not _is_str(src.get("suffix")):
        errors.append(at + ".suffix must be a string")


def _check_relation(src: dict, at: str, data: Any, errors: list, what: str) -> None:
    """Validate what a join and a spec.relationships entry share: refs, on, axes."""
    for side in ("left", "right"):
        ref = src.get(side)
        if not _is_str(ref) or not ref:
            errors.append(at + " %s requires a %s ref string" % (what, side))
        elif not _is_obj(data) or ref not in data:
            errors.append(at + '.%s "%s" has no matching entry in spec.data' % (side, ref))
    on = src.get("on")
    if on is not None:
        keys = on if isinstance(on, list) else [on]

        def valid_key(key: Any) -> bool:
            if _is_str(key):
                return bool(key)
            return _is_obj(key) and _is_str(key.get("left")) and bool(key.get("left")) \
                and _is_str(key.get("right")) and bool(key.get("right"))

        if not keys or not all(valid_key(k) for k in keys):
            errors.append(
                at + ".on must be a column name, {left, right}, or a non-empty array of those")
    for field in ("axis", "leftAxis", "rightAxis"):
        if src.get(field) is not None and src.get(field) not in _AXES:
            errors.append(at + '.%s must be "smps" or "vars"' % field)


def source_inputs(src: Any) -> list:
    """The refs a derived source reads (mirrors ``sourceInputs`` in ``src/join.js``)."""
    if not _is_obj(src):
        return []
    if src.get("kind") == "join":
        return [src.get("left"), src.get("right")]
    if src.get("kind") == "function":
        inputs = src.get("inputs")
        if isinstance(inputs, list):
            return list(inputs)
        if _is_obj(inputs):
            return list(inputs.values())
    return []


def join_cycle(ref: str, data: dict) -> Optional[list]:
    """Find a cycle through join / data-function inputs reachable from ``ref``.

    Mirrors ``derivedCycle`` in ``src/join.js``.

    :returns: the cycle path (first ref repeated last), or ``None``.
    """
    path: list = []
    done: set = set()

    def visit(node: Any) -> Optional[list]:
        if node in path:
            return path[path.index(node):] + [node]
        if not _is_str(node) or node in done:
            return None
        inputs = source_inputs(data.get(node))
        if inputs:
            path.append(node)
            found = None
            for item in inputs:
                found = found or visit(item)
            path.pop()
            if found:
                return found
        done.add(node)
        return None

    return visit(ref)


_PUSHDOWN_KEYS = ("columns", "groupBy", "measures", "where", "orderBy", "limit", "filters")
_PUSHDOWN_FNS = ("count", "count_distinct", "sum", "avg", "mean", "min", "max")
_PUSHDOWN_OPS = ("=", "!=", "<", "<=", ">", ">=", "in", "not_in", "between", "is_null",
                 "not_null")


def _check_calculated_fields(src: dict, at: str, errors: list) -> None:
    """Check a source's ``calculatedFields`` shape (mirrors validateSpec.js).

    Each entry is ``{name, target?, formula}`` or ``{name, target?, bin: {field,
    method?, bins?, breaks?}}``; the formula itself is parsed by the CanvasXpress
    engine at render. A live source's ticks bypass the computed columns.
    """
    here = at + ".calculatedFields"
    if src.get("kind") == "live":
        errors.append(here + " is not supported on a live source (its ticks bypass the computed columns)")
        return
    defs = src.get("calculatedFields")
    if not isinstance(defs, list):
        errors.append(here + " must be an array")
        return
    seen = set()
    for i, d in enumerate(defs):
        item = "%s[%d]" % (here, i)
        if not _is_obj(d):
            errors.append(item + " must be an object")
            continue
        name = d.get("name")
        if not _is_str(name) or not name:
            errors.append(item + " requires a name string")
        elif name in seen:
            errors.append(item + ' repeats the name "%s"' % name)
        else:
            seen.add(name)
        if d.get("target") is not None and d.get("target") not in _CALC_TARGETS:
            errors.append(item + '.target must be "variable", "sampleAnnotation", or "variableAnnotation"')
        formula = d.get("formula")
        has_formula = _is_str(formula) and formula.strip() != ""
        has_bin = d.get("bin") is not None
        if has_formula == has_bin:
            errors.append(item + " needs exactly one of a formula string or a bin")
        if formula is not None and not _is_str(formula):
            errors.append(item + ".formula must be a string")
        if has_bin:
            b = d.get("bin")
            if not _is_obj(b):
                errors.append(item + ".bin must be an object")
                continue
            if not _is_str(b.get("field")) or not b.get("field"):
                errors.append(item + ".bin requires a field string")
            if b.get("method") is not None and b.get("method") not in _BIN_METHODS:
                errors.append(item + '.bin.method must be "equalWidth", "quantile", "percentile", or "custom"')
            if b.get("bins") is not None and not (_is_int(b.get("bins")) and b.get("bins") >= 1):
                errors.append(item + ".bin.bins must be a whole number of at least 1")
            brk = b.get("breaks")
            if brk is not None and not (
                isinstance(brk, list)
                and all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in brk)
            ):
                errors.append(item + ".bin.breaks must be a list of numbers")


def _check_pushdown(src: dict, at: str, params: Any, errors: list) -> None:
    """Check a source's ``pushdown`` block (mirrors validateSpec.js).

    It runs in the source's database when it has one (connector, a database
    join), otherwise in the browser (inline, dataset, function, a browser join).
    """
    p = src.get("pushdown")
    here = at + ".pushdown"
    # A join takes `true` (join in the database) or a query run over the joined rows.
    if src.get("kind") == "join" and isinstance(p, bool):
        return
    if src.get("kind") == "live":
        errors.append(here + " is not supported on a live source")
        return
    if not _is_obj(p):
        errors.append(here + " must be an object")
        return
    for k in p:
        if k not in _PUSHDOWN_KEYS:
            errors.append(here + ' has an unknown key "%s"' % k)
    for k in ("columns", "groupBy"):
        v = p.get(k)
        if v is not None and not (isinstance(v, list) and all(_is_str(c) and c for c in v)):
            errors.append(here + "." + k + " must be a list of column names")
    measures = p.get("measures")
    if measures is not None:
        if not isinstance(measures, list):
            errors.append(here + ".measures must be a list")
        else:
            for i, m in enumerate(measures):
                mat = here + ".measures[%d]" % i
                if not _is_obj(m) or m.get("fn") not in _PUSHDOWN_FNS:
                    errors.append(mat + ".fn must be one of: " + ", ".join(_PUSHDOWN_FNS))
                elif m.get("fn") != "count" and not _is_str(m.get("column")):
                    errors.append(mat + " needs a column")
    where = p.get("where")
    if where is not None:
        if not isinstance(where, list):
            errors.append(here + ".where must be a list")
        else:
            for i, w in enumerate(where):
                wat = here + ".where[%d]" % i
                if not _is_obj(w) or not _is_str(w.get("column")):
                    errors.append(wat + " needs a column")
                    continue
                if w.get("op") is not None and w.get("op") not in _PUSHDOWN_OPS:
                    errors.append(wat + ".op must be one of: " + " ".join(_PUSHDOWN_OPS))
                value = w.get("value")
                if _is_str(value) and value.startswith("$"):
                    name = value[1:]
                    if not _is_obj(params) or name not in params:
                        errors.append(wat + '.value references undeclared param "%s"' % name)
    order = p.get("orderBy")
    if order is not None and not (isinstance(order, list) and all(
            _is_str(o) or (_is_obj(o) and _is_str(o.get("column"))) for o in order)):
        errors.append(here + ".orderBy must be a list of names or {column, desc}")
    limit = p.get("limit")
    if limit is not None and not (_is_int(limit) and 1 <= limit <= 1000000):
        errors.append(here + ".limit must be a whole number from 1 to 1000000")
    if p.get("filters") is not None and not isinstance(p.get("filters"), bool):
        errors.append(here + ".filters must be true or false")
    grouped = bool((isinstance(p.get("groupBy"), list) and p.get("groupBy"))
                   or (isinstance(measures, list) and measures))
    if isinstance(p.get("columns"), list) and p.get("columns") and grouped:
        errors.append(here + " uses columns (rows) or groupBy/measures (aggregates), not both")


def _check_function(src: dict, at: str, data: dict, params: Any, errors: list) -> None:
    """Validate a kind:"function" source (mirrors ``checkFunction`` in validateSpec.js)."""
    if src.get("language") not in _FUNCTION_LANGUAGES:
        errors.append(at + ' of kind "function" requires language "python" or "r"')
    code = src.get("code")
    if not _is_str(code) or not code.strip():
        errors.append(at + ' of kind "function" requires a code string')
    inputs = src.get("inputs")
    pairs = []
    if isinstance(inputs, list):
        pairs = [(ref, ref) for ref in inputs]
    elif _is_obj(inputs):
        pairs = list(inputs.items())
    elif inputs is not None:
        errors.append(at + ".inputs must be an array of refs or a {name: ref} map")
    for name, ref in pairs:
        if not _is_str(name) or not _IDENTIFIER.match(name):
            errors.append(at + '.inputs name "%s" must be an identifier (letters, digits, _)'
                          % _js_str(name))
        if not _is_str(ref) or ref not in data:
            errors.append(at + '.inputs "%s" has no matching entry in spec.data' % _js_str(ref))
    args = src.get("args")
    if args is not None:
        if not _is_obj(args):
            errors.append(at + ".args must be an object map")
        else:
            for name, token in args.items():
                if not (_is_str(token) and token.startswith("$")):
                    continue
                if not _is_obj(params) or token[1:] not in params:
                    errors.append(at + '.args["%s"] references undeclared param "%s"'
                                  % (name, token[1:]))
    if src.get("runtime") is not None and not _is_str(src.get("runtime")):
        errors.append(at + ".runtime must be a URL string")


def _js_str(value: Any) -> str:
    """Render a value the way JS string concatenation would (for parity)."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _check_filters_panel(panel: dict, at: str, data: Any, errors: list) -> None:
    """Validate a type:"filters" panel's fields and sources."""
    fields = panel.get("fields")
    if fields is not None and not isinstance(fields, list):
        errors.append(at + ".fields must be an array")
        return
    fields = fields or []
    if panel.get("dataRef") is None and not fields:
        errors.append(at + ' of type "filters" requires a dataRef or fields')
    for i, f in enumerate(fields):
        fat = at + ".fields[%d]" % i
        if _is_str(f):
            if not f:
                errors.append(fat + " must be a non-empty field name")
            elif panel.get("dataRef") is None:
                errors.append(fat + " needs the panel dataRef (or use {field, dataRef})")
            continue
        if not _is_obj(f) or not _is_str(f.get("field")) or not f.get("field"):
            errors.append(fat + " must be a field name or {field, dataRef?, kind?}")
            continue
        if f.get("dataRef") is not None and (not _is_obj(data) or f.get("dataRef") not in data):
            errors.append(fat + '.dataRef "%s" has no matching entry in spec.data'
                          % f.get("dataRef"))
        elif f.get("dataRef") is None and panel.get("dataRef") is None:
            errors.append(fat + " needs a dataRef (on the field or the panel)")
        if f.get("kind") is not None and f.get("kind") not in _FIELD_KINDS:
            errors.append(fat + '.kind must be "values", "range", or "search"')


def _check_insights(insights: Any, data: Any, errors: list) -> None:
    """Check the optional ``insights`` block (mirrors ``checkInsights`` in validateSpec.js)."""
    if insights is None:
        return
    if not isinstance(insights, dict):
        errors.append("spec.insights must be an object")
        return
    ops = ("lt", "le", "gt", "ge")
    thresholds = insights.get("thresholds")
    if thresholds is not None:
        if not isinstance(thresholds, list):
            errors.append("spec.insights.thresholds must be an array")
        else:
            for i, t in enumerate(thresholds):
                at = "spec.insights.thresholds[%d]" % i
                if not isinstance(t, dict):
                    errors.append(at + " must be an object")
                    continue
                if not isinstance(t.get("field"), str) or not t.get("field"):
                    errors.append(at + ".field must be a non-empty field name")
                if t.get("op") not in ops:
                    errors.append(at + ".op must be one of: " + ", ".join(ops))
                value = t.get("value")
                if isinstance(value, bool) or not isinstance(value, (int, float)) or value != value \
                        or value in (float("inf"), float("-inf")):
                    errors.append(at + ".value must be a number")
                if t.get("label") is not None and not isinstance(t.get("label"), str):
                    errors.append(at + ".label must be a string")
                if t.get("source") is not None and (not isinstance(data, dict) or t["source"] not in data):
                    errors.append(at + '.source "%s" has no matching entry in spec.data' % t["source"])
    sources = insights.get("sources")
    if sources is not None:
        if not isinstance(sources, list):
            errors.append("spec.insights.sources must be an array of source names")
        else:
            for i, ref in enumerate(sources):
                if not isinstance(ref, str) or not isinstance(data, dict) or ref not in data:
                    errors.append('spec.insights.sources[%d] "%s" has no matching entry in spec.data' % (i, ref))
    limit = insights.get("max")
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
        errors.append("spec.insights.max must be a positive whole number")


def _check_filter_schemes(schemes: Any, data: Any, errors: list) -> None:
    """Validate spec.filterSchemes: name -> array of saved filters."""
    if schemes is None:
        return
    if not _is_obj(schemes):
        errors.append("spec.filterSchemes must be an object map")
        return
    for name, scheme in schemes.items():
        at = 'spec.filterSchemes["%s"]' % name
        if not isinstance(scheme, list):
            errors.append(at + " must be an array of filters")
            continue
        for i, p in enumerate(scheme):
            pat = at + "[%d]" % i
            if not _is_obj(p):
                errors.append(pat + " must be an object")
                continue
            if not _is_str(p.get("dataRef")) or not _is_obj(data) or p.get("dataRef") not in data:
                errors.append(pat + ".dataRef has no matching entry in spec.data")
            if not _is_str(p.get("field")) or not p.get("field"):
                errors.append(pat + ".field must be a non-empty string")
            if p.get("values") is not None and not isinstance(p.get("values"), list):
                errors.append(pat + ".values must be an array")
            for k in ("min", "max"):
                v = p.get(k)
                if v is not None and (isinstance(v, bool) or not isinstance(v, (int, float))
                                      or v != v or v in (float("inf"), float("-inf"))):
                    errors.append(pat + ".%s must be a number" % k)
            if p.get("text") is not None and not _is_str(p.get("text")):
                errors.append(pat + ".text must be a string")


def _check_relationships(relationships: Any, data: Any, errors: list) -> None:
    """Validate spec.relationships (cross-source marking / filtering)."""
    if relationships is None:
        return
    if not isinstance(relationships, list):
        errors.append("spec.relationships must be an array")
        return
    for i, rel in enumerate(relationships):
        at = "spec.relationships[%d]" % i
        if not _is_obj(rel):
            errors.append(at + " must be an object")
            continue
        _check_relation(rel, at, data, errors, "relationship")


def _check_params(params: Any, errors: list) -> None:
    """Validate spec.params — a bare default value or a {value, type} object."""
    if params is None:
        return
    if not _is_obj(params):
        errors.append("spec.params must be an object map")
        return
    for name, definition in params.items():
        if _is_obj(definition) and definition.get("type") is not None \
                and definition.get("type") not in _PARAM_TYPES:
            errors.append('spec.params["%s"].type must be "string", "number", or "boolean"' % name)


def _check_controls(controls: Any, data: Any, errors: list) -> None:
    """Validate the top-level controls array."""
    if controls is None:
        return
    if not isinstance(controls, list):
        errors.append("spec.controls must be an array")
        return
    for i, control in enumerate(controls):
        at = "spec.controls[%d]" % i
        if not _is_obj(control):
            errors.append(at + " must be an object")
            continue
        if control.get("kind") not in ("filter", "table"):
            errors.append(at + '.kind must be "filter" or "table"')
        ref = control.get("dataRef")
        if ref is not None and (not _is_obj(data) or ref not in data):
            errors.append(at + '.dataRef "%s" has no matching entry in spec.data' % ref)
