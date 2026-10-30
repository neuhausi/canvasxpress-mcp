#!/usr/bin/env python3
"""
generate_few_shots.py
=====================
Regenerate few-shot examples with CanvasXpress's own generator, ``generateExamples``,
running in headless Chromium against a released engine.

The few-shot corpus in ``data/few_shot_examples.json`` is the output of that function:
records with the keys ``id, data, header, description, config``. The function picks
random phrases, so the engine's ``Math.random`` is replaced by a seeded generator and the
same ``--seed`` reproduces the same output. Each graph type is reseeded from the seed and
its own name, so adding or removing a graph type does not change the others.

    python scripts/generate_few_shots.py                       # all graph types, seed 1
    python scripts/generate_few_shots.py --graphs "Bar graph,Boxplot" --per-graph 20
    python scripts/generate_few_shots.py --list-graphs
    python scripts/generate_few_shots.py --engine ./canvasXpress.min.js --seed 7

The output goes to ``data/few_shot_examples.generated.json`` unless ``--output`` is given;
it never overwrites ``data/few_shot_examples.json`` on its own. After copying a corpus
into place, run ``python build_index.py``.

Requires Playwright and Chromium:  pip install playwright && playwright install chromium
Requires a CanvasXpress build that exposes ``generateExamples``.
"""

import argparse
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_ENGINE = "https://www.canvasxpress.org/dist/canvasXpress.min.js"
DEFAULT_OUTPUT = BASE_DIR / "data" / "few_shot_examples.generated.json"

_PAGE = '<!doctype html><html><body><canvas id="cx" width="400" height="300"></canvas></body></html>'

# The engine attaches its methods to the instance asynchronously after construction.
_INIT_JS = """
() => {
  window.__cx = new CanvasXpress("cx",
    {y: {vars: ["v"], smps: ["s1", "s2"], data: [[1, 2]]}}, {graphType: "Bar"});
}
"""
_READY_JS = "() => window.__cx && typeof window.__cx.updateData === 'function'"

# Runs in the page. Graph names are the engine's display names ("Bar graph", "Boxplot").
_RUN_JS = """
([requested, perGraph, sentences, seed]) => {
  function hash(text) {
    var h = 2166136261;
    for (var i = 0; i < text.length; i++) { h = Math.imul(h ^ text.charCodeAt(i), 16777619); }
    return h >>> 0;
  }
  function prng(state) {
    return function () {
      state |= 0; state = state + 0x6D2B79F5 | 0;
      var t = Math.imul(state ^ state >>> 15, 1 | state);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }
  var cx = window.__cx;
  if (typeof cx.generateExamples !== "function") { return {missing: true}; }
  if (!cx.autocompleteObject) { cx.setAutocompleteObject(); }
  var available = cx.autocompleteObject.data.graphs.slice();
  if (requested === null) { return {graphs: available}; }
  var names = requested.length ? requested : available;
  var unknown = names.filter(function (n) { return available.indexOf(n) === -1; });
  if (unknown.length) { return {unknown: unknown, graphs: available}; }
  var original = Math.random, records = [];
  try {
    names.forEach(function (name) {
      Math.random = prng(seed ^ hash(name));
      cx.generateExamples(name, perGraph, sentences).forEach(function (r) { records.push(r); });
    });
  } finally { Math.random = original; }
  return {records: records, version: (typeof CanvasXpress.version !== "undefined" ? String(CanvasXpress.version) : null)};
}
"""


def renumber(records: list) -> list:
    """Give the combined records ids 1..N in order (the engine numbers each call from 1)."""
    for index, record in enumerate(records, start=1):
        record["id"] = index
    return records


def _run(engine: str, requested, per_graph: int, sentences: int, seed: int) -> dict:
    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:  # noqa: BLE001
        sys.exit("Playwright is required: pip install playwright && playwright install chromium "
                 f"({exc})")
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        try:
            page = browser.new_page()
            page.set_content(_PAGE)
            if engine.startswith(("http://", "https://")):
                page.add_script_tag(url=engine)
            else:
                page.add_script_tag(path=str(Path(engine).expanduser()))
            page.evaluate(_INIT_JS)
            try:
                page.wait_for_function(_READY_JS, timeout=15000)
            except Exception:  # noqa: BLE001
                sys.exit(f"The engine did not initialise within 15 s: {engine}")
            return page.evaluate(_RUN_JS, [requested, per_graph, sentences, seed])
        finally:
            browser.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate few-shot examples with generateExamples.")
    parser.add_argument("--engine", default=DEFAULT_ENGINE,
                        help="canvasXpress.min.js URL or file path (default: the published engine)")
    parser.add_argument("--graphs", default="all",
                        help='comma-separated display names (e.g. "Bar graph,Boxplot") or "all"')
    parser.add_argument("--per-graph", type=int, default=60, help="examples per graph type (default 60)")
    parser.add_argument("--sentences", type=int, default=3, help="average sentences per prompt (default 3)")
    parser.add_argument("--seed", type=int, default=1, help="random seed (default 1)")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="output JSON path")
    parser.add_argument("--list-graphs", action="store_true", help="print the graph names and exit")
    args = parser.parse_args()

    if args.list_graphs:
        result = _run(args.engine, None, 0, 0, 0)
    else:
        requested = [] if args.graphs.strip().lower() == "all" else \
            [g.strip() for g in args.graphs.split(",") if g.strip()]
        result = _run(args.engine, requested, args.per_graph, args.sentences, args.seed)

    if result.get("missing"):
        sys.exit("This engine does not expose generateExamples. Use a CanvasXpress build that "
                 f"does (the function is excluded from name obfuscation): {args.engine}")
    if result.get("unknown"):
        sys.exit("Unknown graph name(s): " + ", ".join(result["unknown"]) +
                 "\nUse --list-graphs for the valid names.")
    if args.list_graphs:
        print("\n".join(result["graphs"]))
        return

    records = renumber(result["records"])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records, indent=1) + "\n", encoding="utf-8")
    graph_types = {r["config"].get("graphType") for r in records}
    print(f"Wrote {len(records)} examples ({len(graph_types)} graph types) to {output}")
    print(f"Engine: {args.engine}" + (f" (version {result['version']})" if result.get("version") else ""))
    print(f"Seed: {args.seed}, per graph: {args.per_graph}, sentences: {args.sentences}")


if __name__ == "__main__":
    main()
