"""
Headless server-side rendering for the canvasxpress-mcp server (agent-native #9c).

Renders a CanvasXpress figure to a PNG in a real headless Chromium (via Playwright), using the
same approach as the canvas-ai regression render-harness — highest fidelity, since it runs the
actual published engine. Lets an agent SEE the chart it generated without a browser.

Playwright is an optional dependency: it is imported lazily so the server still starts if it
(or its Chromium download) is not installed; callers get a clear error instead of an import-time
crash. Deploy note: `pip install playwright && playwright install chromium`.
"""

import base64
import json

_HARNESS_HTML = """<!doctype html><html><head><meta charset="utf-8">
<link rel="stylesheet" href="https://www.canvasxpress.org/dist/canvasXpress.css">
<script src="https://www.canvasxpress.org/dist/canvasXpress.min.js"></script>
</head><body>
<canvas id="cx" width="%(w)d" height="%(h)d"></canvas>
<script>
  window.__cxDone = false;
  try {
    new CanvasXpress("cx", %(data)s, %(config)s, false, false,
      [["__cxSetDone", []]]);
  } catch (e) { window.__cxError = String(e); }
  // Fallback: mark done shortly after construction for charts without an afterRender hook.
  setTimeout(function () { window.__cxDone = true; }, 1200);
</script></body></html>"""


def render_png(data: dict, config: dict, width: int = 800, height: int = 600,
               timeout_ms: int = 8000) -> bytes:
    """Render a figure to PNG bytes. Raises RuntimeError if Playwright is unavailable."""
    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Headless rendering requires Playwright. Install with: "
            "pip install playwright && playwright install chromium"
        ) from exc

    html = _HARNESS_HTML % {
        "w": int(width),
        "h": int(height),
        "data": json.dumps(data or {}),
        "config": json.dumps(config or {}),
    }
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        try:
            page = browser.new_page(viewport={"width": int(width), "height": int(height)})
            page.set_content(html, wait_until="networkidle")
            try:
                page.wait_for_function("window.__cxDone === true", timeout=timeout_ms)
            except Exception:  # noqa: BLE001 - best-effort settle; still screenshot what drew
                pass
            png = page.locator("#cx").screenshot()
        finally:
            browser.close()
    return png


def render_png_b64(data: dict, config: dict, **kwargs) -> str:
    """Render a figure and return a base64-encoded PNG string."""
    return base64.b64encode(render_png(data, config, **kwargs)).decode("ascii")
