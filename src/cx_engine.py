"""Engine-URL resolution for the MCP's HTML surfaces.

By default the MCP loads the published CanvasXpress engine from the public CDN
(canvasxpress.org/dist, and pinned jsdelivr for the inline widget). Setting the
env var CX_ENGINE_BASE to a base URL — e.g. a local dev server serving the
canvas-ai working tree, http://localhost:8123/dist — repoints every engine
<link>/<script> at that base instead, so local source edits are picked up with
no rebuild.

The override is OPT-IN: when CX_ENGINE_BASE is unset, inject_engine() returns
its input unchanged, so production (which never sets it) emits byte-identical
HTML to before this module existed.
"""
import os

# Public hosts the MCP's HTML hardcodes today; both are rewritten to the base
# ONLY when CX_ENGINE_BASE is set.
_PUBLIC_HOSTS = (
    "https://www.canvasxpress.org/dist",
    "https://canvasxpress.org/dist",
)


def engine_base():
    """The configured engine base URL (no trailing slash), or None when unset."""
    return (os.getenv("CX_ENGINE_BASE") or "").strip().rstrip("/") or None


def engine_css_url():
    """URL of the engine stylesheet, honoring CX_ENGINE_BASE."""
    return (engine_base() or "https://www.canvasxpress.org/dist") + "/canvasXpress.css"


def engine_js_url(minified=True):
    """URL of the engine script, honoring CX_ENGINE_BASE.

    minified=True -> canvasXpress.min.js (the public /dist name); the dev server
    serves both that and the unminified canvasXpress.js name.
    """
    name = "canvasXpress.min.js" if minified else "canvasXpress.js"
    return (engine_base() or "https://www.canvasxpress.org/dist") + "/" + name


def inject_engine(html):
    """Rewrite hardcoded public engine URLs in `html` to CX_ENGINE_BASE.

    Returns `html` unchanged when CX_ENGINE_BASE is unset, so the default
    (production) output is identical to the literal source.
    """
    base = engine_base()
    if not base:
        return html
    for host in _PUBLIC_HOSTS:
        html = html.replace(host, base)
    return html
