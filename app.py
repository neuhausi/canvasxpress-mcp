"""
Posit Connect / generic ASGI entry point.

Posit Connect looks for an app object in a
top-level app.py when serving Python ASGI content. This module 
adds the src/ directory to sys.path and re-exports the FastMCP
Starlette application defined in src/server.py.

This file is a hosting wrapper only and does not alter server behaviour.
"""

import sys
from pathlib import Path

_SRC = Path(__file__).parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from server import app
