"""
utils package
-------------
Shared helpers for the app.

Modules:
- persistence.py   -> safe JSON I/O, settings, session-file scaffolding, writes
- layout.py        -> small Streamlit layout utilities (headings, alignment)
- bokeh_embed.py   -> embedded Bokeh server & right-panel controls
"""

# Avoid heavy imports here (e.g., Bokeh) to keep Streamlit startup quick.
# Import from submodules directly where you use them, e.g.:
#   from utils.persistence import create_or_update_session_file

__all__ = ["persistence", "layout", "bokeh_embed", "tms_module"]