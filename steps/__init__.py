"""
steps package
-------------
Step handlers for the Streamlit app.

- step_input.py     -> collects paths & subject metadata, validates inputs
- step_confirmInputs.py   -> shows summary, creates/updates session JSON
- step_segmentation.py   -> embeds Bokeh app and saves segmentation ranges
"""

# Keep startup fast: don't eagerly import step modules here.
# Import them directly where needed, e.g.:
#   from steps.step_input import run_step as step_input

__all__ = ["step_input", "step_confirmInputs", "step_segmentation"]
