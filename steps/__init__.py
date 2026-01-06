"""
steps/__init__.py
-------------
Step handlers for the Streamlit app.

- steps/step_input.py     -> collects paths & subject metadata, validates inputs
- steps/step_confirmInputs.py   -> shows summary, creates/updates session JSON
- steps/step_segmentation.py   -> embeds Bokeh app to plot and segment EMG data
- steps/step_mepWindow.py       -> embeds Bokeh app to plot and define MEP range
- steps/step_peakChecking.py    -> embeds Bokeh app to plot and flag automatic MEP peak detection
"""

# Keep startup fast: don't eagerly import step modules here.
# Import them directly where needed, e.g.:
#   from steps.step_input import run_step as step_input

__all__ = ["step_input", "step_confirmInputs", "step_segmentation", 'step_mepWindow', 'step_peakChecking']
