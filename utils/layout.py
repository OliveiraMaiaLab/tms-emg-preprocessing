"""
layout.py
---------
Small UI helpers (HTML headings, alignment, etc.) to keep Streamlit code tidy.
"""
import streamlit as st

def render_text(text, font_color="black", font_weight="normal",
                horizontal_alignment="right", font_size=None, nowrap=True, heading_level=None):
    justify_map = {"left": "flex-start", "center": "center", "right": "flex-end"}
    h_align = justify_map.get(horizontal_alignment.lower(), "flex-end")
    white_space = "nowrap" if nowrap else "normal"
    heading_sizes = {1:"2rem", 2:"1.75rem", 3:"1.5rem", 4:"1.25rem", 5:"1rem"}
    if heading_level in heading_sizes and font_size is None:
        font_size = heading_sizes[heading_level]; font_weight = "bold"
    html = (f"<div style='display:flex;width:100%;justify-content:{h_align};"
            f"font-weight:{font_weight};color:{font_color};font-size:{font_size};"
            f"white-space:{white_space};margin:0.25em 0;'>{text}</div>")
    st.markdown(html, unsafe_allow_html=True)
