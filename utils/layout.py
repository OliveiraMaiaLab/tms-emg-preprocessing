"""
layout.py
---------
Small UI helpers (HTML headings, alignment, etc.) to keep Streamlit code tidy.
"""

import streamlit as st

def step_nav(
    step_id: str,
    *,
    next_step: str | None = None,
    back_step: str | None = None,
    on_next=None,
    on_back=None,
    right_label: str = "Advance ▶",
    left_label: str = "◀ Back",
    disabled_next: bool = False,
):
    """
    Renders a top nav row:
      - optional Back button on the left
      - optional Advance button on the right

    If on_next/on_back are provided, they run first; otherwise we set st.session_state.step.
    """
    c_left, c_spacer, c_right = st.columns([1, 6, 1])

    with c_left:
        if back_step or on_back:
            if st.button(left_label, key=f"nav_back_{step_id}", use_container_width=True):
                if on_back:
                    on_back()
                elif back_step:
                    st.session_state.step = back_step
                st.rerun()

    with c_right:
        if st.button(
            right_label,
            key=f"nav_next_{step_id}",
            use_container_width=True,
            disabled=disabled_next,
        ):
            if on_next:
                on_next()
            elif next_step:
                st.session_state.step = next_step
            st.rerun()

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
