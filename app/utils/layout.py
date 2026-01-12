# app/utils/layout.py
"""
layout.py
---------
Small UI helpers (headings, alignment, nav bar) to keep Streamlit steps tidy.
"""

from __future__ import annotations

import streamlit as st


def render_text(
    text: str,
    *,
    font_color: str = "black",
    font_weight: str = "normal",
    horizontal_alignment: str = "right",
    font_size: str | None = None,
    nowrap: bool = True,
    heading_level: int | None = None,
) -> None:
    justify_map = {"left": "flex-start", "center": "center", "right": "flex-end"}
    h_align = justify_map.get(horizontal_alignment.lower(), "flex-end")
    white_space = "nowrap" if nowrap else "normal"

    heading_sizes = {1: "2rem", 2: "1.75rem", 3: "1.5rem", 4: "1.25rem", 5: "1rem"}
    if heading_level in heading_sizes and font_size is None:
        font_size = heading_sizes[heading_level]
        font_weight = "bold"

    # If font_size stays None, let browser default.
    font_size_css = f"font-size:{font_size};" if font_size else ""

    html = (
        f"<div style='display:flex;width:100%;justify-content:{h_align};"
        f"font-weight:{font_weight};color:{font_color};{font_size_css}"
        f"white-space:{white_space};margin:0.25em 0;'>{text}</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def step_nav(
    step_id: str,
    step_title: str,
    *,
    next_step: str | None = None,
    back_step: str | None = None,
    on_next=None,
    on_back=None,
    right_label: str = "Advance ▶",
    left_label: str = "◀ Back",
    disabled_next: bool = False,
) -> None:
    """
    Back:
      - run on_back() if provided, else go to back_step
      - rerun

    Next:
      - if on_next is provided, it should return True (proceed) or False (block)
      - if proceed and next_step is provided, we go to next_step
      - rerun only when we actually navigate OR the callback navigates itself
    """
    c_left, c_spacer, c_right = st.columns([1, 6, 1])

    with c_spacer:
        render_text(
            step_title,
            font_color="black",
            font_weight="normal",
            horizontal_alignment="center",
            heading_level=1,
            nowrap=True,
        )

    with c_left:
        if back_step or on_back:
            if st.button(
                left_label,
                key=f"nav_back::{step_id}::{back_step}",
                use_container_width=True,
            ):
                if on_back:
                    on_back()
                elif back_step:
                    st.session_state.step = back_step
                st.rerun()

    with c_right:
        if st.button(
            right_label,
            key=f"nav_next::{step_id}::{next_step}",
            use_container_width=True,
            disabled=disabled_next,
        ):
            proceed = True
            if on_next:
                ret = on_next()
                proceed = (ret is True)

            # If next_step is None, on_next likely navigated itself.
            if proceed and next_step:
                st.session_state.step = next_step
                st.rerun()
