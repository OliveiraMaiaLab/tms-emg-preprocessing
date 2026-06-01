# app/steps/step_peakCorrection.py
# -*- coding: utf-8 -*-
"""
steps/step_peakCorrection.py
-----------------------------
Step 6: Peak correction (with draggable points via embedded Dash).

Only MEPs where peaks_flag == 1 AND noise_flag == 0 appear in the queue.
Only active MEP blocks (from session["active_blocks"]) are included.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import socket
import threading
import urllib.parse

import streamlit as st

from app.utils.dash_peak_editor import create_dash_peak_editor, SHOW_DEBUG
from app.utils.persistence import (
    ensure_metadata,
    ensure_template_loaded,
    ensure_session_file,
    update_processed_sessions_registry,
)
from app.utils.layout import render_text, step_nav
from app.utils.tms_module import read_json

PREV_STEP = "peak_checking"
THIS_STEP = "peak_correction"
NEXT_STEP = None


@dataclass(frozen=True)
class MEPKey:
    block:   str
    hemi:    str
    mep_idx: int


def _find_free_port() -> int:
    s = socket.socket()
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return int(port)


class _StripFrameHeadersMiddleware:
    """Strip frame-blocking headers from Dash responses (local embedding)."""

    def __init__(self, app):
        self.app = app

    def __call__(self, environ, start_response):
        def _start_response(status, headers, exc_info=None):
            blocked  = {"x-frame-options", "content-security-policy", "frame-options"}
            headers2 = [(k, v) for (k, v) in headers if k.lower() not in blocked]
            return start_response(status, headers2, exc_info)

        return self.app(environ, _start_response)


def _dash_code_sig(meta: dict) -> float:
    base = Path(meta.get("_script_dir", ".")).resolve()
    for p in [base / "app" / "utils" / "dash_peak_editor.py",
              base / "utils" / "dash_peak_editor.py"]:
        try:
            if p.exists():
                return float(p.stat().st_mtime)
        except Exception:
            pass
    return 0.0


def _ensure_dash_running(meta: dict, session_file: str | Path) -> int:
    session_file = str(session_file)
    sig          = _dash_code_sig(meta)

    if (
        st.session_state.get("_pcorr_dash_started")
        and st.session_state.get("_pcorr_dash_port")
        and st.session_state.get("_pcorr_dash_sig") == sig
    ):
        return int(st.session_state["_pcorr_dash_port"])

    port     = _find_free_port()
    dash_app = create_dash_peak_editor(meta=meta, session_file=session_file)

    try:
        dash_app.server.wsgi_app = _StripFrameHeadersMiddleware(dash_app.server.wsgi_app)
    except Exception:
        pass

    def _run():
        dash_app.run(host="0.0.0.0", port=port, debug=False)

    threading.Thread(target=_run, daemon=True).start()

    st.session_state["_pcorr_dash_started"] = True
    st.session_state["_pcorr_dash_port"]    = int(port)
    st.session_state["_pcorr_dash_sig"]     = sig
    return int(port)


def _guess_streamlit_hostname() -> str:
    for c in [st.get_option("browser.serverAddress"), st.get_option("server.address")]:
        if isinstance(c, str):
            c = c.strip()
            if c and c not in {"0.0.0.0", "127.0.0.1"}:
                return c
    return "localhost"


def _dash_url(port: int, *, block: str, hemi: str, mep_idx: int) -> str:
    qs = urllib.parse.urlencode({"block": block, "hemi": hemi, "mep": str(int(mep_idx))})
    return f"http://{_guess_streamlit_hostname()}:{int(port)}/?{qs}"


def _ensure_payload_lists(payload: dict) -> None:
    """Ensure all list fields are present and length-matched to pulses."""
    pulses = payload.get("pulses", [])
    n = len(pulses) if isinstance(pulses, list) else 0

    for k in ("min", "max", "peaks_flag", "noise_flag", "below_threshold_flag"):
        if not isinstance(payload.get(k), list):
            payload[k] = []

    def _fit(name: str, fill):
        arr = payload[name]
        if len(arr) < n:
            arr.extend([fill] * (n - len(arr)))
        elif len(arr) > n:
            payload[name] = arr[:n]

    _fit("peaks_flag",           0)
    _fit("noise_flag",           0)
    _fit("below_threshold_flag", 0)
    _fit("min",  None)
    _fit("max",  None)


def _get_payload(session: dict, block: str, hemi: str) -> dict:
    meps    = session.get("meps", {}) or {}
    payload = ((meps.get(str(block)) or {}).get(str(hemi)) or {})
    if not isinstance(payload, dict):
        payload = {}
    _ensure_payload_lists(payload)
    return payload


def _build_queue(session: dict, blocks: list[str], hemis: list[str]) -> list[MEPKey]:
    """
    Build the list of MEPs that need correction:
      peaks_flag == 1  AND  noise_flag == 0.

    MEPs flagged as noise are excluded even if peaks_flag is set — there is
    nothing useful to correct in a noisy trace.
    """
    out: list[MEPKey] = []
    for b in blocks:
        for h in hemis:
            payload     = _get_payload(session, b, h)
            flags       = payload.get("peaks_flag",  []) or []
            noise_flags = payload.get("noise_flag",  []) or []

            for i, f in enumerate(flags):
                try:
                    if int(f) != 1:
                        continue
                    noise = int(noise_flags[i]) if i < len(noise_flags) else 0
                    if noise == 1:
                        continue
                    out.append(MEPKey(block=str(b), hemi=str(h), mep_idx=int(i)))
                except Exception:
                    continue
    return out


def _visited_key()  -> str: return "_pcorr_visited"
def _current_key()  -> str: return "_pcorr_current"
def _key_str(k: MEPKey) -> str: return f"{k.block}|{k.hemi}|{k.mep_idx}"

def _parse_key_str(s: str) -> MEPKey | None:
    try:
        b, h, i = s.split("|")
        return MEPKey(block=b, hemi=h, mep_idx=int(i))
    except Exception:
        return None


def _remaining_queue(session: dict, blocks: list[str], hemis: list[str],
                     visited: set[str]) -> list[MEPKey]:
    return [k for k in _build_queue(session, blocks, hemis) if _key_str(k) not in visited]


def _finish_to_input(meta: dict, session_file: Path) -> None:
    update_processed_sessions_registry(
        data_file=meta.get("input_file", "UNKNOWN_DATA_FILE"),
        session_file=session_file,
        researcher_id=meta.get("researcher_id", ""),
        pipeline_version=meta.get("version", ""),
    )
    st.session_state["_global_flash_success"] = f'Finished processing "{session_file.name}".'
    st.session_state.step = "input"
    st.rerun()


def run_step(meta: dict):
    meta         = ensure_metadata()
    meta         = ensure_template_loaded(meta)
    session_file = Path(ensure_session_file(meta))

    if _visited_key() not in st.session_state:
        st.session_state[_visited_key()] = set()
    visited: set[str] = st.session_state[_visited_key()]

    session = read_json(session_file)

    # Only process active MEP blocks.
    active_set = set(session.get("active_blocks", [
        b for b in meta.get("exp_structure", []) if b != "emg_ref"
    ]))
    blocks = [
        b for b in (meta.get("exp_structure", []) or [])
        if str(b).lower().endswith("meps") and b in active_set
    ]
    if not blocks:
        st.error("No active '*meps' blocks found in meta['exp_structure'].")
        return

    hemis = list((session.get("info", {}) or {}).get("hemispheres", meta.get("hemispheres", ["left"])))
    if not hemis:
        hemis = ["left"]

    remaining = _remaining_queue(
        session,
        blocks=[str(b) for b in blocks],
        hemis=[str(h) for h in hemis],
        visited=visited,
    )

    if not remaining:
        step_nav(
            THIS_STEP,
            step_title="Peak Correction",
            right_label="Finish ✅",
            next_step=NEXT_STEP,
            back_step=PREV_STEP,
            disabled_next=False,
            on_next=(lambda: _finish_to_input(meta, session_file)),
        )
        st.success("No remaining flagged MEPs to correct.")
        return

    cur_s = st.session_state.get(_current_key(), "") or ""
    cur   = _parse_key_str(cur_s)

    remaining_set = {_key_str(k) for k in remaining}
    if cur is None or _key_str(cur) not in remaining_set:
        cur = remaining[0]
        st.session_state[_current_key()] = _key_str(cur)

    pos     = next((i for i, k in enumerate(remaining) if _key_str(k) == _key_str(cur)), 0)
    key     = remaining[pos]
    is_last = (pos >= len(remaining) - 1)

    step_nav(
        THIS_STEP,
        step_title="Peak Correction",
        right_label="Finish ✅",
        next_step=NEXT_STEP,
        back_step=PREV_STEP,
        disabled_next=(not is_last),
        on_next=(lambda: _finish_to_input(meta, session_file)) if is_last else None,
    )

    render_text(
        f"MEP {key.mep_idx} in&nbsp;<em>{key.block}</em>",
        horizontal_alignment="center",
        nowrap=True,
        heading_level=3,
    )

    port     = _ensure_dash_running(meta, session_file)
    dash_url = _dash_url(port, block=key.block, hemi=key.hemi, mep_idx=key.mep_idx)

    if SHOW_DEBUG:
        st.link_button("Open Dash directly", dash_url, use_container_width=True)

    st.markdown(
        """
        <style>
        .dash-embed-wrap  { width: 100%; display: flex; justify-content: center; overflow: hidden; }
        .dash-embed-frame { width: min(1200px, 100%); height: 55vh; border: 0; overflow: hidden; display: block; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="dash-embed-wrap">
        <iframe
            class="dash-embed-frame"
            src="{dash_url}"
            scrolling="no"
            sandbox="allow-scripts allow-same-origin allow-forms allow-popups"
        ></iframe>
        </div>
        """,
        unsafe_allow_html=True,
    )

    b1, b2, b3 = st.columns([1, 1, 1])

    with b1:
        if st.button("◀ Prev", use_container_width=True, disabled=(pos == 0)):
            prev_key = remaining[pos - 1]
            st.session_state[_current_key()] = _key_str(prev_key)
            st.rerun()

    with b2:
        render_text(
            f"MEP {pos+1}/{len(remaining)}  ({key.hemi} hemisphere)",
            horizontal_alignment="center",
            nowrap=True,
        )

    with b3:
        if st.button("Next ▶", use_container_width=True, disabled=is_last):
            visited.add(_key_str(key))
            st.session_state[_visited_key()] = visited

            session2    = read_json(session_file)
            remaining2  = _remaining_queue(
                session2,
                blocks=[str(b) for b in blocks],
                hemis=[str(h) for h in hemis],
                visited=visited,
            )

            if not remaining2:
                st.session_state[_current_key()] = ""
                st.rerun()

            next_pos = min(pos, len(remaining2) - 1)
            st.session_state[_current_key()] = _key_str(remaining2[next_pos])
            st.rerun()
