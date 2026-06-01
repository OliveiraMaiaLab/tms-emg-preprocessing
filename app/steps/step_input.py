# app/steps/step_input.py
# -*- coding: utf-8 -*-
"""
steps/step_input.py
-------------------
Step 1: Collect paths and general metadata. Validates inputs and saves
last-used paths to settings.

Layout (requested order):
1) (one line)
   - Experiment template file
   - Data directory
   - Output folder
2) (one line)
   - EMG file (filename only)
   - Researcher ID
3) (one line)
   - Subject ID
   - Session
   - Sampling Rate (Hz)
   - Hemisphere(s)
"""

from __future__ import annotations

import os
import json
from pathlib import Path

import streamlit as st

from app.utils.persistence import (
    ensure_metadata,
    ensure_output_dir,
    save_persisted_defaults,
    resolve_template_path,
)


def _default_registry_path(output_dir: str | None) -> Path:
    """
    processed_sessions.json lives in /config now.
    If output_dir is provided and contains a registry, we allow it, but default is repo_root/config.
    """
    if output_dir:
        p = Path(output_dir) / "processed_sessions.json"
        if p.exists():
            return p

    cwd = Path(os.getcwd())
    p2 = cwd / "config" / "processed_sessions.json"
    return p2


def _load_processed_data_filenames(output_dir: str, registry_name: str = "processed_sessions.json") -> set[str]:
    """
    Reads processed_sessions.json and returns a set of processed data_file basenames.
    If missing/corrupt, returns empty set.
    """
    try:
        reg_path = _default_registry_path(output_dir)
        if registry_name != "processed_sessions.json":
            rp = Path(registry_name)
            reg_path = rp if rp.is_absolute() or rp.parent != Path(".") else (Path(output_dir) / registry_name)

        if not reg_path.exists():
            return set()

        data = json.loads(reg_path.read_text(encoding="utf-8"))
        processed = data.get("processed", [])
        if not isinstance(processed, list):
            return set()

        out: set[str] = set()
        for rec in processed:
            if isinstance(rec, dict) and rec.get("data_file"):
                out.add(str(rec["data_file"]))
        return out
    except Exception:
        return set()


def _list_bin_files(data_dir: str) -> list[str]:
    try:
        p = Path(data_dir).expanduser()
        if not p.exists() or not p.is_dir():
            return []
        return sorted([f.name for f in p.glob("*.bin") if f.is_file()])
    except Exception:
        return []


def _join_data_path(data_dir: str, emg_name: str) -> str:
    data_dir = str(data_dir or "").strip()
    emg_name = str(emg_name or "").strip()

    if not emg_name:
        return ""

    p = Path(emg_name)
    if p.parent != Path(".") or p.is_absolute():
        return str(p)

    if data_dir:
        return str(Path(data_dir) / emg_name)

    return emg_name


def run_step(meta: dict):
    meta = ensure_metadata()

    # Bug #2: resolve_template_path expects a string, not the meta dict.
    # This call is a no-op (result discarded) but serves as a validation
    # that the path resolver handles the current value without crashing.
    resolve_template_path(meta.get("template_file", ""))

    # Flash message (from finishing other steps)
    msg = st.session_state.pop("_global_flash_success", None)
    if msg:
        st.success(msg)

    st.title("TMS-EMG Preprocessing GUI")
    st.caption(f"Pipeline version: `{meta.get('version', '')}`")

    if "_input_name" not in st.session_state:
        st.session_state["_input_name"] = Path(str(meta.get("input_file", ""))).name

    # -------------------------
    # Line 1: template / data_dir / output_dir
    # -------------------------
    a, b, c = st.columns([2.2, 2.2, 2.2])
    with a:
        meta["template_file"] = st.text_input(
            "Experiment template file",
            value=str(meta.get("template_file", "")),
        )

    with b:
        meta["data_dir"] = st.text_input(
            "Data directory (folder containing EMG files)",
            value=str(meta.get("data_dir", "")),
        )

    with c:
        meta["output_dir"] = st.text_input(
            "Output folder (where to save session files and results)",
            value=str(meta.get("output_dir", str(Path(os.getcwd()) / "output"))),
        )

    # -------------------------
    # Line 2: EMG filename (dropdown) / researcher
    # -------------------------
    d, e = st.columns([2.2, 1.2])

    with d:
        data_dir = str(meta.get("data_dir", "")).strip()
        output_dir = str(meta.get("output_dir", "")).strip()

        bin_files = _list_bin_files(data_dir)
        processed = _load_processed_data_filenames(output_dir)

        items = []
        for fn in bin_files:
            is_done = fn in processed
            items.append((is_done, fn))

        items.sort(key=lambda t: (t[0], t[1].lower()))

        display_to_filename: dict[str, str] = {}
        display_labels: list[str] = []

        for is_done, fn in items:
            label = f"✅ {fn}" if is_done else fn
            if label in display_to_filename:
                label = f"{label} "
            display_to_filename[label] = fn
            display_labels.append(label)

        prev_name = st.session_state.get("_input_name", "") or Path(str(meta.get("input_file", ""))).name
        prev_label = None
        for lbl, fn in display_to_filename.items():
            if fn == prev_name:
                prev_label = lbl
                break

        if not display_labels:
            st.warning("No .bin files found in Data directory.")
            selected_label = st.text_input("EMG file (no .bin found)", value=prev_name)
            selected_name = selected_label.strip()
        else:
            selected_label = st.selectbox(
                "EMG file",
                options=display_labels,
                index=(display_labels.index(prev_label) if prev_label in display_labels else 0),
            )
            selected_name = display_to_filename[selected_label]

        st.session_state["_input_name"] = selected_name
        meta["input_file"] = _join_data_path(meta.get("data_dir", ""), selected_name)

    with e:
        meta["researcher_id"] = st.text_input(
            "Researcher ID",
            value=str(meta.get("researcher_id", "")),
        )

    # -------------------------
    # Line 3: subject/session/sampling/hemi
    # -------------------------
    col1, col2, col3, col4 = st.columns([2.0, 1.0, 1.4, 1.8])
    with col1:
        meta["subj_id"] = st.text_input("Subject ID", value=str(meta.get("subj_id", "example_sub")))
    with col2:
        meta["session"] = st.number_input("Session", min_value=1, value=int(meta.get("session", 1)), step=1)
    with col3:
        meta["sampling_rate"] = st.number_input(
            "Sampling Rate (Hz)", value=int(meta.get("sampling_rate", 4000)), step=100
        )
    with col4:
        st.markdown("Hemisphere(s):")
        h1, h2 = st.columns(2)
        left = h1.checkbox("Left", value=("left" in meta.get("hemispheres", ["left"])))
        right = h2.checkbox("Right", value=("right" in meta.get("hemispheres", [])))
        meta["hemispheres"] = [h for h, v in (("left", left), ("right", right)) if v]

    # -------------------------
    # Advance
    # -------------------------
    if st.button("Advance ▶", type="primary"):
        tpl_raw = str(meta.get("template_file", "") or "").strip()
        tpl_path = resolve_template_path(tpl_raw)
        if not tpl_raw or not tpl_path.exists():
            st.error(f"Template file does not exist: {tpl_path}")
            st.stop()

        if meta.get("data_dir") and not os.path.isdir(str(meta["data_dir"])):
            st.error("Data directory does not exist.")
            st.stop()

        if not meta.get("input_file") or not os.path.exists(meta["input_file"]):
            st.error("EMG file does not exist (check Data directory + filename).")
            st.stop()

        if not meta.get("hemispheres"):
            st.error("Select at least one hemisphere.")
            st.stop()

        try:
            meta["output_dir"] = ensure_output_dir(meta.get("output_dir", ""), show_toast=True)
        except Exception:
            st.stop()

        template_name = Path(tpl_path).name
        input_name = str(st.session_state.get("_input_name") or Path(meta["input_file"]).name)

        save_persisted_defaults(
            template_name=template_name,
            input_name=input_name,
            output_dir=meta["output_dir"],
            data_dir=str(meta.get("data_dir", "")),
            researcher_id=str(meta.get("researcher_id", "")),
        )

        st.session_state.step = "confirmInputs"
        st.rerun()
