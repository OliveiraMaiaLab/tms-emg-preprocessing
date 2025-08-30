"""
persistence.py
--------------
Centralized JSON I/O and app/state bootstrap helpers.

- App defaults (.tms_emg_gui_settings.json)
- Subject/session file scaffolding (sub-<id>_ses-<n>.json) saved under output_dir
- Writing segmentation ranges from the Bokeh UI (FLAT schema)
- Bootstrap guards to hydrate session state when jumping to later steps

FLAT segmentation schema:
"segmentation": { "<block>": [[start, end]] | [] }
"""
from __future__ import annotations
import os
import json
from json import JSONDecodeError
from pathlib import Path
from typing import Dict, List

import numpy as np
import streamlit as st

RUN_DIR = Path.cwd()
SETTINGS_FILE = RUN_DIR / ".tms_emg_gui_settings.json"
DEFAULT_TEMPLATE = str(RUN_DIR / "experiment_template.json")
DEFAULT_INPUT = str(RUN_DIR / "example_data.bin")
DEFAULT_OUTPUT = str(RUN_DIR)

# ---------- low-level json helpers ----------
def _json_read(path: Path, fallback: dict) -> dict:
    if not path.exists():
        return fallback
    try:
        return json.loads(path.read_text())
    except (JSONDecodeError, OSError, ValueError):
        return fallback

def _json_write_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)

# ---------- app-level defaults ----------
def load_persisted_defaults():
    """
    Returns (template_file, input_file, output_dir).
    Falls back to sensible defaults if settings file is missing/corrupt.
    """
    defaults = {
        "template_file": DEFAULT_TEMPLATE,
        "input_file": DEFAULT_INPUT,
        "output_dir": DEFAULT_OUTPUT,
    }
    data = _json_read(SETTINGS_FILE, defaults)
    if not SETTINGS_FILE.exists():
        _json_write_atomic(SETTINGS_FILE, defaults)
    return (
        data.get("template_file", DEFAULT_TEMPLATE),
        data.get("input_file", DEFAULT_INPUT),
        data.get("output_dir", DEFAULT_OUTPUT),
    )

def save_persisted_defaults(template_path, input_path, output_dir):
    _json_write_atomic(
        SETTINGS_FILE,
        {
            "template_file": str(template_path),
            "input_file": str(input_path),
            "output_dir": str(output_dir),
        },
    )

# ---------- helpers for FLAT segmentation ----------
def _normalize_flat_range(value):
    """
    Normalize to [] or [[s, e]]:
      []                      -> []
      (s,e) / [s,e]           -> [[float(s), float(e)]]
      [[s,e]] or [[s,e], ...] -> [[float(s), float(e)]] (first pair kept)
    """
    if value is None:
        return []
    if isinstance(value, list) and len(value) == 0:
        return []
    if isinstance(value, (list, tuple)) and len(value) == 2 and all(isinstance(x, (int, float)) for x in value):
        s, e = float(value[0]), float(value[1])
        if s > e:
            s, e = e, s
        return [[s, e]]
    if isinstance(value, list) and value and isinstance(value[0], (list, tuple)) and len(value[0]) >= 2:
        s, e = value[0][0], value[0][1]
        try:
            s, e = float(s), float(e)
        except (TypeError, ValueError):
            return []
        if s > e:
            s, e = e, s
        return [[s, e]]
    return []

def _first_pair_from_legacy_hemi_map(hemi_map: dict) -> list:
    """
    Legacy: hemi_map = {hemi: [[s,e]] | []} -> returns [[s,e]] from first hemi with data, else [].
    """
    if not isinstance(hemi_map, dict):
        return []
    for _h, rngs in hemi_map.items():
        norm = _normalize_flat_range(rngs)
        if norm:
            return norm
    return []

def _migrate_segmentation_to_flat(seg: dict) -> dict:
    if not isinstance(seg, dict):
        return {}
    out = {}
    for part, val in seg.items():
        if isinstance(val, list):   # already flat
            out[part] = _normalize_flat_range(val)
        elif isinstance(val, dict): # legacy per-hemi
            out[part] = _first_pair_from_legacy_hemi_map(val)
        else:
            out[part] = []
    return out

# ---------- subject/session file scaffolding ----------
def _init_session_payload(meta, exp_structure: List[str]):
    hemis = list(meta.get("hemispheres", []))
    meps_parts = [p for p in exp_structure if "meps" in p.lower()]

    segmentation = {part: [] for part in exp_structure}
    for key in ("mh", "mvic"):
        segmentation.setdefault(key, [])

    return {
        "info": {
            "template_file": meta["template_file"],
            "input_file": meta["input_file"],
            "sampling_rate": meta["sampling_rate"],
            "session": meta["session"],
            "hemispheres": hemis,
            "output_dir": str(meta.get("output_dir", DEFAULT_OUTPUT)),
        },
        "segmentation": segmentation,
        # meps & mep_amplitudes remain hemi-nested
        "meps": {part: {h: [] for h in hemis} for part in meps_parts},
        "mep_amplitudes": {part: {h: [] for h in hemis} for part in meps_parts},
    }

def _ensure_structure(data: dict, hemis: List[str], exp_structure: List[str]) -> dict:
    data.setdefault("info", {})
    data.setdefault("segmentation", {})
    data.setdefault("meps", {})
    data.setdefault("mep_amplitudes", {})

    # migrate segmentation if legacy
    seg = data.get("segmentation", {})
    if isinstance(seg, dict) and any(isinstance(v, dict) for v in seg.values()):
        seg = _migrate_segmentation_to_flat(seg)
    elif not isinstance(seg, dict):
        seg = {}

    for part in exp_structure:
        seg.setdefault(part, [])
    for key in ("mh", "mvic"):
        seg.setdefault(key, [])
    data["segmentation"] = seg

    meps_parts = [p for p in exp_structure if "meps" in p.lower()]
    for part in meps_parts:
        data["meps"].setdefault(part, {})
        data["mep_amplitudes"].setdefault(part, {})
        for h in hemis:
            data["meps"][part].setdefault(h, [])
            data["mep_amplitudes"][part].setdefault(h, [])

    # ensure info.output_dir
    info = data.setdefault("info", {})
    info.setdefault("output_dir", info.get("output_dir", DEFAULT_OUTPUT))
    data["info"] = info

    return data

def ensure_output_dir(path: str, show_toast: bool = True) -> str:
    """
    Ensure the output directory exists. Creates it if missing.
    Shows a short-lived toast when created (if show_toast=True).
    Returns the absolute path as a string.
    """
    if not path:
        raise ValueError("Output directory path is empty.")

    p = Path(path).expanduser().resolve()
    existed = p.exists()
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        # Bubble up after surfacing the error in the UI
        try:
            st.error(f"Could not create output folder: {e}")
        except Exception:
            pass
        raise

    if show_toast and not existed:
        # Streamlit toasts auto-dismiss after a short while
        try:
            st.toast(f"Created output folder at: {p}", icon="📁")
        except Exception:
            # Fallback (non-ephemeral) if toast unavailable
            st.success(f"Created output folder at: {p}")

    return str(p)

def create_or_update_session_file(meta, exp_structure: List[str]) -> str:
    """
    Create/update session JSON at: <output_dir>/sub-<id>_ses-<n>.json
    Also writes info.output_dir in the file.
    """
    outdir = Path(meta.get("output_dir") or DEFAULT_OUTPUT)
    outdir.mkdir(parents=True, exist_ok=True)

    fname = f"sub-{meta['subj_id']}_ses-{meta['session']}.json"
    fpath = outdir / fname
    hemis = list(meta.get("hemispheres", []))

    if fpath.exists():
        data = _json_read(fpath, {})
        data.setdefault("info", {})
        data["info"].update({
            "template_file": meta["template_file"],
            "input_file": meta["input_file"],
            "sampling_rate": meta["sampling_rate"],
            "session": meta["session"],
            "hemispheres": hemis,
            "output_dir": str(outdir),
        })
        data = _ensure_structure(data, hemis, exp_structure)
    else:
        data = _init_session_payload(meta, exp_structure)

    _json_write_atomic(fpath, data)
    return str(fpath)

# ---------- write segmentation from Bokeh UI (FLAT) ----------
def write_segmentation_ranges(session_file: str,
                              block_ranges: Dict[str, object],
                              hemis: List[str] | None,
                              exp_structure: List[str]) -> str:
    """
    Overwrite only provided fields for FLAT schema:
      segmentation[part] = []       # clear
      segmentation[part] = [[s,e]]  # set one range
    """
    fpath = Path(session_file)
    data = _json_read(fpath, {})
    hemis_list = list(hemis or data.get("info", {}).get("hemispheres", []))
    data = _ensure_structure(data, hemis_list, exp_structure)

    seg = data.get("segmentation", {})
    for part, value in (block_ranges or {}).items():
        seg[part] = _normalize_flat_range(value)

    for part in exp_structure:
        seg.setdefault(part, [])
    for key in ("mh", "mvic"):
        seg.setdefault(key, [])

    data["segmentation"] = seg
    _json_write_atomic(fpath, data)
    return str(fpath)

# ---------- bootstrap guards ----------
def ensure_metadata() -> dict:
    """Ensure st.session_state.metadata exists and is populated with sane defaults."""
    if "metadata" not in st.session_state:
        tdef, idef, odef = load_persisted_defaults()
        st.session_state.metadata = {
            "template_file": tdef,
            "input_file": idef,
            "output_dir": odef,
            "sampling_rate": 4000,
            "subj_id": "example_sub",
            "session": 1,
            "hemispheres": ["left"],
            "_script_dir": os.getcwd(),
        }
    meta = st.session_state.metadata
    # fill gaps (handles partial/legacy sessions)
    if not meta.get("template_file"):
        meta["template_file"] = load_persisted_defaults()[0]
    if not meta.get("input_file"):
        meta["input_file"] = load_persisted_defaults()[1]
    if not meta.get("output_dir"):
        meta["output_dir"] = load_persisted_defaults()[2]
    meta.setdefault("sampling_rate", 4000)
    meta.setdefault("subj_id", "example_sub")
    meta.setdefault("session", 1)
    meta.setdefault("hemispheres", ["left"])
    meta.setdefault("_script_dir", os.getcwd())
    return meta

def ensure_template_loaded(meta: dict) -> dict:
    """Load experiment template -> exp_name, channels, exp_structure."""
    need = any(k not in meta for k in ("exp_name", "channels", "exp_structure"))
    if not need and isinstance(meta.get("channels"), np.ndarray):
        return meta

    tpath = meta.get("template_file")
    if not tpath or not os.path.exists(tpath):
        tdef, _, _ = load_persisted_defaults()
        if os.path.exists(tdef):
            meta["template_file"] = tdef
            tpath = tdef
    if not tpath or not os.path.exists(tpath):
        st.error("Template file not found. Please set it in the Input step.")
        st.session_state.step = "input"
        st.stop()

    with open(tpath, "r") as f:
        template = json.load(f)

    meta["exp_name"] = template["experiment_name"]
    meta["channels"] = np.array([
        template["channels"]["synch_pulse"],
        template["channels"]["right"],
        template["channels"]["left"],
    ])
    meta["exp_structure"] = template["experiment_structure"]
    return meta

def ensure_session_file(meta: dict) -> str:
    """Ensure sub-<id>_ses-<n>.json exists under output_dir and record its path."""
    if "_session_file" in st.session_state and st.session_state["_session_file"]:
        return st.session_state["_session_file"]
    if "exp_structure" not in meta:
        ensure_template_loaded(meta)
    path = create_or_update_session_file(meta, meta["exp_structure"])
    st.session_state["_session_file"] = path
    return path

def is_segmentation_complete(session_file: str,
                             hemis: list[str],  # kept for compatibility, unused
                             exp_structure: list[str]) -> bool:
    """
    FLAT schema completion check:
    True if segmentation[part] has at least one valid [start, end] pair
    for every part in exp_structure (plus 'mh' and 'mvic').
    """
    try:
        data = json.loads(Path(session_file).read_text())
    except Exception:
        return False

    seg = data.get("segmentation", {}) or {}
    parts_to_check = list(exp_structure) + ["mh", "mvic"]

    def _valid_flat(ranges):
        if not (isinstance(ranges, list) and ranges):
            return False
        first = ranges[0]
        if not (isinstance(first, (list, tuple)) and len(first) >= 2):
            return False
        try:
            s, e = float(first[0]), float(first[1])
        except (TypeError, ValueError):
            return False
        return s < e

    for part in parts_to_check:
        if not _valid_flat(seg.get(part, [])):
            return False

    return True
