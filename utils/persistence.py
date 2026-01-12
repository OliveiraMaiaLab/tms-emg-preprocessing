"""
persistence.py
--------------
Centralized JSON I/O and app/state bootstrap helpers.

Session schema (relevant bits):

{
  "info": {...},
  "segmentation": { "<block>": [[start_s, end_s]] | [] },
  "mep_window": [beg_s, end_s],   # relative to pulse (seconds)
  "meps": {
    "<block>": {
      "<hemi>": {
        "pulses": [...],               # list[int] (sample indices)
        "preactivation_flag": [...],   # list[int] 0/1
        "min": [[min1_ms, min1_val],...],                  # list[float|None]
        "max": [[max1_ms, max1_val]...],                  # list[float|None]
        "peaks_flag": [...]            # list[int] 0/1
      }
    }
  }
}
"""
from __future__ import annotations

import os
import json
from json import JSONDecodeError
from pathlib import Path
from typing import Dict, List, Tuple, Any

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
    Normalize to [] or [[s, e]].
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


def _blank_mep_hemi_payload() -> dict:
    return {
        "pulses": [],
        "min": [],
        "max": [],
        "hinder_preactivation_flag": [],
        "std_preactivation_flag": [],
        "peaks_flag": [],
    }


def _ensure_meps_block_hemi(meps_root: dict, block: str, hemi: str) -> None:
    meps_root.setdefault(block, {})
    payload = meps_root[block].get(hemi)
    if not isinstance(payload, dict):
        payload = {}
    # ensure keys exist
    base = _blank_mep_hemi_payload()
    for k, v in base.items():
        payload.setdefault(k, v)

    # normalize list types
    for k in ("pulses", "min", "max", "hinder_preactivation_flag", "std_preactivation_flag", "peaks_flag"):
        if not isinstance(payload.get(k), list):
            payload[k] = []

    # if pulses exist, ensure same length for the parallel arrays
    n = len(payload["pulses"])
    def _fit_list(name: str, fill):
        arr = payload[name]
        if len(arr) < n:
            arr.extend([fill] * (n - len(arr)))
        elif len(arr) > n:
            payload[name] = arr[:n]

    _fit_list("hinder_preactivation_flag", 0)
    _fit_list("std_preactivation_flag", 0)
    _fit_list("peaks_flag", 0)
    _fit_list("min", None)
    _fit_list("max", None)

    meps_root[block][hemi] = payload


# ---------- subject/session file scaffolding ----------
def _init_session_payload(meta, exp_structure: List[str]):
    hemis = list(meta.get("hemispheres", []))
    mep_blocks = [p for p in exp_structure if str(p).lower().endswith("meps")]

    segmentation = {part: [] for part in exp_structure}
    for key in ("mh", "mvic"):
        segmentation.setdefault(key, [])

    meps_root = {}
    for b in mep_blocks:
        meps_root[b] = {}
        for h in hemis:
            meps_root[b][h] = _blank_mep_hemi_payload()

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
        "mep_window": [None, None],  # relative to pulse in seconds
        "meps": meps_root,
    }


def _ensure_structure(data: dict, hemis: List[str], exp_structure: List[str]) -> dict:
    data.setdefault("info", {})
    data.setdefault("segmentation", {})
    data.setdefault("mep_window", [None, None])
    data.setdefault("meps", {})

    # segmentation keys
    seg = data.get("segmentation")
    if not isinstance(seg, dict):
        seg = {}
    for part in exp_structure:
        seg.setdefault(part, [])
    for key in ("mh", "mvic"):
        seg.setdefault(key, [])
    # normalize values
    for part in list(seg.keys()):
        seg[part] = _normalize_flat_range(seg.get(part))
    data["segmentation"] = seg

    # meps keys
    meps_root = data.get("meps")
    if not isinstance(meps_root, dict):
        meps_root = {}
    mep_blocks = [p for p in exp_structure if str(p).lower().endswith("meps")]
    for b in mep_blocks:
        for h in hemis:
            _ensure_meps_block_hemi(meps_root, b, h)
    data["meps"] = meps_root

    # info.output_dir
    info = data.setdefault("info", {})
    info.setdefault("output_dir", info.get("output_dir", DEFAULT_OUTPUT))
    data["info"] = info

    return data


def ensure_output_dir(path: str, show_toast: bool = True) -> str:
    if not path:
        raise ValueError("Output directory path is empty.")

    p = Path(path).expanduser().resolve()
    existed = p.exists()
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        try:
            st.error(f"Could not create output folder: {e}")
        except Exception:
            pass
        raise

    if show_toast and not existed:
        try:
            st.toast(f"Created output folder at: {p}", icon="📁")
        except Exception:
            st.success(f"Created output folder at: {p}")

    return str(p)


def create_or_update_session_file(meta, exp_structure: List[str]) -> str:
    outdir = Path(meta.get("output_dir") or DEFAULT_OUTPUT)
    outdir.mkdir(parents=True, exist_ok=True)

    fname = f"sub-{meta['subj_id']}_ses-{meta['session']}.json"
    fpath = outdir / fname
    hemis = list(meta.get("hemispheres", []))

    if fpath.exists():
        data = _json_read(fpath, {})
        data.setdefault("info", {})
        data["info"].update(
            {
                "template_file": meta["template_file"],
                "input_file": meta["input_file"],
                "sampling_rate": meta["sampling_rate"],
                "session": meta["session"],
                "hemispheres": hemis,
                "output_dir": str(outdir),
            }
        )
        data = _ensure_structure(data, hemis, exp_structure)
    else:
        data = _init_session_payload(meta, exp_structure)

    _json_write_atomic(fpath, data)
    return str(fpath)


# ---------- write segmentation from Bokeh UI (FLAT) ----------
def write_segmentation_ranges(
    session_file: str,
    block_ranges: Dict[str, object],
    hemis: List[str] | None,
    exp_structure: List[str],
) -> str:
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

def _template_requests_emg_ref(template: dict) -> bool:
    other = (template.get("other") or {})
    return str(other.get("include_rest_emg_ref", "")).strip().lower() == "yes"



def ensure_template_loaded(meta: dict) -> dict:
    """Load experiment template -> exp_name, channels, exp_structure (and enforce emg_ref if requested)."""
    tpath = meta.get("template_file")

    # Resolve template path if missing/bad
    if not tpath or not os.path.exists(tpath):
        tdef, _, _ = load_persisted_defaults()
        if os.path.exists(tdef):
            meta["template_file"] = tdef
            tpath = tdef

    if not tpath or not os.path.exists(tpath):
        st.error("Template file not found. Please set it in the Input step.")
        st.session_state.step = "input"
        st.stop()

    # If template already loaded, still enforce emg_ref requirement
    already_loaded = (
        isinstance(meta.get("channels"), np.ndarray)
        and isinstance(meta.get("exp_structure"), list)
        and meta.get("exp_name") is not None
    )

    if already_loaded:
        # Minimal read: just to check include_rest_emg_ref
        try:
            with open(tpath, "r") as f:
                template = json.load(f)
        except Exception:
            return meta

        exp_structure = list(meta.get("exp_structure", []))
        if _template_requests_emg_ref(template) and "emg_ref" not in exp_structure:
            exp_structure.append("emg_ref")
            meta["exp_structure"] = exp_structure
        return meta

    # Otherwise fully load template
    with open(tpath, "r") as f:
        template = json.load(f)

    meta["exp_name"] = template["experiment_name"]
    meta["channels"] = np.array([
        template["channels"]["synch_pulse"],
        template["channels"]["right"],
        template["channels"]["left"],
    ])

    exp_structure = list(template["experiment_structure"])
    if _template_requests_emg_ref(template) and "emg_ref" not in exp_structure:
        exp_structure.append("emg_ref")

    meta["exp_structure"] = exp_structure
    return meta



def ensure_session_file(meta: dict) -> str:
    if "_session_file" in st.session_state and st.session_state["_session_file"]:
        return st.session_state["_session_file"]
    if "exp_structure" not in meta:
        ensure_template_loaded(meta)
    path = create_or_update_session_file(meta, meta["exp_structure"])
    st.session_state["_session_file"] = path
    return path


def is_segmentation_complete(session_file: str, hemis: list[str], exp_structure: list[str]) -> bool:
    try:
        data = json.loads(Path(session_file).read_text())
    except Exception:
        return False

    seg = data.get("segmentation", {}) or {}
    parts_to_check = list(exp_structure)

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

    return all(_valid_flat(seg.get(part, [])) for part in parts_to_check)
