# app/utils/persistence.py
"""
persistence.py
--------------
Centralized JSON I/O and app/state bootstrap helpers.

This module owns:
- config paths (config/.tms_emg_gui_settings.json, config/experiment_template.json,
  config/processed_sessions.json)
- persisted defaults (auto-upgrade)
- session file scaffolding + structure hydration
- processed sessions registry (in config/)
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from json import JSONDecodeError
from pathlib import Path
from typing import Dict, List

import numpy as np
import streamlit as st


# =============================================================================
# Project paths
# =============================================================================
# persistence.py is at: <root>/app/utils/persistence.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

SETTINGS_FILE = CONFIG_DIR / ".tms_emg_gui_settings.json"
PROCESSED_REGISTRY_FILE = CONFIG_DIR / "processed_sessions.json"
DEFAULT_TEMPLATE_PATH = CONFIG_DIR / "experiment_template.json"

# Reasonable project defaults
DEFAULT_DATA_DIR = str(PROJECT_ROOT / "example_data")
DEFAULT_OUTPUT_DIR = str(PROJECT_ROOT / "output")
DEFAULT_INPUT_FILE = ""  # filename (selected from data_dir)
DEFAULT_RESEARCHER_ID = ""
PIPELINE_VERSION = "1.0.0"


# =============================================================================
# Low-level JSON helpers
# =============================================================================
def _json_read(path: Path, fallback: dict) -> dict:
    if not path.exists():
        return fallback
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (JSONDecodeError, OSError, ValueError):
        return fallback


def _json_write_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


# =============================================================================
# Persisted GUI settings (.tms_emg_gui_settings.json)
# =============================================================================
def load_persisted_defaults() -> dict:
    """
    Returns persisted GUI defaults.

    Always returns a complete dict (new keys added if missing),
    and auto-upgrades older settings files.
    """
    defaults = {
        "template_file": DEFAULT_TEMPLATE_PATH.name,  # filename
        "input_file": DEFAULT_INPUT_FILE,             # filename
        "output_dir": DEFAULT_OUTPUT_DIR,             # path
        "data_dir": DEFAULT_DATA_DIR,                 # path
        "researcher_id": DEFAULT_RESEARCHER_ID,
        "version": PIPELINE_VERSION,
    }

    data = _json_read(SETTINGS_FILE, defaults)
    if not isinstance(data, dict):
        data = dict(defaults)

    upgraded = False
    for k, v in defaults.items():
        if k not in data:
            data[k] = v
            upgraded = True

    if not SETTINGS_FILE.exists() or upgraded:
        _json_write_atomic(SETTINGS_FILE, data)

    return data


def save_persisted_defaults(
    *,
    template_name: str,
    input_name: str,
    output_dir: str,
    data_dir: str,
    researcher_id: str,
) -> None:
    payload = load_persisted_defaults()
    payload.update(
        {
            "template_file": str(template_name),
            "input_file": str(input_name),
            "output_dir": str(output_dir),
            "data_dir": str(data_dir),
            "researcher_id": str(researcher_id),
            "version": PIPELINE_VERSION,
        }
    )
    _json_write_atomic(SETTINGS_FILE, payload)


# =============================================================================
# Processed sessions registry (config/processed_sessions.json)
# =============================================================================
def _read_processed_registry(path: Path) -> dict:
    empty = {"version": 1, "processed": []}
    if not path.exists():
        return empty

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        # Backup corrupt file (best effort) and start fresh
        try:
            bak = path.with_suffix(path.suffix + ".corrupt")
            bak.write_text(path.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")
        except Exception:
            pass
        return empty

    if not isinstance(data, dict):
        return empty

    data.setdefault("version", 1)
    data.setdefault("processed", [])
    if not isinstance(data["processed"], list):
        data["processed"] = []

    # Normalize records (best effort)
    normed = []
    for rec in data["processed"]:
        if not isinstance(rec, dict):
            continue
        if "data_file" not in rec:
            continue
        normed.append(
            {
                "data_file": str(rec.get("data_file", "")),
                "session_file": str(rec.get("session_file", "")),
                "finished_at": str(rec.get("finished_at", "")),
                "researcher_id": str(rec.get("researcher_id", "")),
                "pipeline_version": str(rec.get("pipeline_version", "")),
            }
        )
    data["processed"] = normed
    return data


def _write_processed_registry_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def load_processed_sessions_registry() -> dict:
    """Convenience wrapper used by Step 1 to color + sort dropdown entries."""
    return _read_processed_registry(PROCESSED_REGISTRY_FILE)


def update_processed_sessions_registry(
    *,
    data_file: str | Path,
    session_file: str | Path,
    researcher_id: str = "",
    pipeline_version: str = "",
) -> Path:
    """
    Upsert mapping: data filename -> session filename into config/processed_sessions.json
    """
    reg_path = PROCESSED_REGISTRY_FILE

    data_name = Path(data_file).name
    sess_name = Path(session_file).name
    finished_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    reg = _read_processed_registry(reg_path)

    for rec in reg["processed"]:
        if isinstance(rec, dict) and rec.get("data_file") == data_name:
            rec["session_file"] = sess_name
            rec["finished_at"] = finished_at
            rec["researcher_id"] = str(researcher_id or "")
            rec["pipeline_version"] = str(pipeline_version or "")
            _write_processed_registry_atomic(reg_path, reg)
            return reg_path

    reg["processed"].append(
        {
            "data_file": data_name,
            "session_file": sess_name,
            "finished_at": finished_at,
            "researcher_id": str(researcher_id or ""),
            "pipeline_version": str(pipeline_version or ""),
        }
    )
    _write_processed_registry_atomic(reg_path, reg)
    return reg_path


# =============================================================================
# Template path helpers (template is a filename in config/)
# =============================================================================
def resolve_template_path(template_name_or_path: str) -> Path:
    """
    Accepts either:
    - full path to a template file, OR
    - filename stored in config/

    Returns a Path (may or may not exist; caller can validate).
    """
    p = Path(str(template_name_or_path))
    if p.is_absolute() and p.exists():
        return p

    # if user already passed something like "config/experiment_template.json"
    if p.exists():
        return p.resolve()

    # otherwise interpret as config filename
    return (CONFIG_DIR / p.name).resolve()


# =============================================================================
# Flat segmentation helpers + meps schema hydration
# =============================================================================
def _normalize_flat_range(value):
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

    base = _blank_mep_hemi_payload()
    for k, v in base.items():
        payload.setdefault(k, v)

    for k in base.keys():
        if not isinstance(payload.get(k), list):
            payload[k] = []

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


# =============================================================================
# Session file scaffolding
# =============================================================================
def _init_session_payload(meta: dict, exp_structure: List[str]) -> dict:
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
            "template_file": str(meta.get("template_file", "")),  # filename or path
            "input_file": str(meta.get("input_file", "")),        # filename or path
            "data_dir": str(meta.get("data_dir", "")),
            "sampling_rate": int(meta.get("sampling_rate", 4000)),
            "session": int(meta.get("session", 1)),
            "hemispheres": hemis,
            "output_dir": str(meta.get("output_dir", DEFAULT_OUTPUT_DIR)),
            "researcher_id": str(meta.get("researcher_id", "")),
            "pipeline_version": str(meta.get("version", "")),
        },
        "segmentation": segmentation,
        "mep_window": [None, None],
        "meps": meps_root,
    }


def _ensure_structure(data: dict, hemis: List[str], exp_structure: List[str]) -> dict:
    data.setdefault("info", {})
    data.setdefault("segmentation", {})
    data.setdefault("mep_window", [None, None])
    data.setdefault("meps", {})

    seg = data.get("segmentation")
    if not isinstance(seg, dict):
        seg = {}
    for part in exp_structure:
        seg.setdefault(part, [])
    for key in ("mh", "mvic"):
        seg.setdefault(key, [])
    for part in list(seg.keys()):
        seg[part] = _normalize_flat_range(seg.get(part))
    data["segmentation"] = seg

    meps_root = data.get("meps")
    if not isinstance(meps_root, dict):
        meps_root = {}
    mep_blocks = [p for p in exp_structure if str(p).lower().endswith("meps")]
    for b in mep_blocks:
        for h in hemis:
            _ensure_meps_block_hemi(meps_root, b, h)
    data["meps"] = meps_root

    info = data.setdefault("info", {})
    info.setdefault("output_dir", str(DEFAULT_OUTPUT_DIR))
    info.setdefault("researcher_id", "")
    info.setdefault("pipeline_version", "")
    data["info"] = info

    return data


def ensure_output_dir(path: str, show_toast: bool = True) -> str:
    if not path:
        raise ValueError("Output directory path is empty.")

    p = Path(path).expanduser().resolve()
    existed = p.exists()
    p.mkdir(parents=True, exist_ok=True)

    if show_toast and not existed:
        try:
            st.toast(f"Created output folder at: {p}", icon="📁")
        except Exception:
            st.success(f"Created output folder at: {p}")

    return str(p)


def create_or_update_session_file(meta: dict, exp_structure: List[str]) -> str:
    outdir = Path(meta.get("output_dir") or DEFAULT_OUTPUT_DIR)
    outdir.mkdir(parents=True, exist_ok=True)

    fname = f"sub-{meta['subj_id']}_ses-{meta['session']}.json"
    fpath = outdir / fname
    hemis = list(meta.get("hemispheres", []))

    if fpath.exists():
        data = _json_read(fpath, {})
        data.setdefault("info", {})
        data["info"].update(
            {
                "template_file": str(meta.get("template_file", "")),
                "input_file": str(meta.get("input_file", "")),
                "data_dir": str(meta.get("data_dir", "")),
                "sampling_rate": int(meta.get("sampling_rate", 4000)),
                "session": int(meta.get("session", 1)),
                "hemispheres": hemis,
                "output_dir": str(outdir),
                "researcher_id": str(meta.get("researcher_id", "")),
                "pipeline_version": str(meta.get("version", "")),
            }
        )
        data = _ensure_structure(data, hemis, exp_structure)
    else:
        data = _init_session_payload(meta, exp_structure)

    _json_write_atomic(fpath, data)
    return str(fpath)


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


# =============================================================================
# Bootstrap guards
# =============================================================================
def ensure_metadata() -> dict:
    if "metadata" not in st.session_state:
        defaults = load_persisted_defaults()
        st.session_state.metadata = {
            "template_file": defaults.get("template_file", DEFAULT_TEMPLATE_PATH.name),
            "input_file": defaults.get("input_file", DEFAULT_INPUT_FILE),
            "output_dir": defaults.get("output_dir", DEFAULT_OUTPUT_DIR),
            "data_dir": defaults.get("data_dir", DEFAULT_DATA_DIR),
            "researcher_id": defaults.get("researcher_id", DEFAULT_RESEARCHER_ID),
            "version": defaults.get("version", PIPELINE_VERSION),
            "sampling_rate": 4000,
            "subj_id": "example_sub",
            "session": 1,
            "hemispheres": ["left"],
            "_script_dir": str(PROJECT_ROOT),
        }

    meta = st.session_state.metadata
    defaults = load_persisted_defaults()

    meta.setdefault("template_file", defaults.get("template_file", DEFAULT_TEMPLATE_PATH.name))
    meta.setdefault("input_file", defaults.get("input_file", DEFAULT_INPUT_FILE))
    meta.setdefault("output_dir", defaults.get("output_dir", DEFAULT_OUTPUT_DIR))
    meta.setdefault("data_dir", defaults.get("data_dir", DEFAULT_DATA_DIR))
    meta.setdefault("researcher_id", defaults.get("researcher_id", DEFAULT_RESEARCHER_ID))
    meta.setdefault("version", defaults.get("version", PIPELINE_VERSION))

    meta.setdefault("sampling_rate", 4000)
    meta.setdefault("subj_id", "example_sub")
    meta.setdefault("session", 1)
    meta.setdefault("hemispheres", ["left"])
    meta.setdefault("_script_dir", str(PROJECT_ROOT))

    return meta


def _template_requests_emg_ref(template: dict) -> bool:
    other = (template.get("other") or {})
    return str(other.get("include_rest_emg_ref", "")).strip().lower() == "yes"


def ensure_template_loaded(meta: dict) -> dict:
    """
    Load experiment template -> exp_name, channels, exp_structure
    and enforce emg_ref if requested in template.other.include_rest_emg_ref == "yes".
    """
    # meta["template_file"] can be filename or path
    tpath = resolve_template_path(str(meta.get("template_file", "")))

    if not tpath.exists():
        st.error(f"Template file not found: {tpath}")
        st.session_state.step = "input"
        st.stop()

    already_loaded = (
        isinstance(meta.get("channels"), np.ndarray)
        and isinstance(meta.get("exp_structure"), list)
        and meta.get("exp_name") is not None
    )

    if already_loaded:
        # minimal enforcement
        try:
            template = json.loads(tpath.read_text(encoding="utf-8"))
        except Exception:
            return meta

        exp_structure = list(meta.get("exp_structure", []))
        if _template_requests_emg_ref(template) and "emg_ref" not in exp_structure:
            exp_structure.append("emg_ref")
            meta["exp_structure"] = exp_structure
        return meta

    # full load
    template = json.loads(tpath.read_text(encoding="utf-8"))

    meta["exp_name"] = template["experiment_name"]
    meta["channels"] = np.array(
        [
            template["channels"]["synch_pulse"],
            template["channels"]["right"],
            template["channels"]["left"],
        ]
    )

    exp_structure = list(template["experiment_structure"])
    if _template_requests_emg_ref(template) and "emg_ref" not in exp_structure:
        exp_structure.append("emg_ref")

    meta["exp_structure"] = exp_structure
    return meta


def ensure_session_file(meta: dict) -> str:
    if st.session_state.get("_session_file"):
        return st.session_state["_session_file"]

    if "exp_structure" not in meta:
        ensure_template_loaded(meta)

    path = create_or_update_session_file(meta, meta["exp_structure"])
    st.session_state["_session_file"] = path
    return path
