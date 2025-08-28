"""
persistence.py
--------------
Centralized, safe JSON read/write and app-level persistence helpers.

- App defaults (.tms_emg_gui_settings.json)
- Subject/session file creation/patching (sub-<id>_ses-<n>.json)
- Writing segmentation ranges from the Bokeh UI
"""
from pathlib import Path
from json import JSONDecodeError
import json
from typing import Dict, List, Tuple

RUN_DIR = Path.cwd()
SETTINGS_FILE = RUN_DIR / ".tms_emg_gui_settings.json"
DEFAULT_TEMPLATE = str(RUN_DIR / "example_data/experiment_template.json")
DEFAULT_INPUT = str(RUN_DIR / "example_data/example_data.bin")

# ---------- low-level json helpers ----------
def _json_read(path: Path, fallback: dict) -> dict:
    if not path.exists():
        return fallback
    try:
        return json.loads(path.read_text())
    except (JSONDecodeError, OSError, ValueError):
        return fallback

def _json_write_atomic(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)

# ---------- app-level defaults ----------
def load_persisted_defaults():
    defaults = {"template_file": DEFAULT_TEMPLATE, "input_file": DEFAULT_INPUT}
    data = _json_read(SETTINGS_FILE, defaults)
    if not SETTINGS_FILE.exists():
        _json_write_atomic(SETTINGS_FILE, defaults)
    return data.get("template_file", DEFAULT_TEMPLATE), data.get("input_file", DEFAULT_INPUT)

def save_persisted_defaults(template_path, input_path):
    _json_write_atomic(SETTINGS_FILE, {
        "template_file": str(template_path),
        "input_file": str(input_path),
    })

# ---------- subject/session file scaffolding ----------
def _init_session_payload(meta, exp_structure: List[str]):
    hemis = list(meta.get("hemispheres", []))
    meps_parts = [p for p in exp_structure if "meps" in p.lower()]
    segmentation = {part: {h: [] for h in hemis} for part in exp_structure}
    for key in ("mh", "mvic"):
        segmentation.setdefault(key, {h: [] for h in hemis})
    return {
        "info": {
            "template_file": meta["template_file"],
            "input_file": meta["input_file"],
            "sampling_rate": meta["sampling_rate"],
            "session": meta["session"],
            "hemispheres": hemis,
        },
        "segmentation": segmentation,
        "meps": {part: {h: [] for h in hemis} for part in meps_parts},
        "mep_amplitudes": {part: {h: [] for h in hemis} for part in meps_parts},
    }

def _ensure_structure(data: dict, hemis: List[str], exp_structure: List[str]) -> dict:
    data.setdefault("info", {})
    data.setdefault("segmentation", {})
    data.setdefault("meps", {})
    data.setdefault("mep_amplitudes", {})
    for part in exp_structure:
        data["segmentation"].setdefault(part, {})
        for h in hemis:
            data["segmentation"][part].setdefault(h, [])
    for key in ("mh", "mvic"):
        data["segmentation"].setdefault(key, {})
        for h in hemis:
            data["segmentation"][key].setdefault(h, [])
    meps_parts = [p for p in exp_structure if "meps" in p.lower()]
    for part in meps_parts:
        data["meps"].setdefault(part, {})
        data["mep_amplitudes"].setdefault(part, {})
        for h in hemis:
            data["meps"][part].setdefault(h, [])
            data["mep_amplitudes"][part].setdefault(h, [])
    return data

def create_or_update_session_file(meta, exp_structure: List[str]) -> str:
    fname = f"sub-{meta['subj_id']}_ses-{meta['session']}.json"
    fpath = RUN_DIR / fname
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
        })
        data = _ensure_structure(data, hemis, exp_structure)
    else:
        data = _init_session_payload(meta, exp_structure)
    _json_write_atomic(fpath, data)
    return str(fpath)

# ---------- write segmentation from Bokeh UI ----------
def write_segmentation_ranges(
    session_file: str,
    block_ranges: Dict[str, Dict[str, Tuple[float, float]]],
    hemis: List[str],
    exp_structure: List[str],
) -> str:
    """
    Merge 'block_ranges' into the session JSON.

    block_ranges example:
      {
        "bmeps": {"left": (0.0, 240.0), "right": (0.0, 240.0)},
        "mvic":  {"left": (10.0, 30.0)}
      }

    Effect:
      segmentation[part][hemi] = [[start, end]]  # (overwrites that hemi for 'part')
    """
    fpath = Path(session_file)
    data = _json_read(fpath, {})
    hemis = list(hemis or data.get("info", {}).get("hemispheres", []))
    data = _ensure_structure(data, hemis, exp_structure)

    for part, hemi_map in (block_ranges or {}).items():
        data["segmentation"].setdefault(part, {})
        for h, rng in (hemi_map or {}).items():
            if rng is None:
                continue
            s, e = float(rng[0]), float(rng[1])
            # Store as list of pairs to allow future multiple ranges
            data["segmentation"][part][h] = [[s, e]]

    _json_write_atomic(fpath, data)
    return str(fpath)
