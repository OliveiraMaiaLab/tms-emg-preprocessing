# Architecture

This project is a Streamlit-based GUI for preprocessing TMS–EMG motor evoked potentials (MEPs). The GUI guides the user through a sequence of steps and stores the processing state in a single JSON “session file” per subject/session.

The architecture is intentionally simple:

- **Streamlit** orchestrates steps and navigation.
- **Bokeh server apps** are embedded inside some steps for richer interaction (large time-series exploration, RangeTools).
- A **session JSON** is the single source of truth for all user selections and derived artifacts.

---

## Repository layout


```
── main_gui.py
── utils/
 ├── persistence.py
 ├── tms_module.py
 ├── layout.py
 ├── mep_loading.py
 └── peak_checking_io.py
── bk_embedding/
 ├── segmentation.py
 └── mepOverlap.py
── steps/
── step_input.py
── step_confirmInputs.py
── step_segmentation.py
── step_mepWindow.py
── step_peakChecking.py
```


---

## Core concepts

### Session file (single source of truth)

Each subject/session has exactly one JSON file:

  sub-<subject>_ses-<n>.json

This file stores:
- Metadata (input paths, sampling rate, hemispheres)
- Segmentation windows
- MEP pulse indices
- Derived MEP metrics and flags

This file is created and maintained exclusively by utils/persistence.py.

---

SESSION SCHEMA (RELEVANT FIELDS)
-------------------------------

info:
  template_file
  input_file
  sampling_rate
  session
  hemispheres
  output_dir

segmentation:
  <block>: [[start_s, end_s]]

mep_window:
  [start_s, end_s]   (relative to pulse, seconds)

meps:
  <block>:
    <hemi>:
      pulses             : list[int]
      preactivation_flag : list[int] (0/1)
      min                : list[float or null]
      max                : list[float or null]
      peaks_flag         : list[int] (0/1)

---

PIPELINE STEPS
--------------

1) Input
   - Select template, input file, output directory
   - Define subject/session identifiers

2) Confirm Inputs
   - Load experiment template
   - Resolve channel mappings and experiment structure

3) Segmentation
   - Define time windows per experimental block
   - Derive TMS pulse indices per MEP block
   - Write pulses to session["meps"][block][hemi]["pulses"]

4) MEP Window
   - Visualize MEP epochs aligned to the pulse
   - Select a time window relative to pulse (RangeTool)
   - Persist window to session["mep_window"]
   - Compute and store min/max per MEP

5) Peak Checking
   - Plot individual MEPs (no overlap)
   - Mark epoch bounds and min/max
   - Allow manual flagging of bad MEPs
   - Store flags in session["meps"][block][hemi]["peaks_flag"]

---

RESPONSIBILITIES BY MODULE
--------------------------

utils/persistence.py
  - Session JSON creation and validation
  - Schema enforcement
  - Step completion checks

utils/tms_module.py
  - Load raw EMG data
  - Detect TMS pulse indices from sync channel
  - Extract per-block MEP waveforms from raw data

bk_embedding/segmentation.py
  - Interactive visualization for segmentation

bk_embedding/mepOverlap.py
  - Overlay MEP epochs
  - RangeTool for MEP window selection

utils/mep_loading.py

utils/peak_checking_io.py
  - Read/write peak flags and epoch definition

---

UNITS AND CONVENTIONS
--------------------

- Segmentation windows: seconds (absolute time)
- Pulses: sample indices
- MEP window: seconds relative to pulse
- Plotting: milliseconds (visual only)

---

DESIGN NOTES
------------

- Steps only write the data they own
- No step infers missing upstream data
- Session JSON is the only persisted state
- Bokeh apps communicate selections via explicit sinks

---
