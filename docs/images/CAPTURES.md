# Image manifest

All documentation images live in this folder (`docs/images/`). Setup screenshots
are referenced from the [main README](../../README.md); GUI screenshots from the
[Quick Guide](../quickguide.md) and [User Guide](../user-guide.md).

## Setup (README) — Windows PowerShell + Miniconda Prompt

| Filename | Shows |
|----------|-------|
| `setup_01_git-clone.png` | PowerShell: `git lfs` + `git clone` commands and output. |
| `setup_02_1_conda-env.png` | Miniconda Prompt: the `conda create -n cfom_mep_preprocessing …` command. |
| `setup_02_2_conda-env.png` | Miniconda Prompt: the solver's "Proceed ([y]/n)?" prompt (answer `y`). |
| `setup_02_3_conda-env.png` | Miniconda Prompt: creation finishing + `conda activate`. |
| `setup_03_startapp.png` | Miniconda Prompt: activate, `cd` to repo, `streamlit run …` showing the Local URL. |

## GUI walkthrough (Quick Guide / User Guide)

| Filename | Shows |
|----------|-------|
| `gui_01_input.png` | Step 1 — Input, fields filled in. |
| `gui_02_confirm.png` | Step 2 — Confirm inputs summary. |
| `gui_03_segmentation.png` | Step 3 — Segmentation, blocks up to `bmeps` defined, range over `bmeps`. |
| `gui_04_mepwindow.png` | Step 4 — MEP window, overlay with a window selected. |
| `gui_05_peakchecking.png` | Step 5 — Peak-checking grid; MEPs 0–4 marked noise, one flagged for correction. |
| `gui_06_1_peakcorrecting.png` | Step 6 — Peak correction, two cursors placed incorrectly. |
| `gui_06_2_peakcorrecting.png` | Step 6 — same MEP, cursors corrected onto true min/max. |
| `gui_07_review.png` | Step 7 — Review & flag (final menu). |

## GUI extras

| Filename | Shows |
|----------|-------|
| `guiextra_01_segmentation.png` | Segmentation extra-controls menu: delete a timepoint / segment, adjust downsampling. |

## Guidance

- **Anonymise.** Crop or blank real subject IDs, file paths, and researcher names
  before committing; prefer the bundled `example_data.bin` for captures.
- **Updating docs.** If the UI changes, re-capture the affected screen with the
  **same filename** so the guides stay current without editing the Markdown.