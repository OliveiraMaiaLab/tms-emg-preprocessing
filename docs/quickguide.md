# Quick Guide

A step-by-step walkthrough to process one TMS–EMG recording from start to finish.
This assumes the environment is installed and the GUI settings are configured — if
not, see the [main README](../README.md) sections 3 and 4 first.

For deeper explanations of any screen, the output files, or the flags, see the
[User Guide](user-guide.md).

---

## Before you start

You need:
- a raw EMG recording (`.bin`) in your **data folder**,
- an **experiment template** that matches how the recording was acquired
  (channel numbers and block order) — the bundled `experiment_template.json`
  works for the PDM-TMS layout,
- a writable **output folder**.

Launch the app from the repository root with the environment activated:

```bash
streamlit run app/main_gui.py
```

Your browser opens at `http://localhost:8501`.

---

## Step 1 — Input

![Input screen](images/gui_01_input.png)

Fill in, from top to bottom:

1. **Experiment template file** — leave as `experiment_template.json` unless you
   made your own (see the User Guide).
2. **Data directory** — folder containing your `.bin` recordings.
3. **Output folder** — where session files are written (created if missing).
4. **EMG file** — pick your recording from the dropdown (it lists the `.bin`
   files found in the data directory).
5. **Researcher ID** — your initials; stamped into the output for provenance.
6. **Subject ID**, **Session**, **Sampling Rate (Hz)** (usually `4000`),
   **Hemisphere(s)** — `left`, `right`, or both.

Click **Next**.

---

## Step 2 — Confirm inputs

![Confirm screen](images/gui_02_confirm.png)

Check the summary. If anything is wrong, go **Back**. When correct, **Confirm** —
this creates (or updates) the session file
`sub-<SubjectID>_ses-<Session>.json` in your output folder.

---

## Step 3 — Segmentation

![Segmentation screen](images/gui_03_segmentation.png)

You see the whole recording in an interactive plot.

1. In **Active blocks**, keep only the blocks present in this recording.
2. For each block, **drag-select the time span** it occupies in the recording
   (e.g. the baseline-MEPs block, each post-intervention block, and — if your
   template uses it — the resting-EMG reference span).
3. Click **Update** to apply block changes (this refreshes the viewer and clears
   ranges for any block you switched off).

When every active block has a span, click **Next**. The app detects the TMS
pulses inside each MEP block and computes the pre-activation flags.

The segmentation view also has an **extra controls** menu for deleting a
timepoint or a whole segment and for adjusting the plot **downsampling** (lower it
to draw long recordings faster):

![Segmentation extra controls](images/guiextra_01_segmentation.png)

---

## Step 4 — MEP window

![MEP window screen](images/gui_04_mepwindow.png)

All detected MEPs are overlaid, aligned to the pulse.

1. **Drag-select the window** (relative to the pulse) over which the MEP
   peak-to-peak amplitude should be measured — typically the ~20–60 ms
   post-pulse deflection.
2. Click **Next**. The app stores the window and computes each MEP's min and max
   inside it.

---

## Step 5 — Peak checking

![Peak checking screen](images/gui_05_peakchecking.png)

MEPs are shown as a grid, per block and hemisphere, with the detected peaks
marked. For each MEP you can set two checkboxes:

- **Noise** — the trial is unusable. Excludes it and disables its correction.
- **Needs correction** — the peaks were detected in the wrong place; you'll fix
  them in the next step.

Use **Select all / Deselect all** to set the noise flag for a whole page at once.
Click **Next** when done.

---

## Step 6 — Peak correction

Only MEPs you marked **Needs correction** (and not noise) appear here.

1. **Drag the two cursors** onto the true min and max of the MEP.
2. The **peak-to-peak amplitude updates live** as you drag.
3. Move through each flagged MEP, then click **Next**.

Before — cursors not yet on the true peaks:

![Peak correction, cursors placed incorrectly](images/gui_06_1_peakcorrecting.png)

After — cursors moved onto the real min and max:

![Peak correction, cursors placed correctly](images/gui_06_2_peakcorrecting.png)

---

## Step 7 — Review & flag

![Review screen](images/gui_07_review.png)

Decide whether this file needs a second look before it goes into analysis:

- Tick **Flag for review** and add a note if something was uncertain, **or**
- Leave it unticked if you're confident.

Click **Finish**. The app saves the flag, records the file in the
processed-sessions registry, and returns to the Input screen ready for the next
recording.

---

## What you get

In your output folder:
- `sub-<SubjectID>_ses-<Session>.json` — the full session file (segmentation,
  MEP window, per-MEP amplitudes and flags, review status). Its structure is
  documented in the [User Guide](user-guide.md#output-data-structure).

You can now load that JSON in your analysis scripts to pull MEP amplitudes and
exclude flagged trials.