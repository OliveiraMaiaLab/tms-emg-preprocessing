# Documentation — MEP Preprocessing Streamlit GUI

Welcome to the documentation for the TMS–EMG MEP preprocessing toolbox.

If you just want to get a file processed, start with the **Quick Guide**. For the
full picture — what every menu does, the structure of the output files, what each
quality-control flag means, and how to set up your own experiment — read the
**User Guide**.

| Document | Read this if you want to… |
|----------|---------------------------|
| [Quick Guide](quickguide.md) | run the pipeline end-to-end with the fewest steps |
| [User Guide](user-guide.md) | understand the menus, outputs, flags, and templates in depth |

For installation and environment setup, see the [main README](../README.md).

> **Screenshots:** image links in these guides point to files in
> [`images/`](images/). See [`images/CAPTURES.md`](images/CAPTURES.md) for the full
> image manifest (filenames and what each one shows).

---

## At a glance

The toolbox turns a raw TMS–EMG recording into a structured, quality-controlled
session file of MEP amplitudes, through a seven-screen workflow:

1. **Input** — choose the recording, paths, and metadata.
2. **Confirm inputs** — review and create the session file.
3. **Segmentation** — mark the time spans of each experiment block.
4. **MEP window** — define the window used to measure MEP amplitude.
5. **Peak checking** — flag noisy or mis-detected MEPs.
6. **Peak correction** — manually fix flagged MEP peaks.
7. **Review & flag** — mark the file for review (or not) and finish.

The result is one JSON file per subject/session in your output folder, plus an
entry in the processed-sessions registry.