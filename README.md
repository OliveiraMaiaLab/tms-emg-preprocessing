# MEP Preprocessing — Streamlit GUI (dev branch)

**Branch:** `streamlit-gui-dev`
**Status:** *Developer preview (actively changing). Expect breaking changes.*

A **GUI-driven pipeline** for preprocessing **motor evoked potentials (MEPs)**
acquired with TMS. It wraps the signal-processing steps from the main pipeline
into an interactive **Streamlit** app that guides you from file selection through
quality control to export. Some steps embed **Bokeh server** apps (large
time-series exploration, RangeTools) and **Dash/Plotly** views for richer
interaction.

> This GUI is intended to replace the older step-by-step guide. The previous
> implementation remains available in `old_pipeline/` on the `main` branch.

---

## 1. Prerequisites

| Tool | Why | Link |
|------|-----|------|
| **Git** | clone the repository | https://git-scm.com/downloads |
| **Git LFS** | the example `.bin` data is stored via Git LFS | https://git-lfs.com |
| **Miniconda** (recommended) | create the Python environment | https://www.anaconda.com/download/success |

> A pure-`pip`/`venv` route is also provided (Option B below) if you'd rather
> not use Conda.

### Install Miniconda

- **Windows** — download and run the *Miniconda3 Windows 64-bit* installer, then
  use the **"Anaconda Prompt"** for the commands below.
- **macOS / Linux** — download the installer for your platform and run it, e.g.:
  ```bash
  # macOS (Apple Silicon shown; pick the matching installer for your machine)
  bash Miniconda3-latest-MacOSX-arm64.sh
  # Linux
  bash Miniconda3-latest-Linux-x86_64.sh
  ```
  Restart your shell afterwards so `conda` is on the `PATH`.

---

## 2. Clone the repository

```bash
git lfs install        # one-time, so the LFS-tracked .bin example data downloads
git clone <your-repo-url> tms-emg-preprocessing
cd tms-emg-preprocessing
```

> The example dataset (`example_data/*.bin`) is large (~110 MB each). If you
> don't need it, you can skip LFS content with
> `GIT_LFS_SKIP_SMUDGE=1 git clone <your-repo-url>`.

---

## 3. Set up the environment

This project ships three environment specifications:

| File | Purpose | Platform |
|------|---------|----------|
| **`environment.yml`** | curated, portable Conda spec (versions pinned, no build hashes) | Windows / macOS / Linux |
| **`requirements-pip.txt`** | pure-pip install path (no Conda) | Windows / macOS / Linux |
| **`conda-win64-explicit.txt`** | exact, build-locked clone of the lab machine | **Windows only** |

### ▶️ Option A — Conda from `environment.yml` *(recommended, cross-platform)*

```bash
conda env create -f environment.yml
conda activate cfom_mep_preprocessing
```

To update an existing env after the spec changes:
```bash
conda env update -f environment.yml --prune
```

### ▶️ Option B — Plain virtual environment (pip, no Conda)

```bash
python -m venv .venv
# Windows:        .venv\Scripts\activate
# macOS / Linux:  source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements-pip.txt
```

### ▶️ Option C — Exact Windows clone *(maximum reproducibility, win-64 only)*

```bash
conda create -n cfom_mep_preprocessing --file conda-win64-explicit.txt
conda activate cfom_mep_preprocessing
```
Use this only if you need to match the original lab machine bit-for-bit
(identical Conda builds). It will **not** solve on macOS or Linux.

---

## 4. Run the app

From the repository root, with the environment activated:

```bash
streamlit run app/main_gui.py
```

Streamlit will print a local URL (default http://localhost:8501) and open it in
your browser.


## Package organization

```
app/
├── __init__.py
├── main_gui.py            # entry point / step router (run this)
├── utils/
│   ├── persistence.py
│   ├── tms_module.py
│   ├── layout.py
│   ├── dash_peak_editor.py
│   └── peak_checking_io.py
├── bk_embedding/
│   ├── segmentation.py
│   └── mepOverlap.py
└── steps/
    ├── step_input.py
    ├── step_confirmInputs.py
    ├── step_segmentation.py
    ├── step_mepWindow.py
    ├── step_peakChecking.py
    ├── step_peakCorrection.py
    └── step_reviewFlag.py
```

See `ARCHUTECTURE.md` for the session-file schema and per-module responsibilities.