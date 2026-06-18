# MEP Preprocessing — Streamlit GUI

This repository provides a structured preprocessing pipeline for **motor evoked
potentials (MEPs)** acquired with **transcranial magnetic stimulation (TMS)**.

The workflow includes:

- **Signal denoising** with a three-level adaptive wavelet filter (Daubechies
  family, db1) and BayesShrink soft thresholding.
- **Smoothing** with a Savitzky–Golay filter, chosen over conventional band-pass
  filtering to minimize distortions.
- **Automated screening** of trials based on amplitude thresholds and background
  EMG activity, with criteria adapted from established literature.

The pipeline is designed to facilitate amplitude extraction and quality control
of MEPs with minimal coding expertise. It combines automated detection with
guided manual correction to ensure reliable results.

The processing is delivered through an interactive **Streamlit** app; some steps
embed **Bokeh server** apps (large time-series exploration, RangeTools) and
**Dash/Plotly** views for richer interaction. A step-by-step guide supports file
selection, EMG segmentation, epoch definition, visual inspection, and correction
of flagged trials.

### Citation

This pipeline has been used and described in:

- Faro Viana F, et al. *Reducing motor evoked potential amplitude variability
  through normalization.* Front Psychiatry. 2024;15:1279072.
  doi:[10.3389/fpsyt.2024.1279072](https://doi.org/10.3389/fpsyt.2024.1279072)
- Seybert C, et al. *Replicability of motor cortex-excitability modulation by
  intermittent theta burst stimulation.* Clin Neurophysiol. 2023;152:22–33.
  doi:[10.1016/j.clinph.2023.04.014](https://doi.org/10.1016/j.clinph.2023.04.014)

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

---

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