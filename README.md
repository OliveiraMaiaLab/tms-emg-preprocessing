# MEP Preprocessing — Streamlit GUI (dev branch)

**Branch:** `streamlit-gui-dev`
**Status:** *Developer preview (actively changing). Expect breaking changes.*

This branch contains a **GUI-driven pipeline** for preprocessing **motor evoked potentials (MEPs)** acquired with TMS. It wraps the established signal-processing steps from the main pipeline into an interactive **Streamlit** app that guides you from file selection to quality control and export.

---

**Run**

conda activate cfom_mep_preprocessing
cd /d D:\FV\Projects\pdm_tms\jove_acquisition_preprocessing\tms-emg-preprocessing
streamlit run main_gui.py

---

This GUI is intended to replace the older step-by-step guide. The previous implementation remains available in `old_pipeline/` on the `main` branch for reference.

--- 
## 🧪 Environment setup

This project provides three environment specifications:

- **`environment.yml`**  
  Portable Conda environment specification (package versions pinned, no build hashes).

- **`conda-win64-explicit.txt`**  
  Exact Conda build lock for Windows (fully reproducible on win-64).

- **`requirements-pip.txt`**  
  Exact pip package versions.

---

### ▶️ Option 1 — Create from `environment.yml` (recommended for general use)
```bash
conda env create -f environment.yml
```

### ▶️ Option 2 — Create an exact Windows clone (maximum reproducibility)
```bash
conda create -n <env_name> --file conda-win64-explicit.txt
conda activate <env_name>
python -m pip install -r requirements-pip.txt
```
Use this option if you need the environment to match the original exactly (same Conda builds and pip versions, Windows only).

---

Package organization

__init__.py
main_gui.py
utils/
├── persistence.py
├── tms_module.py
├── layout.py
├── peak_checking_io
├── mep_loading.py
bk_embedding/
├── segmentation.py
├── mepOverlap.py
steps/
├── step_input.py
├── step_confirmInputs.py  
├── step_segmentation.py
├── step_mepWindow.py 
├── step_peakChecking.py