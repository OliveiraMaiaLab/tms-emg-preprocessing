# emg_utils.py
# -*- coding: utf-8 -*-
"""
Utility functions for EMG data processing, peak detection, filtering, and preprocessing.
"""
from __future__ import annotations

# --- Core scientific stack ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Signal processing ---
from skimage.restoration import denoise_wavelet
from scipy.signal import find_peaks, savgol_filter, detrend
from math import isnan


from typing import Tuple, List
from pathlib import Path
import json

from dataclasses import dataclass


# ------------------------------
# Time & Conversion Functions
# ------------------------------

def timearray(arr, fs, unit='min'):
    factor = {'min': 60, 's': 1, 'ms': 1e-3}.get(unit, 60)
    N = len(arr)
    return np.linspace(0, N/fs/factor, N)

def samp2time(x, fs=4000, unit='min'):
    factor = {'min': 60, 's': 1, 'ms': 1e-3}.get(unit, 60)
    return x / fs / factor

def time2samp(x, fs=4000, unit='min'):
    factor = {'min': 60, 's': 1, 'ms': 1e-3}.get(unit, 60)
    return int(x * factor * fs)

def cm2in(x):
    return x * 0.393701

# ------------------------------
# File Handling
# ------------------------------

def create_MEPeacks_excel(EXTREMES, info, output_path):
    dfs = {}
    for key, value in EXTREMES.items():
        if len(value) < len(value[0]):
            temp = [[value[0][i], value[1][i]] for i in range(len(value[0]))]
            EXTREMES[key] = temp
        dfs[key] = pd.DataFrame(np.array(EXTREMES[key]), columns=['mins', 'maxs'])
    
    xls_name = f"{info[0]}_{info[1]}_{info[2]}_peaks.xlsx"
    xlsx_path = f"{output_path}XLSX/{xls_name}"
    with pd.ExcelWriter(xlsx_path, engine='xlsxwriter') as writer:
        for key, df in dfs.items():
            df.to_excel(writer, sheet_name=key)
    return xlsx_path

def SavePreprocessedData(xlsx_path, output_path, DATA, PULSES, EXTREMES, 
                         CONTRACTIONS, FS, FLAGS, info, markers):
    for key in EXTREMES.keys():
        tmp = pd.read_excel(xlsx_path, sheet_name=key)
        mins = tmp['mins'].tolist()
        maxs = tmp['maxs'].tolist()
        EXTREMES[key] = [mins, maxs]
        
    out = {
        'data': DATA,
        'pulses': PULSES,
        'extremes': EXTREMES,
        'contractions': CONTRACTIONS,
        'fs': FS,
        'flags': FLAGS,
        'markers': markers,
        'info': info,
    }
    
    fname = f"{info[0]}_{info[1]}_{info[2]}_preprocessed.npy"
    np.save(f"{output_path}NPY/{fname}", out)
    print(f'Preprocessed data saved to:\n\t{output_path}NPY/{fname}')

# ------------------------------
# Data Preparation
# ------------------------------

def prepDataDiccionary(data, markers, fs=4000, ref_time=200, show_plot=True):
    """Prepares data dictionary from raw EMG data."""
    
    channel_names = ['Synch Pulse', 'Left Hemisphere', 'Right Hemisphere']
    markers_names = list(markers.keys())

    DATA = {'full_trial': data}
    beg = time2samp(markers[markers_names[0]], fs, 'min')
    DATA['ref_emg'] = data[:, beg:beg + time2samp(ref_time, fs=fs, unit='ms')]
    
    for p in range(1, len(markers)):
        key = markers_names[p]
        key1 = markers_names[p-1]
        if key1 == 'ref_emg':
            end = time2samp(markers[key], fs, 'min')
            DATA[key] = data[:, :end]
        elif isnan(markers[key]):
            DATA[key] = np.nan
        else:
            i = 0
            while isnan(markers[key1]):
                i -= 1
                key1 = markers_names[p+i]
            beg = time2samp(markers[key1], fs, 'min')
            try:
                end = time2samp(markers[key], fs, 'min')
                DATA[key] = data[:, beg:end]
            except:
                DATA[key] = data[:, beg:]
    
    mvicData = DATA['mvic'][1, :]
    
    if show_plot:
        screen_width, screen_height = 1600, 900  # default for plotting if GUI not used
        px = 1/plt.rcParams['figure.dpi']
        figure_size = (0.8*screen_width*px, 0.5*screen_height*px)
        fig, ax = plt.subplots(figsize=figure_size)
        ax.plot(timearray(mvicData, fs, unit='s'), mvicData/1000)
        ax.set(title='MVIC data', xlabel='Time (s)', ylabel='Amplitude (mV)')
        plt.tight_layout()
    
    return DATA, mvicData

# ------------------------------
# Filtering
# ------------------------------

def DenoiseSignal(data, wav='db1', md='soft', wav_levels=3, meth='BayesShrink', 
                  re_sig=True, sg_WinSize=5, sg_Order=3):
    """Wavelet denoise + Savitzky-Golay smoothing."""
    for p in range(1, 3):
        tmp = denoise_wavelet(data[p, :], wavelet=wav, mode=md, wavelet_levels=wav_levels,
                              method=meth, rescale_sigma=re_sig)
        data[p, :] = detrend(savgol_filter(tmp, sg_WinSize, sg_Order))
    return data

# ------------------------------
# Pulse Detection & Muscle Activity
# ------------------------------

def find_pulse(synch_pulse_channel, key=None):
    try:
        av = np.mean(synch_pulse_channel)
        temp = np.where(synch_pulse_channel > av*5)[0]
        temp1 = np.diff(temp)
        temp1 = np.append(temp1, 10)
        temp2 = temp1 == 1
        te = temp2.tobytes().find(False)
        pulses = temp[temp1 > 1] - te
        return pulses
    except:
        if key: print(key)

def std_muscle_activity(pulse, emg, rest_ref, fs=4000):
    t_interval = 0.05
    samples = range(int(pulse-5-t_interval*fs), int(pulse-5))
    pre_pulse = np.std(np.abs(emg[samples]))
    return 2 if pre_pulse > rest_ref else 0

def hinder_muscle_activity(pulse, emg, fs=4000):
    rest_ref = 15
    t_interval = 0.05
    samples = range(int(pulse-5-t_interval*fs), int(pulse-5))
    pre_pulse = np.std(np.abs(emg[samples]))
    return 3 if pre_pulse > rest_ref else 0

def check_muscle_activity(DATA, info, fs=4000):
    PULSES, FLAGS = {}, {}
    for key in DATA.keys():
        flags, pulses = [], []
        if key.split('_')[-1] == 'meps' and type(DATA[key]) != float:
            pulses = find_pulse(DATA[key][0, :], key)
            emg = DATA[key][1, :] if info[2] == 'l' else DATA[key][2, :]
            for pulse in pulses:
                flags_ = [hinder_muscle_activity(pulse, emg, fs),
                          std_muscle_activity(pulse, emg, np.std(np.abs(DATA['ref_emg'])), fs)]
                flags.append([i for i in flags_ if i != 0])
        PULSES[key] = pulses
        FLAGS[key] = flags
    return PULSES, FLAGS

# ------------------------------
# Data Loading
# ------------------------------

def load_data(file_dir, channels=np.array((3,1,2)), fs=4000,
              normalize_voltage=True, voltage_gain=133, reshape=4):
    """
    Loads data from binary file and returns left/right channels and TMS pulse indexes.

    Parameters
    ----------
    file_dir : str
        Path to the binary data file.
    channels : np.array
        Channel ordering: (synch_pulse, right, left)
    fs : int
        Sampling rate.
    normalize_voltage : bool
        If True, normalize EMG channels to mV.
    voltage_gain : float
        Amplifier gain.
    reshape : int
        Number of channels per block.

    Returns
    -------
    data : np.ndarray
        2 x n_samples array with left and right hemisphere channels.
    tms_indexes : np.ndarray
        Indexes of TMS pulse events.
    """

    c_synch_pulse = channels[0]
    c_left = channels[2]
    c_right = channels[1]

    # Load and reshape binary data
    temp = np.fromfile(file_dir, dtype='uint16').astype(float)
    temp = np.reshape(temp, (reshape, len(temp)//reshape), order='F')
    
    n_samples = temp.shape[1]
    inds = np.arange(n_samples)

    # Normalize EMG channels if requested
    if normalize_voltage:
        temp_c_left = ((temp[c_left, inds]*(3/2**16) - 1.5) / voltage_gain * 1e6)
        temp_c_right = ((temp[c_right, inds]*(3/2**16) - 1.5) / voltage_gain * 1e6)
    else:
        temp_c_left = temp[c_left, inds]
        temp_c_right = temp[c_right, inds]

    data = np.array([temp_c_left, temp_c_right])

    # Detect TMS pulse indexes: rising edges of the binary synch signal
    pulse_signal = temp[c_synch_pulse, inds]
    threshold = pulse_signal.mean() + 3*pulse_signal.std()  # simple threshold
    above_thresh = pulse_signal > threshold
    tms_indexes = np.where(np.diff(above_thresh.astype(int)) == 1)[0] + 1  # rising edges

    return data, tms_indexes

from pathlib import Path
from typing import Tuple
import json
import numpy as np


def load_meps_for_block(
    meta: dict,
    session_file: str | Path,
    block_name: str,
    hemi: str = "left",
    pre_s: float = 0.100,
    post_s: float = 0.400,
    epoch=None,  # expects .tmin_ms and .tmax_ms if provided
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract per-block MEP waveforms from raw EMG using pulse indices stored in session JSON.

    Returns:
    - t_ms: (T,) time axis in milliseconds (relative to pulse)
    - meps: (N, T) waveforms (in uV), mean-detrended per trial

    Memory-efficient:
    - If epoch is provided, only [epoch.tmin_ms, epoch.tmax_ms) is extracted.
    - Otherwise, defaults to [-pre_s, +post_s) seconds.
    """
    session_file = Path(session_file)

    js = json.loads(session_file.read_text())
    meps_root = js.get("meps", {}) or {}
    hp = ((meps_root.get(block_name) or {}).get(hemi) or {})
    pulses = hp.get("pulses", [])
    if not isinstance(pulses, list) or not pulses:
        return np.asarray([], dtype=float), np.zeros((0, 0), dtype=float)

    fs = float(meta["sampling_rate"])
    data, _tms = load_data(meta["input_file"], channels=meta.get("channels"))

    hemi_to_ch = {"left": 0, "right": 1}
    ch = hemi_to_ch.get(hemi, 0)
    sig = data[ch, :]

    # -------------------------
    # Define window directly (epoch-aware)
    # -------------------------
    if epoch is not None:
        start_ms = float(epoch.tmin_ms)
        end_ms = float(epoch.tmax_ms)
    else:
        start_ms = -float(pre_s) * 1000.0
        end_ms = float(post_s) * 1000.0

    if end_ms <= start_ms:
        raise ValueError(f"Invalid window: start_ms={start_ms} must be < end_ms={end_ms}")

    start_n = int(np.round(start_ms / 1000.0 * fs))
    end_n = int(np.round(end_ms / 1000.0 * fs))

    n_win = end_n - start_n
    if n_win <= 0:
        raise ValueError("Window length is zero/negative after sample conversion.")

    # Time axis (relative to pulse)
    t_ms = (np.arange(start_n, end_n) / fs) * 1000.0

    trials = []
    for p in pulses:
        p = int(p)
        a = p + start_n
        b = p + end_n
        if a < 0 or b > sig.shape[0]:
            continue

        y = sig[a:b].astype(float)

        # Simple mean detrending (whole window)
        y -= float(y.mean())

        trials.append(y)

    if not trials:
        return t_ms, np.zeros((0, n_win), dtype=float)

    meps = np.asarray(trials, dtype=float)
    return t_ms, meps



def get_info(path):
    fname = path.split('/')[-1]
    temp = fname.split('.')[0].split('_')
    subject, session, hemi = temp[:3]
    date = temp[3] + ' at ' + temp[4]
    return subject, session, hemi, date

# ------------------------------
# Helpers for reading/writing peaks_flag and reading the MEP window.
# ------------------------------

@dataclass(frozen=True)
class Epoch:
    """Epoch bounds in milliseconds (relative to pulse)."""
    tmin_ms: float
    tmax_ms: float


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    tmp.replace(path)


def get_epoch_from_session(session: dict) -> Epoch:
    """
    Reads epoch bounds from:
      session["mep_window"] = [beg_s, end_s]   (seconds, relative to pulse)
    Returns in milliseconds.
    """
    w = session.get("mep_window", None)
    if not (isinstance(w, list) and len(w) == 2 and w[0] is not None and w[1] is not None):
        raise KeyError(
            "Missing MEP window definition in session JSON. "
            "Expected session['mep_window'] = [beg_s, end_s]."
        )
    beg_s = float(w[0])
    end_s = float(w[1])
    return Epoch(beg_s * 1000.0, end_s * 1000.0)


def get_peaks_flag_list(session: dict, block: str, hemi: str = "left") -> List[int]:
    meps = session.get("meps", {}) or {}
    hp = ((meps.get(block) or {}).get(hemi) or {})
    flags = hp.get("peaks_flag", [])
    if not isinstance(flags, list):
        return []
    out = []
    for v in flags:
        try:
            out.append(1 if int(v) else 0)
        except Exception:
            out.append(0)
    return out


def set_peaks_flag_list(session: dict, block: str, flags: List[int], hemi: str = "left") -> None:
    session.setdefault("meps", {})
    session["meps"].setdefault(block, {})
    session["meps"][block].setdefault(hemi, {})
    hp = session["meps"][block][hemi]
    hp["peaks_flag"] = [1 if int(v) else 0 for v in flags]
    session["meps"][block][hemi] = hp


def save_peaks_flag_list(session_file: Path, block: str, flags: List[int], hemi: str = "left") -> None:
    session = read_json(session_file)
    set_peaks_flag_list(session, block, flags, hemi=hemi)
    write_json(session_file, session)

    

