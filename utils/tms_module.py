# emg_utils.py
# -*- coding: utf-8 -*-
"""
Utility functions for EMG data processing, peak detection, filtering, and preprocessing.
"""
# --- Core scientific stack ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Signal processing ---
from skimage.restoration import denoise_wavelet
from scipy.signal import find_peaks, savgol_filter, detrend
from math import isnan

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

def get_info(path):
    fname = path.split('/')[-1]
    temp = fname.split('.')[0].split('_')
    subject, session, hemi = temp[:3]
    date = temp[3] + ' at ' + temp[4]
    return subject, session, hemi, date
    

