# -*- coding: utf-8 -*-
"""
Created on Sun Jan  9 18:29:26 2022

@author: Admin
"""

# %% Define Paths

code_path = 'C:/Users/Admin/Desktop/FV/TMS pipeline/'
data_path = 'C:/Users/Admin/Desktop/FV/TMS pipeline/data/raw/'

# %% Import Packages
import os
import sys
sys.path.insert(0,code_path)
from TMS_preprocessing_module import *
import numpy as np
from math import nan

# %% Define file for preprocessing

# CHANGE FILE NAME BELOW!
name = 'HC36_s1_l_2021-07-29_15-58-36'
# =========================================

rawData_path = data_path + name + ".bin"
output_path = data_path + 'preprocessed/'

if not os.path.exists(output_path):
    # Create a new directory for preprocessed data if it does not exist 
    os.makedirs(output_path)
    os.makedirs(output_path + 'XLSX/')
    os.makedirs(output_path + 'NPY/')


info = get_info(rawData_path)   # Get session information
data = load_data(rawData_path,fs = FS)
FS = 4000   # Sampling frequency (Hz)
# Print session information
print('-'*20)
print('Preprocessing file:')
print('\tSubject: {};\n\tSession: {};\n\tCollected side: {};\n\tCollection Date: {}.'.format(info[0],info[1],info[2],info[3]))
print('-'*20)
view_channels(data,FS)

# %% Split Trial
'''
  For each variable (l_rmt, mvic, etc.), type the time (in minutes) corresponding 
to the end of the section in question.
  In the case of "ref_emg", type the time where you want to extract a 500 ms 
emg trace for baseline noise reference.

Use at least 2 decimal points
'''
# CHANGE BELOW!
plt.close('all')
ref_emg = 38.80
l_rmt = 5.36
mvic = 9.00
l_amt = 13.16
r_rmt = nan
b_meps = 18.51
itbs = 23.12
t0_meps = 28.45
rest0 = 31.65
t10_meps = 36.98
rest10 = 42.23
t20_meps = 47.53
rest20 = 52.47
t30_meps = 57.83
r_rmt_pos = nan
l_rmt_pos = 60.63
# =========================================

markers = {'ref_emg':ref_emg, 'l_rmt':l_rmt ,'mvic':mvic ,'l_amt':l_amt ,
           'r_rmt':r_rmt ,'b_meps':b_meps ,'itbs':itbs,'t0_meps':t0_meps ,
           'rest0':rest0 ,'t10_meps':t10_meps ,'rest10':rest10 ,
           't20_meps':t20_meps ,'rest20':rest20 ,'t30_meps':t30_meps ,
           'r_rmt_pos':r_rmt_pos ,'l_rmt_pos':l_rmt_pos}


data = DenoiseSignal(data, markers)
#split session and plot MVICs
DATA, mvicData = prepDataDiccionary(data,markers,FS)


# %% Segment MVICs

plt.close('all')

'''
Type the time (in seconds) corresponding to the begining of each MVIC
'''
# CHANGE BELOW!
mvic_start_1 = 38.4
mvic_start_2 = 106.5
mvic_start_3 = 176.2
# =========================================

CONTRACTIONS = plot_MVICs(mvicData,[mvic_start_1,mvic_start_2,mvic_start_3])

# %% Check for muscle activity and plot overlapped MEPs

PULSES, FLAGS = check_muscle_activity(DATA,info,fs = FS)

plot_MEPoverlap(DATA,PULSES,info,fs = FS)

# %% Plot baseline MEPs
plt.close('all')

''' 
Type the time (in miliseconds) corresponding to an interval where all the MEPs
are contained.
'''
# CHANGE BELOW! (if needed)
mep_start = 23
mep_end = 34


EXTREMES = {}

# ================================== b_meps =================================
key = 'b_meps'
extremes = plot_allMEPs(key,mep_start,mep_end,info,DATA,PULSES,fs = FS)

# %% Check b_meps

''' 
Type the pulse index for badly charachterized MEPs, seperated by ','. 
    Example:
        bad_meps = [10,15,37]
'''
# CHANGE BELOW! (if needed)
bad_meps = []
if bad_meps != []:
    for p in bad_meps:
        FLAGS[key][int(p)].append(4)
EXTREMES[key] = extremes
plt.close('all')
# %% Plot T0 MEPs

# ================================== t0_meps =================================
key = 't0_meps'
extremes = plot_allMEPs(key,mep_start,mep_end,info,DATA,PULSES,fs = FS)

#%% Check t0_meps

''' 
Type the pulse index for badly charachterized MEPs, seperated by ','. 
    Example:
        bad_meps = [10,15,37]
'''
# CHANGE BELOW! (if needed)
bad_meps = []
if bad_meps != []:
    for p in bad_meps:
        FLAGS[key][int(p)].append(4)
EXTREMES[key] = extremes
plt.close('all')

# %% Plot T10 MEPs

# ================================== t10_meps =================================
key = 't10_meps'
extremes = plot_allMEPs(key,mep_start,mep_end,info,DATA,PULSES,fs = FS)

#%% Check t10_meps

''' 
Type the pulse index for badly charachterized MEPs, seperated by ','. 
    Example:
        bad_meps = [10,15,37]
'''
# CHANGE BELOW! (if needed)
bad_meps = []
if bad_meps != []:
    for p in bad_meps:
        FLAGS[key][int(p)].append(4)
EXTREMES[key] = extremes
plt.close('all')

# %% Plot T20 MEPs

# ================================== t20_meps =================================
key = 't20_meps'
extremes = plot_allMEPs(key,mep_start,mep_end,info,DATA,PULSES,fs = FS)

#%% Check t20_meps

''' 
Type the pulse index for badly charachterized MEPs, seperated by ','. 
    Example:
        bad_meps = [10,15,37]
'''
# CHANGE BELOW! (if needed)
bad_meps = []
if bad_meps != []:
    for p in bad_meps:
        FLAGS[key][int(p)].append(4)
EXTREMES[key] = extremes
plt.close('all')

# %% Plot T30 MEPs

# ================================== t30_meps =================================
key = 't30_meps'
extremes = plot_allMEPs(key,mep_start,mep_end,info,DATA,PULSES,fs = FS)

#%% Check t30_meps

''' 
Type the pulse index for badly charachterized MEPs, seperated by ','. 
    Example:
        bad_meps = [10,15,37]
'''
# CHANGE BELOW! (if needed)
bad_meps = []
if bad_meps != []:
    for p in bad_meps:
        FLAGS[key][int(p)].append(4)
EXTREMES[key] = extremes
plt.close('all')

# %% Create Peak Excel

# Create excell with MEP peak values
xlsx_path = create_MEPeacks_excel(EXTREMES,info,output_path)

# Plot MEPs that where mischarachterized by algorithm
plot_badMEPs(mep_start,mep_end,info,DATA,PULSES,EXTREMES,FLAGS,fs = FS)


'''
If some MEPs where badly identified, follow these steps:
    1. Open corresponding .xlsx file (ex: HC36_s1_l_peaks.xlsx)
    2. Check the correct value in the plotted MEPs
    3. Make changes in excel file and save document
'''
# %% Correct, Save and export results

plt.close('all')

SavePreprocessedData(xlsx_path, output_path, DATA, PULSES, EXTREMES, 
                     CONTRACTIONS, FS, FLAGS, info)
