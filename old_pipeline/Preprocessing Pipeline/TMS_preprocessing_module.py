# -*- coding: utf-8 -*-
"""
Created on Sat Jan  8 20:24:53 2022

@author: Admin
"""

#%% Packages

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tkinter as tk
from skimage.restoration import denoise_wavelet
from scipy.signal import find_peaks, savgol_filter, detrend
from math import isnan, ceil

#%% Usefull functions


def timearray(arr,fs,unit = 'min'):
    '''
    

    Parameters
    ----------
    arr : TYPE
        DESCRIPTION.
    fs : TYPE
        DESCRIPTION.
    unit : TYPE, optional
        DESCRIPTION. The default is 'min'.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    if unit == 'min':
        factor = 60
    if unit == 's':
        factor = 1
    elif unit == 'ms':
        factor = 1e-3
    
    N = len(arr)
    return np.linspace(0, N/fs/factor, N)

def samp2time(x,fs=4000,unit='min'):
    '''
    

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    fs : TYPE, optional
        DESCRIPTION. The default is 4000.
    unit : TYPE, optional
        DESCRIPTION. The default is 'min'.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    if unit == 'min':
        factor = 60
    if unit == 's':
        factor = 1
    elif unit == 'ms':
        factor = 1e-3
        
    return x/fs/factor

def time2samp(x,fs=4000,unit='min'):
    
    if unit == 'min':
        factor = 60
    if unit == 's':
        factor = 1
    elif unit == 'ms':
        factor = 1e-3
        
    return int(x*factor*fs)

def cm2in(x):
    return x*0.393701

def create_MEPeacks_excel(EXTREMES,info,output_path):
    
    dfs = {}

    for key in EXTREMES.keys():
        if len(EXTREMES[key]) < len(EXTREMES[key][0]):
            temp = []
            for i in range(len(EXTREMES[key][0])):
                temp += [[EXTREMES[key][0][i],EXTREMES[key][1][i]]]
            EXTREMES[key] = temp
        dfs[key] = pd.DataFrame(np.array(EXTREMES[key]), columns = ['mins','maxs'])
        
    xls_name = info[0] + '_' + info[1] + '_' + info[2] + '_peaks.xlsx'
    xlsx_path = output_path + 'XLSX/' + xls_name
    writer = pd.ExcelWriter(xlsx_path, engine='xlsxwriter')
    for key in dfs.keys():
        df_name = key
        df = dfs[key]
        df.to_excel(writer, sheet_name = df_name)
        
    writer.save()
    return xlsx_path

def SavePreprocessedData(xlsx_path, output_path, DATA, PULSES, EXTREMES, 
                         CONTRACTIONS, FS, FLAGS, info):
    
    for key in EXTREMES.keys():
        tmp= pd.read_excel(xlsx_path, sheet_name=key)
        mins = tmp['mins'].tolist()
        maxs = tmp['maxs'].tolist()
        EXTREMES[key]=[mins,maxs]
        
    out = {'data': DATA,
       'pulses': PULSES,
       'extremes': EXTREMES,
       'contractions': CONTRACTIONS,
       'fs': FS,
       'flags':FLAGS,
       'info': info
       }
    
    fname = info[0] + '_' + info[1] + '_' + info[2] + '_preprocessed.npy'

    np.save(output_path+'NPY/'+fname,out)
    
    print('Preprocessed data saved to:')
    print('\t'+output_path+'NPY/'+fname)
        

# %%

def prepDataDiccionary(data,markers, fs = 4000, ref_time = 200):
    
    screen = tk.Tk()
    screen_width = screen.winfo_screenwidth()
    screen_height = screen.winfo_screenheight()
    px = 1/plt.rcParams['figure.dpi'] # pixel in inches
    
    figure_size = (0.8 *screen_width*px, 0.5 *screen_height*px)
    
    channel_names = ['Synch Pulse', 'Left Hemisphere', 'Right Hemisphere']
    
    markers_names = list(markers.keys())

    DATA = {}
    DATA['full_trial'] = data
    beg = time2samp(markers[markers_names[0]] , fs, 'min')
    DATA['ref_emg'] = data[:,beg:beg + time2samp(ref_time, fs, unit='ms')]
    for p in range(1,len(markers)):
        key = markers_names[p]
        key1 = markers_names[p-1]
        if key1 == 'ref_emg':
            end = time2samp(markers[key], fs, 'min')
            DATA[key] = data[:,:end]
        elif isnan(markers[key]):
            DATA[key] = np.nan
        else:
            i=0
            while isnan(markers[key1]):
                i-=1
                key1 = markers_names[p+i]
            beg = time2samp(markers[key1], fs, 'min')
            try:
                end = time2samp(markers[key], fs, 'min')
                DATA[key] = data[:,beg:end]
            except:
                DATA[key] = data[:,beg:]
                

    mvicData = DATA['mvic'][1,:]
    
    fig, ax = plt.subplots(1,1,figsize=figure_size)
    ax.plot(timearray(mvicData, fs,unit='s'),mvicData/1000)
    ax.set_title('MVIC data')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude (mV)')
    
    plt.tight_layout()
    
    return DATA, mvicData

#%% Filtering

def DenoiseSignal(data, markers, ref_time = 200, wav = 'db1', md = 'soft', wav_levels = 3, meth = 'BayesShrink', re_sig = 'True', sg_WinSize = 5, sg_Order = 3):
    '''
    

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    markers : TYPE
        DESCRIPTION.
    ref_time : TYPE, optional
        DESCRIPTION. The default is 200.
    wav : TYPE, optional
        DESCRIPTION. The default is 'db1'.
    md : TYPE, optional
        DESCRIPTION. The default is 'soft'.
    wav_levels : TYPE, optional
        DESCRIPTION. The default is 3.
    meth : TYPE, optional
        DESCRIPTION. The default is 'BayesShrink'.
    re_sig : TYPE, optional
        DESCRIPTION. The default is 'True'.
    sg_WinSize : TYPE, optional
        DESCRIPTION. The default is 5.
    sg_Order : TYPE, optional
        DESCRIPTION. The default is 3.

    Returns
    -------
    data : TYPE
        DESCRIPTION.

    '''
    markers_names = list(markers.keys())
    beg = time2samp(markers[markers_names[0]] , fs, 'min')
    noise = data[:,beg:beg + time2samp(ref_time ,fs, unit='ms')]
    for p in range(1,3):
        # Wavelet denoising
        tmp = denoise_wavelet(data[p,:], wavelet = wav ,mode = md ,wavelet_levels = wav_levels ,method = meth,rescale_sigma =re_sig, sigma = sigma)
        # Savotsky Golay smoothing
        data[p,:] = detrend(savgol_filter(tmp,sg_WinSize,sg_Order))
    
    return data

#%% MVIC functions

def window_rms(a, window_size):
    '''
    

    Parameters
    ----------
    a : TYPE
        DESCRIPTION.
    window_size : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    a2 = np.power(a,2)
    window = np.ones(window_size)/float(window_size)
    return np.sqrt(np.convolve(a2, window, 'same'))

def half_wave_rec(data,win_size_ms=200, output ='short', fs=4000):

    win_size = int((win_size_ms*fs)/1000)

    con_neg = np.zeros((len(data)))
    con_pos = np.zeros((len(data)))
    i = -1
    for p in data:
        i += 1
        if p <= 0:
            con_neg[i] = abs(p)
            con_pos[i] = 0
        else:
            con_neg[i] = 0
            con_pos[i] = abs(p)

    con_neg_rec = window_rms(con_neg,win_size)
    con_pos_rec = window_rms(con_pos,win_size)

    maximum = np.max(con_pos_rec)
    minimum = np.max(con_neg_rec)
    if output == 'short':
        return maximum+minimum
    else:
        return maximum+minimum, con_neg_rec, con_pos_rec
    
def full_wave_rec(data,win_size_ms=200, output ='short', fs=4000):

    win_size = int((win_size_ms*fs)/1000)

    con_rec = abs(data)
    
    con_rec_filt = window_rms(con_rec,win_size)

    maximum = np.max(con_rec)
    if output == 'short':
        return maximum
    else:
        return maximum, con_rec_filt
    


#%% TMS pulse finder

def find_pulse(synch_pulse_channel):
    '''
    

    Parameters
    ----------
    synch_pulse_channel : TYPE
        DESCRIPTION.

    Returns
    -------
    pulses : TYPE
        DESCRIPTION.

    '''
    av = np.mean(synch_pulse_channel)
    temp = np.where(synch_pulse_channel > av*5)[0]
    temp1= np.diff(temp)
    temp1 = np.append(temp1,10)# include all pulses
    temp2 = temp1==1
    te = temp2.tobytes().find(False) 
    pulses = temp[temp1>1] - te
    return pulses


#%% Muscle Activity Functions

# McCambridge et al, 2020
def std_muscle_activity(pulse, emg, rest_ref, fs = 4000):
    t_interval = 0.05 #seconds
    samples = range( int(pulse-t_interval*fs),int(pulse) )
    pre_pulse = np.std( np.abs( emg[samples] ) )
    if pre_pulse > 2 * rest_ref:
        flag = 2
    else:
        flag = 0
    return flag
        
# Hinder et al 2014 (40ms before pulse canâ€™t exceed 15 ÂµV)
def hinder_muscle_activity(pulse, emg, fs=4000):
    rest_ref = 15 #microVolts
    t_interval = 0.05 #seconds
    samples = range( int(pulse-t_interval*fs),int(pulse) )
    pre_pulse = np.std( np.abs( emg[samples] ) )
    if pre_pulse > rest_ref:
        flag = 3
    else:
        flag = 0
    return flag


def check_muscle_activity(DATA,info,fs = 4000):
    PULSES = {}
    FLAGS = {}
    for key in DATA.keys():
        flags = []
        pulses = []
        if key.split('_')[-1] == 'meps' and type(DATA[key]) != float:
            # print(key)
            pulses = find_pulse(DATA[key][0,:])
            if info[2] == 'l':
                emg = DATA[key][1,:]
            else:
                emg = DATA[key][2,:]
            for pulse in pulses:
                hinder_flag = hinder_muscle_activity(pulse, emg, fs)
                rest_ref = np.std( np.abs( DATA['ref_emg'] ) )
                std_flag = std_muscle_activity(pulse, emg, rest_ref, fs)
                flags_ = [hinder_flag,std_flag]
                flags.append( [i for i in flags_ if i != 0] )
        PULSES[key] = pulses        
        FLAGS[key] = flags
    return PULSES, FLAGS

#%% Unpacking Functions

def get_info(path):
    fname = path.split('/')[-1]
    temp = fname.split('.')[0].split('_')
    subject = temp[0]
    session = temp[1]
    hemi = temp[2]
    date = temp[3] + ' at ' + temp[4]
    return subject, session, hemi, date

def load_data(file_dir,channels=np.array((3,1,2)),fs=4000,normalize_voltage= True, voltage_gain = 133, reshape = 4):
    '''loads data from binary file.
    For PDM_TMS cohort:
        Use defaults
    For pilot data use:
        reshape = 7
        voltage_gain = 400
        fs = 1000
        channels = np.array((1,4,5))
    For board_validation data use:
        reshape = 3
        voltage_gain = 200
        fs = 5000
        channels = np.array((2,0,1))
            Inputs:
                file_dir: string = (path + file name)
            Outputs:
                data: structure (synch_pulse,left_channel, right_channel)'''
    c_synch_pulse = channels[0]
    c_left = channels[2]
    c_right = channels[1]
    temp = np.array(np.fromfile(file_dir, dtype='uint16')).astype(float)
    temp = np.reshape(temp,(reshape,len(temp)//reshape),order='F')
    
    n_samples = temp.shape[1]
    inds = np.arange(0,n_samples)
    if normalize_voltage == True:
        temp_c_left = ((temp[c_left,inds]*(3/2**16)-1.5)/voltage_gain*10e5)
        temp_c_right = ((temp[c_right,inds]*(3/2**16)-1.5)/voltage_gain*10e5)
        # temp_c_right = temp[c_right,inds]
    data = np.array((temp[c_synch_pulse,inds],temp_c_left,temp_c_right))
    return data

#%% Quick Plots

def view_channels(data,fs = 4000):
    '''
    

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    fs : TYPE, optional
        DESCRIPTION. The default is 4000.

    Returns
    -------
    None.

    '''
    
    screen = tk.Tk()
    screen_width = screen.winfo_screenwidth()
    screen_height = screen.winfo_screenheight()
    px = 1/plt.rcParams['figure.dpi'] # pixel in inches
    
    figure_size = (0.8 *screen_width*px, 0.8 *screen_height*px)
    
    channel_names = ['Synch Pulse', 'Left Hemisphere', 'Right Hemisphere']
    
    fig, ax = plt.subplots(data.shape[0],1, figsize=figure_size ,
                           sharex='col')


    for chan in range(data.shape[0]):
    
        ax[chan].plot(timearray(data[chan], fs), data[chan]/1000)
        ax[chan].set_title(channel_names[chan])
        ax[chan].set_ylabel('Amplitude (mV)')
        if chan == data.shape[0]-1:
            ax[chan].set_xlabel('Time (min)')
        
    plt.tight_layout()
    plt.show()
        
def plot_MVICs(mvicData,snipits,fs= 4000, samples = 8000):
    
    screen = tk.Tk()
    screen_width = screen.winfo_screenwidth()
    screen_height = screen.winfo_screenheight()
    px = 1/plt.rcParams['figure.dpi'] # pixel in inches
    
    figure_size = (0.8 *screen_width*px, 0.7 *screen_height*px)
    
    samp_snips = [time2samp(snip,fs,unit='s') for snip in snipits]
    
    mvic1 = mvicData[samp_snips[0]:samp_snips[0]+samples]
    mvic2 = mvicData[samp_snips[1]:samp_snips[1]+samples]
    if samp_snips[2]+samples > len(mvicData):
        mvic3 = mvicData[samp_snips[2]:]
    else:
        mvic3 = mvicData[samp_snips[2]:samp_snips[2]+samples]
        
    contracs = [mvic1,mvic2,mvic3]
    
    full_mvic = []
    envelope_mvic = []
    for p in range(len(contracs)):
        envelope_mvic += [half_wave_rec(contracs[p],fs)]
        full_mvic += [full_wave_rec(contracs[p],fs)]
        
    CONTRACTIONS = {'raw':contracs,'envelope':envelope_mvic,'full_wave':full_mvic}
    
    fig, ax = plt.subplots(3,1,figsize=figure_size,sharex = True)
    ax[0].plot(timearray(mvic1, fs,'s'), mvic1/1000)
    ax[0].set_title('MVIC 1')
    ax[0].set_ylabel('Amplitude (mV)')
    
    ax[1].plot(timearray(mvic2, fs,'s'), mvic2/1000)
    ax[1].set_title('MVIC 2')
    ax[1].set_ylabel('Amplitude (mV)')
    
    ax[2].plot(timearray(mvic3, fs,'s'), mvic3/1000)
    ax[2].set_title('MVIC 3')
    ax[2].set_ylabel('Amplitude (mV)')
    ax[2].set_xlabel('Time (s)')
    
    plt.tight_layout()
    
    return CONTRACTIONS
    
def plot_MEPoverlap(DATA,PULSES,info,bef_pulse = 10,aft_pulse = 60,fs = 4000):
    
    screen = tk.Tk()
    screen_width = screen.winfo_screenwidth()
    screen_height = screen.winfo_screenheight()
    px = 1/plt.rcParams['figure.dpi'] # pixel in inches
    
    figure_size = (0.6 *screen_width*px, 0.7 *screen_height*px)
    
    keys = ['b_meps', 't0_meps', 't10_meps', 't20_meps', 't30_meps']
    
    bef_pulse_sample = time2samp(bef_pulse,fs,unit = 'ms')
    aft_pulse_sample = time2samp(aft_pulse,fs,unit = 'ms')
    
    if info[2] == 'l':
        channel = 1
    else:
        channel = 2
        
    first = True
    fig, ax = plt.subplots(1,1,figsize=figure_size)    
    for key in keys:
        timepoint_emg = DATA[key][channel,:]
        timepoint_pulses = PULSES[key]
        for pulse in timepoint_pulses:
            mep = timepoint_emg[pulse-bef_pulse_sample:pulse+aft_pulse_sample]
            if first:
                first = False
                time_ms = timearray(mep, fs,'ms') - bef_pulse
                all_meps = mep
            else:
                all_meps = np.vstack((all_meps,mep))
            ax.plot(time_ms,mep/1000,color = 'lightseagreen', alpha = 0.3,
                    linewidth = 3)
    
    ax.plot(time_ms,np.mean(all_meps,axis = 0)/1000,color = 'blue')
    ax.axvline(0)
    ax.set_title('MEPs overlap')
    ax.set_ylabel('Amplitude (mV)')
    ax.set_xlabel('Time (ms)')
    ax.grid('both')
    
    plt.tight_layout()

def plot_allMEPs(key,mep_start,mep_end,info,DATA,PULSES,fs = 4000):
    
    screen = tk.Tk()
    screen_width = screen.winfo_screenwidth()
    screen_height = screen.winfo_screenheight()
    px = 1/plt.rcParams['figure.dpi'] # pixel in inches
    
    figure_size = (0.8 *screen_width*px, 0.8 *screen_height*px)
    
    pk_width = ceil(fs*0.001)
    rows = 4
    cols= 5
    scale = 1000
    
    mep_start_samp = time2samp(mep_start,fs,unit='ms')
    mep_end_samp = time2samp(mep_end,fs,unit='ms')
    
    extremes = []
    first = True
    if info[2] == 'l':
        emg = DATA[key][1,:]
    else:
        emg = DATA[key][2,:]
    pulses = PULSES[key]
    for c in range(len(pulses)):
        
        mep = emg[pulses[c] + mep_start_samp:pulses[c] + mep_end_samp]
        if first:
            first = False
            t_ms = timearray(mep,fs,unit='ms') + mep_start
        
        max_ind = find_peaks(mep,distance = len(mep)-1 , width = pk_width)[0]
        min_ind = find_peaks(-mep,distance = len(mep)-1 , width = pk_width)[0]
        
        if max_ind.size == 0 or min_ind.size == 0:
            max_ind = find_peaks(mep,distance = len(mep)-1)[0]
            min_ind = find_peaks(-mep,distance = len(mep)-1)[0]
            if max_ind.size == 0 or min_ind.size == 0:
                max_ind = 0
                min_ind = 0
        
        max_ms = samp2time(max_ind,fs,unit='ms') + mep_start
        min_ms = samp2time(min_ind,fs,unit='ms') + mep_start
        extrema_ms = np.array((min_ms,max_ms))
        
        extremes.append([ float(mep[min_ind]),float(mep[max_ind]) ])
        extrema_value  = np.array((mep[min_ind],mep[max_ind]))
        
        if c == 0 :
            fig, ax = plt.subplots(rows,cols, figsize=figure_size)
            fig.suptitle(key)
            i = 0
            j = 0
            ax[i,j].plot(t_ms, mep/scale)
            ax[i,j].scatter(extrema_ms, extrema_value/scale, c = ['b'], marker = '+')
            ax[i,j].set_title( 'Pulse '+ str(c))
            ax[i,j].set_ylabel('Voltage (mV)')
            
            
        elif i < rows and j < cols-1:
            j+= 1
            ax[i,j].plot(t_ms, mep/scale)
            ax[i,j].scatter(extrema_ms, extrema_value/scale, c = ['b'], marker = '+')
            ax[i,j].set_title( 'Pulse '+ str(c))
            if i == rows-1:
                ax[i,j].set_xlabel('Time (ms)')
            
        elif j == cols -1 and i < rows-1:
            
            j=0
            i +=1
            ax[i,j].plot(t_ms, mep/scale)
            ax[i,j].scatter(extrema_ms, extrema_value/scale, c = ['b'], marker = '+')
            ax[i,j].set_title( 'Pulse '+ str(c))
            ax[i,j].set_ylabel('Voltage (mV)')
            if i == rows-1:
                ax[i,j].set_xlabel('Time (ms)')
            
        elif i == rows-1:
            plt.tight_layout()    
            
            fig, ax = plt.subplots(rows,cols, figsize=figure_size)
            fig.suptitle(key)
            i = 0
            j = 0
            ax[i,j].plot(t_ms, mep/scale)
            ax[i,j].scatter(extrema_ms, extrema_value/scale, c = ['b'], marker = '+')
            ax[i,j].set_title( 'Pulse '+ str(c))
            ax[i,j].set_ylabel('Voltage (mV)')
            
        plt.tight_layout()  
    
    return extremes
    
def plot_badMEPs(mep_start,mep_end,info,DATA,PULSES,EXTREMES,FLAGS,fs = 4000):
    
    screen = tk.Tk()
    screen_width = screen.winfo_screenwidth()
    screen_height = screen.winfo_screenheight()
    px = 1/plt.rcParams['figure.dpi'] # pixel in inches
    
    figure_size = (0.8 *screen_width*px, 0.8 *screen_height*px)
    
    rows = 3
    cols= 4
    scale = 1
    pk_width = ceil(fs*0.001)
    
    mep_start_samp = time2samp(mep_start,fs,unit='ms')
    mep_end_samp = time2samp(mep_end,fs,unit='ms')
    
    first = True
    
    if info[2] == 'l':
        channel = 1
    else:
        channel = 2
    
    for key in EXTREMES.keys():
        c = 0
        emg = DATA[key][channel,:]
        flags = FLAGS[key]
        pulses = PULSES[key]
        extremes = EXTREMES[key]
        for p in range(len(pulses)):
            if 4 in flags[p]:
                
                mep = emg[pulses[p] + mep_start_samp:pulses[p] + mep_end_samp]
                extrema_value = np.array(extremes[p])
                if first:
                    first = False
                    t_ms = timearray(mep,fs,unit='ms') + mep_start
                    
                max_ind = find_peaks(mep,distance = len(mep)-1 , width = pk_width)[0]
                min_ind = find_peaks(-mep,distance = len(mep)-1 , width = pk_width)[0]
                
                if max_ind.size == 0 or min_ind.size == 0:
                    max_ind = find_peaks(mep,distance = len(mep)-1)[0]
                    min_ind = find_peaks(-mep,distance = len(mep)-1)[0]
                    if max_ind.size == 0 or min_ind.size == 0:
                        max_ind = 0
                        min_ind = 0
                
                max_ms = samp2time(max_ind,fs,unit='ms') + mep_start
                min_ms = samp2time(min_ind,fs,unit='ms') + mep_start
                extrema_ms = np.array((min_ms,max_ms))
        
                if c == 0 :
                    fig, ax = plt.subplots(rows,cols, figsize=figure_size)
                    fig.suptitle(key)
                    i = 0
                    j = 0
                    ax[i,j].plot(t_ms, mep/scale)
                    ax[i,j].scatter(extrema_ms, extrema_value/scale, c = ['b'], marker = '+')
                    ax[i,j].set_title( 'Pulse '+ str(p))
                    ax[i,j].set_xlabel('Time (ms)')
                    ax[i,j].set_ylabel('Voltage (microV)' )
                    
                elif i < rows and j < cols-1:
                    j+= 1
                    ax[i,j].plot(t_ms, mep/scale)
                    ax[i,j].scatter(extrema_ms, extrema_value/scale, c = ['b'], marker = '+')
                    ax[i,j].set_title( 'Pulse '+ str(p))
                    ax[i,j].set_xlabel('Time (ms)')
                    ax[i,j].set_ylabel('Voltage (microV)' )
                    
                elif j == cols -1 and i < rows-1:
                    
                    j=0
                    i +=1
                    ax[i,j].plot(t_ms, mep/scale)
                    ax[i,j].scatter(extrema_ms, extrema_value/scale, c = ['b'], marker = '+')
                    ax[i,j].set_title( 'Pulse '+ str(p))
                    ax[i,j].set_xlabel('Time (ms)')
                    ax[i,j].set_ylabel('Voltage (microV)' )
                        
                elif i == rows-1:
                    plt.tight_layout()    
                    
                    fig, ax = plt.subplots(rows,cols, figsize=figure_size)
                    fig.suptitle(key)
                    i = 0
                    j = 0
                    ax[i,j].plot(t_ms, mep/scale)
                    ax[i,j].scatter(extrema_ms, extrema_value/scale, c = ['b'], marker = '+')
                    ax[i,j].set_title( 'Pulse '+ str(p))
                    ax[i,j].set_xlabel('Time (ms)')
                    ax[i,j].set_ylabel('Voltage (microV)' )
                    
                plt.tight_layout()
                c += 1
    
    