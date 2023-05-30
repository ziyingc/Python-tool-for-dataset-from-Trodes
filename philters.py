import struct
import numpy as np
from itertools import combinations
from scipy.signal import butter, bessel, filtfilt, freqz, firwin
import scipy.signal as signal
from scipy.stats.stats import pearsonr
import os

def read_burst(data, thr1, thr2, idx_clean, merge):
    l_t   = len(data)
    idx_t = np.arange(1,l_t)
    
    idx   = np.zeros([l_t,])
    idx[data>=thr1] = 1

    idx_start_stop = np.diff(idx);
    t_idx_thr1_start = idx_t[idx_start_stop==1]
    t_idx_thr1_stop  = idx_t[idx_start_stop==-1]
    l_thr1_start = len(t_idx_thr1_start);
    l_thr1_stop  = len(t_idx_thr1_stop);
    if l_thr1_start != l_thr1_stop and l_thr1_start>0 and l_thr1_stop>0:
        if t_idx_thr1_stop[0]  < t_idx_thr1_start[0]:
            t_idx_thr1_stop = np.delete(t_idx_thr1_stop, 0)
        if t_idx_thr1_stop[-1] < t_idx_thr1_start[-1]:
            t_idx_thr1_start = np.delete(t_idx_thr1_start, -1)
            
    l_thr1_start = len(t_idx_thr1_start);
    l_thr1_stop  = len(t_idx_thr1_stop);
    
    if l_thr1_start != l_thr1_stop:
        t_idx_thr1_start = np.delete(t_idx_thr1_start, -1)
    
    T_1 = t_idx_thr1_stop - t_idx_thr1_start
    
    
    idx   = np.zeros([l_t,])
    idx[data>=thr2] = 1
    idx_thr1 = t_idx_thr1_start<0
    
#     print(t_idx_thr1_stop.shape, T_1.shape, t_idx_thr1_start.shape)
    ii = 0
    for i in t_idx_thr1_start:
        idx_thr1[ii] = np.sum(idx[i:i+T_1[ii]])>0 #and np.sum(idx_clean[i:i+T_1[ii]])==T_1[ii]
        ii = ii + 1

    T_1     = T_1[idx_thr1]
    t_start = t_idx_thr1_start[idx_thr1]
    
####### merge close burst ##########
    if merge>0:
        d_idx_1 = t_start[1:len(T_1)] - (t_start + T_1)[0:(len(T_1)-1)]
        ii = 0
        idx_merge = t_start>0;
        for i in d_idx_1:
            if i <= merge:
                if T_1[ii] >= T_1[ii+1]:
                    idx_merge[ii+1] = False
                else:
                    idx_merge[ii] = False
            ii = ii+1
        T_1     = T_1[idx_merge]
        t_start = t_start[idx_merge]
    return t_start, T_1


def unique_burst(T_l, T_i, idx_l, idx_i, P_l, P_i, l_t, gap):

    idx_t_l = np.zeros((l_t,))<0;
    for i in np.arange(len(T_l)):
        i_brt = idx_l[i]; T_brt = T_l[i]
        idx_t_l[i_brt-gap:(i_brt+T_brt+gap)] = True
        
    idx_t_i = np.zeros((l_t,))<0;
    for i in np.arange(len(T_i)):
        i_brt = idx_i[i]; T_brt = T_i[i]
        idx_t_i[i_brt-gap:(i_brt+T_brt+gap)] = True

    idx_u_i = np.empty((0,)); T_u_i = np.empty((0,));
    idx_c_i = np.empty((0,)); T_c_i = np.empty((0,));
    for i in np.arange(len(T_i)):
        i_brt = idx_i[i]; T_brt = T_i[i]
        if np.sum(idx_t_l[i_brt:(i_brt+T_brt)])==0:
            idx_u_i = np.append(idx_u_i, i_brt);  T_u_i = np.append(T_u_i, T_brt);
        else:
            idx_c_i = np.append(idx_c_i, i_brt);  T_c_i = np.append(T_c_i, T_brt);
            
    idx_u_l = np.empty((0,)); T_u_l = np.empty((0,));
    idx_c_l = np.empty((0,)); T_c_l = np.empty((0,));
    for i in np.arange(len(T_l)):
        i_brt = idx_l[i]; T_brt = T_l[i]
        if np.sum(idx_t_i[i_brt:(i_brt+T_brt)])==0:
            idx_u_l = np.append(idx_u_l, i_brt); T_u_l = np.append(T_u_l, T_brt);
        else:
            idx_c_l = np.append(idx_c_l, i_brt); T_c_l = np.append(T_c_l, T_brt);
            
    return idx_u_l, idx_c_l, T_u_l, idx_u_i, idx_c_i, T_u_i
            
    
def read_epochs_trans(data, order1, thr_peak, order2):
    
    indexes = np.array(signal.argrelextrema(data, np.greater, order = order1))[0]
    spd_idx = data[indexes]
    idx_2 = spd_idx>=thr_peak
    indexes = indexes[idx_2]
    ################################################
    j = 0
    idx_1 = indexes>0
    for i in np.arange(len(indexes)-1):
        t1 = indexes[i]
        t2 = indexes[i+1]
        if t2-t1 < order2:
            if data[t1]<data[t2]:
                idx_1[i] = False
            else:
                idx_1[i+1] = False
    indexes = indexes[idx_1]
    ###############################################
    return indexes


def read_epochs_peaks(data, order1, thr_peak, order2, drop, order3):

    indexes = np.array(signal.argrelextrema(data, np.greater, order = order1))[0]
    spd_idx = data[indexes]
    idx_1 = spd_idx>=thr_peak
    indexes = indexes[idx_1]
    ######################    ##########################
    j = 0
    idx_1 = indexes>0
    for i in np.arange(len(indexes)-1):
        t1 = indexes[i]
        t2 = indexes[i+1]
        if t2-t1 < order2:
            if data[t1]<data[t2]:
                idx_1[i] = False
            else:
                idx_1[i+1] = False
    indexes = indexes[idx_1]
    ######################     #########################
    indexes = indexes[np.logical_and(indexes>=order3, indexes<=len(data)-order3)]
    spd_idx1 = indexes<0
    peaks = data[indexes]
    j = 0
    for i in indexes:
        
        t1 = int(i-order3)
        t2 = int(i+order3)
        if t1 < 0:
            t1 = 0
        if t2 >= len(data):
            t2 = len(data)-1
        try:
            spd_idx1[j] = np.min(data[t1:i])<(peaks[j]*drop) or np.min(data[(i+1):t2])<(peaks[j]*drop)
        except:
            print(t1,i,i+1,t2, len(data))
        j = j+1
    
    indexes = indexes[spd_idx1]
    return indexes