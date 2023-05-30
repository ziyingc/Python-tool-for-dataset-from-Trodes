import struct
import numpy as np
import time2frequency as t2f
from itertools import combinations
from scipy.signal import butter, bessel, filtfilt, freqz, firwin
import scipy.signal as signal
from scipy.stats.stats import pearsonr
import os


def read_burst(rfdn, epn, subset, fps, fs_out, f, idx_f, flr, fhr, i_ref, merge, mode):
    x_fit = np.log10(f)
    slope_l, intercept_l, slope_i, intercept_i, intercept_li = \
    t2f.flicker_fit_fft(rfdn, epn, fs_out, subset, f, idx_f, mode)
    
    slope     = np.array([slope_l,     slope_i]);
    intercept = np.array([intercept_l, intercept_i]);
    
    idx_brst = np.empty((2,len(epn),len(flr)), dtype = object)
    T_brst   = np.empty((2,len(epn),len(flr)), dtype = object)
    
    kk = 0
    for k in epn:
        fdn = rfdn + k + '/'
        spd_f   = np.load(fdn+'tracking/'+'spd_f.npy');
        idx_clean_f   = np.load(fdn+'tracking/'+'idx_clean_f.npy');
        idx_clean_cwt = np.load(fdn+'tracking/'+'idx_clean_cwt.npy')
        idx_clean   = np.logical_and(idx_clean_f, idx_clean_cwt)
        ts_f = np.load(fdn+'tracking/'+'ts_f.npy');
        T_ts = len(ts_f[idx_clean])/fps/60
        
        i = subset[i_ref]
        chn = str(i//10)+str(i%10)
        Y_f = np.load(fdn + mode+'_'+str(fs_out)+'/ch.'+chn+'.fps.npy').T
        y_fit = 10**(x_fit*slope[i_ref,kk] + intercept[i_ref,kk]);     
        Y_f_fit_ref = (Y_f.T/y_fit)

        
        jj = 0
        for i in subset:
            chn = str(i//10)+str(i%10)
            Y_f = np.load(fdn + mode+'_'+str(fs_out)+'/ch.'+chn+'.fps.npy').T
            y_fit = 10**(x_fit*slope[jj,kk] + intercept[jj,kk]);     
            Y_f_fit = (Y_f.T/y_fit)
            for ii in np.arange(len(flr)):
                fl = flr[ii];     fh = fhr[ii]
                idx_band =  np.logical_and(f>=fl, f<=fh)
                f_win = np.hanning(len(f[idx_band]))
                P_f_ref = np.matmul(Y_f_fit_ref[:,idx_band],f_win)/np.sum(f_win)
                md_f    = np.median(P_f_ref[idx_clean]);
                std_f   = np.std(P_f_ref[idx_clean]);
                

                P_f = np.matmul(Y_f_fit[:,idx_band],f_win)/np.sum(f_win)
                index, T = threshold_burst(P_f, md_f+std_f*2, md_f+std_f*3, idx_clean, merge)
                idx_brst [jj, kk, ii] = index
                T_brst [jj, kk, ii]   = T
            jj = jj+1
        kk = kk+1








def threshold_burst(data, thr1, thr2, idx_clean, merge):
    l_t   = len(data)
    idx_t = np.arange(1,l_t)
    
    idx   = np.zeros([l_t,])
    idx[data>=thr1] = 1

    idx_start_stop = np.diff(idx);
    t_idx_thr1_start = idx_t[idx_start_stop==1]
    t_idx_thr1_stop  = idx_t[idx_start_stop==-1]
    if len(t_idx_thr1_stop) > 2:
        if t_idx_thr1_stop[0]  < t_idx_thr1_start[0]:
            t_idx_thr1_stop = np.delete(t_idx_thr1_stop, 0)
        if t_idx_thr1_stop[-1] < t_idx_thr1_start[-1]:
            t_idx_thr1_start = np.delete(t_idx_thr1_start, -1)
    l_thr1_start = len(t_idx_thr1_start);
    l_thr1_stop  = len(t_idx_thr1_stop);
    T_1 = t_idx_thr1_stop - t_idx_thr1_start

    idx   = np.zeros([l_t,])
    idx[data>=thr2] = 1
    idx_thr1 = t_idx_thr1_start<0
    ii = 0
    for i in t_idx_thr1_start:
        idx_thr1[ii] = np.sum(idx[i:i+T_1[ii]])>0 and np.sum(idx_clean[i:i+T_1[ii]])==T_1[ii]
        ii = ii + 1
        
        
        
    T_1     = T_1[idx_thr1]
    t_start = t_idx_thr1_start[idx_thr1]
    
####### merge close burst ##########
    
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


def unique_burst(T_l, T_i, idx_l, idx_i, l_t, gap):

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