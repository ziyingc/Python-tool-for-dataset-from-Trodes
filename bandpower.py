import struct
import numpy as np
from itertools import combinations
from scipy.signal import butter, bessel, filtfilt, freqz, firwin
import scipy.signal as signal
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.cm as cm; import matplotlib.mlab as mlab
import os
import math
import nelpy as nel
import time2frequency as t2f

fps = 30


def read_band_power(rfdn, rats, epn, fs_out, x_fit, T_pre, T1, T2, l_fb, idx_band, idx_bf_b, idx_bf_l, idx_bf_h, idx_bt_1, idx_bt_2):
    t = np.arange(T_pre+T2).astype('int');
    subset    = np.load(rfdn+'tracking/subset_ephy_li.npy');
    slope     = np.load(rfdn+'tracking/slope_epn_ephy'+rats+'.npy')
    intercept = np.load(rfdn+'tracking/intercept_epn_ephy'+rats+'.npy')

    N_tf = len(idx_bt_1)*np.sum(idx_bf_h)
    
    P_acc_l_T1_b = np.empty((0,))
    P_acc_l_T2_b = np.empty((0,))
    P_acc_i_T1_b = np.empty((0,))
    P_acc_i_T2_b = np.empty((0,))
    
    P_acc_l_T1_l = np.empty((0,))
    P_acc_l_T2_l = np.empty((0,))
    P_acc_i_T1_l = np.empty((0,))
    P_acc_i_T2_l = np.empty((0,))
    
    P_acc_l_T1_h = np.empty((0,))
    P_acc_l_T2_h = np.empty((0,))
    P_acc_i_T1_h = np.empty((0,))
    P_acc_i_T2_h = np.empty((0,))
    
    P_acc_l_T1_b_r = np.empty((0,))
    P_acc_l_T2_b_r = np.empty((0,))
    P_acc_i_T1_b_r = np.empty((0,))
    P_acc_i_T2_b_r = np.empty((0,))
    
    P_acc_l_T1_l_r = np.empty((0,))
    P_acc_l_T2_l_r = np.empty((0,))
    P_acc_i_T1_l_r = np.empty((0,))
    P_acc_i_T2_l_r = np.empty((0,))
    
    P_acc_l_T1_h_r = np.empty((0,))
    P_acc_l_T2_h_r = np.empty((0,))
    P_acc_i_T1_h_r = np.empty((0,))
    P_acc_i_T2_h_r = np.empty((0,))
    
    n_k = len(epn)

    idx_k   = np.zeros((n_k,))>1
    n_acc_total = 0
        
    lag_m       = np.load(rfdn+'tracking/lag_m.npy')
    idx_acc_f_T = np.load(rfdn+'tracking/idx_acc_f_T.npy');
    kk = 0
    for k in epn:
        fdn = rfdn + k + '/'        
        ts_f = np.load(fdn+'tracking/'+'ts_f.npy');

        i_ref = 0;   i = subset[i_ref];    chn = str(i//10)+str(i%10)
        Y_f = np.load(fdn + 'FFT_'+str(fs_out)+'/ch.'+chn+'.fps.npy')
        y_fit = 10**(x_fit*slope[i_ref,kk] + intercept[i_ref,kk]);  Y_f_fit_l = (Y_f/y_fit)[:,idx_band]

        i_ref = 1;   i = subset[i_ref];    chn = str(i//10)+str(i%10)
        Y_f = np.load(fdn + 'FFT_'+str(fs_out)+'/ch.'+chn+'.fps.npy')
        y_fit = 10**(x_fit*slope[i_ref,kk] + intercept[i_ref,kk]);  Y_f_fit_i = (Y_f/y_fit)[:,idx_band]

        idx_acc_f   = np.load(fdn+'tracking/idx_acc_f.npy');
        idx_acc_f_r = np.load(fdn+'tracking/idx_acc_f_r.npy');
        T_acc_f     = np.load(fdn+'tracking/T_acc_f.npy');
        idx_acc_f_k = np.load(fdn+'tracking/idx_acc_f_k.npy');
        
        n_acc = len(idx_acc_f)

        if n_acc>0:
            lag           = lag_m      [idx_acc_f_k];
            idx_acc_f_T_k = idx_acc_f_T[idx_acc_f_k];
            lag       = lag[idx_acc_f_T_k]
            T_acc_f   = T_acc_f[idx_acc_f_T_k]
            idx_acc_f = idx_acc_f[idx_acc_f_T_k]
            n_acc = len(idx_acc_f)
            n_acc_total = n_acc_total+n_acc
            if n_acc>0:
                idx_acc_f = idx_acc_f+T_acc_f-lag
                idx_k[kk] = True
                for i in np.arange(len(idx_acc_f)):
                    t_ref = idx_acc_f[i]
                    idx_t_spec = np.arange(t_ref-T_pre,t_ref+T2).astype('int')
                    y_acc_l = Y_f_fit_l[idx_t_spec,:]; 
                    y_acc_i = Y_f_fit_i[idx_t_spec,:];
                    
                    y_p = y_acc_l[idx_bt_1,:][:,idx_bf_b]
                    y_p = np.mean(y_p, axis=0)
                    p = np.mean(y_p); P_acc_l_T1_b = np.append(P_acc_l_T1_b, p)
    
                    y_p = y_acc_l[idx_bt_2,:][:,idx_bf_b]
                    y_p = np.mean(y_p, axis=0)
                    p = np.mean(y_p); P_acc_l_T2_b = np.append(P_acc_l_T2_b, p)
                    
                    y_p = y_acc_i[idx_bt_1,:][:,idx_bf_b]
                    y_p = np.mean(y_p, axis=0)
                    p = np.mean(y_p); P_acc_i_T1_b = np.append(P_acc_i_T1_b, p)
                    
                    y_p = y_acc_i[idx_bt_2,:][:,idx_bf_b]
                    y_p = np.mean(y_p, axis=0)
                    p = np.mean(y_p);  P_acc_i_T2_b = np.append(P_acc_i_T2_b, p)
                    ################################################################
                    y_p = y_acc_l[idx_bt_1,:][:,idx_bf_l]
                    y_p = np.mean(y_p, axis=0)
                    p = np.mean(y_p); P_acc_l_T1_l = np.append(P_acc_l_T1_l, p)
    
                    y_p = y_acc_l[idx_bt_2,:][:,idx_bf_l]
                    y_p = np.mean(y_p, axis=0)
                    p = np.mean(y_p); P_acc_l_T2_l = np.append(P_acc_l_T2_l, p)
                    
                    y_p = y_acc_i[idx_bt_1,:][:,idx_bf_l]
                    y_p = np.mean(y_p, axis=0)
                    p = np.mean(y_p); P_acc_i_T1_l = np.append(P_acc_i_T1_l, p)
                    
                    y_p = y_acc_i[idx_bt_2,:][:,idx_bf_l]
                    y_p = np.mean(y_p, axis=0)
                    p = np.mean(y_p);  P_acc_i_T2_l = np.append(P_acc_i_T2_l, p)
                    ################################################################
                    y_p = y_acc_l[idx_bt_1,:][:,idx_bf_h]
                    y_p = np.mean(y_p, axis=0)
                    p = np.mean(y_p); P_acc_l_T1_h = np.append(P_acc_l_T1_h, p)
                    
                    y_p = y_acc_l[idx_bt_2,:][:,idx_bf_h]
                    y_p = np.mean(y_p, axis=0)
                    p = np.mean(y_p); P_acc_l_T2_h = np.append(P_acc_l_T2_h, p)
                    
                    y_p = y_acc_i[idx_bt_1,:][:,idx_bf_h]
                    y_p = np.mean(y_p, axis=0)
                    p = np.mean(y_p); P_acc_i_T1_h = np.append(P_acc_i_T1_h, p)
                    
                    y_p = y_acc_i[idx_bt_2,:][:,idx_bf_h]
                    y_p = np.mean(y_p, axis=0)
                    p = np.mean(y_p); P_acc_i_T2_h = np.append(P_acc_i_T2_h, p)


                for i in np.arange(len(idx_acc_f_r)):
                    t_ref = idx_acc_f_r[i]
                    idx_t_spec = np.arange(t_ref-T_pre,t_ref+T2).astype('int')
                    y_acc_l = Y_f_fit_l[idx_t_spec,:]; y_acc_i = Y_f_fit_i[idx_t_spec,:];
                    
                    
                    y_p = y_acc_l[idx_bt_1,:][:,idx_bf_b]
                    y_p = np.mean(y_p, axis=0)
                    p = np.mean(y_p); P_acc_l_T1_b_r = np.append(P_acc_l_T1_b_r, p)
    
                    y_p = y_acc_l[idx_bt_2,:][:,idx_bf_b]
                    y_p = np.mean(y_p, axis=0)
                    p = np.mean(y_p); P_acc_l_T2_b_r = np.append(P_acc_l_T2_b_r, p)
                    
                    y_p = y_acc_i[idx_bt_1,:][:,idx_bf_b]
                    y_p = np.mean(y_p, axis=0)
                    p = np.mean(y_p); P_acc_i_T1_b_r = np.append(P_acc_i_T1_b_r, p)
                    
                    y_p = y_acc_i[idx_bt_2,:][:,idx_bf_b]
                    y_p = np.mean(y_p, axis=0)
                    p = np.mean(y_p);  P_acc_i_T2_b_r = np.append(P_acc_i_T2_b_r, p)
                    ################################################################
                    y_p = y_acc_l[idx_bt_1,:][:,idx_bf_l]
                    y_p = np.mean(y_p, axis=0)
                    p = np.mean(y_p); P_acc_l_T1_l_r = np.append(P_acc_l_T1_l_r, p)
    
                    y_p = y_acc_l[idx_bt_2,:][:,idx_bf_l]
                    y_p = np.mean(y_p, axis=0)
                    p = np.mean(y_p); P_acc_l_T2_l_r = np.append(P_acc_l_T2_l_r, p)
                    
                    y_p = y_acc_i[idx_bt_1,:][:,idx_bf_l]
                    y_p = np.mean(y_p, axis=0)
                    p = np.mean(y_p); P_acc_i_T1_l_r = np.append(P_acc_i_T1_l_r, p)
                    
                    y_p = y_acc_i[idx_bt_2,:][:,idx_bf_l]
                    y_p = np.mean(y_p, axis=0)
                    p = np.mean(y_p);  P_acc_i_T2_l_r = np.append(P_acc_i_T2_l_r, p)
                    ################################################################
                    y_p = y_acc_l[idx_bt_1,:][:,idx_bf_h]
                    y_p = np.mean(y_p, axis=0)
                    p = np.mean(y_p); P_acc_l_T1_h_r = np.append(P_acc_l_T1_h_r, p)
                    
                    y_p = y_acc_l[idx_bt_2,:][:,idx_bf_h]
                    y_p = np.mean(y_p, axis=0)
                    p = np.mean(y_p); P_acc_l_T2_h_r = np.append(P_acc_l_T2_h_r, p)
                    
                    y_p = y_acc_i[idx_bt_1,:][:,idx_bf_h]
                    y_p = np.mean(y_p, axis=0)
                    p = np.mean(y_p); P_acc_i_T1_h_r = np.append(P_acc_i_T1_h_r, p)
                    
                    y_p = y_acc_i[idx_bt_2,:][:,idx_bf_h]
                    y_p = np.mean(y_p, axis=0)
                    p = np.mean(y_p); P_acc_i_T2_h_r = np.append(P_acc_i_T2_h_r, p)
        kk += 1
    return P_acc_l_T1_b, P_acc_l_T2_b, P_acc_i_T1_b, P_acc_i_T2_b, \
           P_acc_l_T1_l, P_acc_l_T2_l, P_acc_i_T1_l, P_acc_i_T2_l, \
           P_acc_l_T1_h, P_acc_l_T2_h, P_acc_i_T1_h, P_acc_i_T2_h, \
           P_acc_l_T1_b_r, P_acc_l_T2_b_r, P_acc_i_T1_b_r, P_acc_i_T2_b_r, \
           P_acc_l_T1_l_r, P_acc_l_T2_l_r, P_acc_i_T1_l_r, P_acc_i_T2_l_r, \
           P_acc_l_T1_h_r, P_acc_l_T2_h_r, P_acc_i_T1_h_r, P_acc_i_T2_h_r






