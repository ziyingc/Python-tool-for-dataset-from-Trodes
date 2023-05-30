import struct; import os; import math; import numpy as np; import subprocess
from itertools import combinations
from scipy.signal import butter, bessel, filtfilt, freqz, firwin
import scipy.signal as signal
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.cm as cm; import matplotlib.mlab as mlab
from sklearn.decomposition import PCA

import read_videoPosTrack as VPT
import nelpy as nel
from scipy.ndimage.filters import gaussian_filter;




def movement_segment(rfdn, epn, fps, fs_out):
    
    j = 0;
    total_acc = 0;
    total_dec = 0;
    dispm_ac_Toi2 = np.empty([0,])
    dispm_de_Toi2 = np.empty([0,])
    for k in epn:
        print(k[9:21])
        fdn   = rfdn + k + '/'
        ts_f  = np.load(fdn+'tracking/'+'ts_f.npy');
        spd_f = np.load(fdn+'tracking/'+'spd_f.npy');
        
        XY    = np.load(fdn+'tracking/'+'pos_Y_f.npy');
#         pos_Y = np.load(fdn+'tracking/'+'pos_Y_f.npy'); pos_X = np.load(fdn+'tracking/'+'pos_X_f.npy')
        
        TN_max =fps*5;      TN_min =fps*2
        l_t = len(ts_f)
        l_t = 100
        d_XY = np.sum(np.diff(XY, axis = 0)^2,axis=1)^0.5;
        dis = np.empty((0,))
        idx_T = np.empty((0,2))
        while i < l_t-1:
            p_i = XY[i,:];
            for j in np.arange(i+TN_min,i+TN_max):
                p_j = XY[j,:];

                l_xy_ij = np.sum(d_XY[i:j])
                d_ij = p_i-p_j;
                dis = sum(d_ij*d_ij)^0.5

                dis_ij = np.matmul(p_i-p_j);
       
    
    
    
# def plot_hist(y,b):
# def fartherest2points():
#     XY;
#     l_t = np.max(XY.shape[0:2]);
#     d_max = 0
#     for i in np.arange(l_t-1):
#         p_i = XY[i,:]
#         for j in np.arange(i,l_t):
#             p_j = XY[j,:]
#             d_ij = p_i-p_j;
#             dis = sum(d_ij*d_ij)^0.5
    
    
            