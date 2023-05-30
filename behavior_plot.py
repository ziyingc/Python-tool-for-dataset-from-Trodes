import struct
import numpy as np
from itertools import combinations
from scipy.signal import butter, bessel, filtfilt, freqz, firwin
import scipy.signal as signal
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.cm as cm; import matplotlib.mlab as mlab
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
import os
import math
import read_videoPosTrack as VPT
import nelpy as nel
from scipy.ndimage.filters import gaussian_filter;

def angularchange(v_ft_X1, v_ft_Y1, v_ft_X2, v_ft_Y2):
    x1 = math.atan(v_ft_Y1/v_ft_X1)*180/np.pi;
    x2 = math.atan(v_ft_Y2/v_ft_X2)*180/np.pi;
    if v_ft_X1<0:
        x1 = x1+180
    if v_ft_X2<0:
        x2 = x2+180
    delta = x2 - x1
    return delta, x1, x2

def plot_tracks(rfdn, epn, epn_id, fs_out):
    r = int(np.sqrt(len(epn)))+1
    c = int(np.sqrt(len(epn)))+1
    fig = plt.figure(1, figsize=(c*4, r*4))
    gs  = GridSpec(r , c);
    color = ['r','b','k']
    j = 0
    for k in epn:
        fdn   = rfdn + k + '/'
        ts_f = np.load(fdn+'tracking/'+'time_frame.npy');
        XY   = np.load(fdn+'tracking/'+'position2D_frame.npy');
        pos_X = XY[:,0]; pos_Y = XY[:,1]
        ax_idx = gs[j];  ax = fig.add_subplot(ax_idx);
        ax.plot(pos_X, pos_Y, color[epn_id[j]]+'.', ms= .1);
        ax.set_xlim([0,640]); ax.set_ylim([0,480]); 
        ax.tick_params(labelsize=8);
        ax.set_title(k, fontsize = 10);   
        j = j+1

    plt.savefig('pic2/tracks_'+rfdn[9:(len(rfdn)-1)]+'.png')
    plt.clf()

def plot_speed(rfdn, epn, fs_out, subset, track, test):
    r = int((len(epn)+1)/2)

    fig = plt.figure(1, figsize=(40, r*3))
    gs  = GridSpec(r , 2);
    j = 0
    for k in epn:
        fdn  = rfdn + k + '/'
        if test == False:
            ts_f  = np.load(fdn+'tracking'+'/ts_f.npy');    
            spd_f = np.load(fdn+'tracking/'+'spd_f.npy')
        else:
            ts_f  = np.load(fdn+'tracking'+'/time_frame.npy');    
            spd_f = np.load(fdn+'tracking/'+'spd.npy')
        f_win = np.hanning(10);
        
        ax_idx = gs[j];  ax = fig.add_subplot(ax_idx);
        ts_f = ts_f - ts_f[0]
        ax.plot(ts_f[0:2000], spd_f[0:2000], lw = 2);
        spd_f = np.convolve(spd_f, f_win/np.sum(f_win), mode='same')
        ax.plot(ts_f[0:2000], spd_f[0:2000], 'r', lw = 2);
        
        
        ax.set_ylim([0,50])
        ax.set_xlim([ts_f[0],ts_f[2000]]);
        ax.tick_params(labelsize=20);
        ax.set_title(k[10:21], fontsize = 20);   
        j = j+1
    plt.savefig('pic2/speed_'+rfdn[9:(len(rfdn)-1)]+'_'+track+'.png')
    plt.clf()
def plot_speed_burst(rfdn, epn, fs_out, T_spd, subset, mode):
    r = int(len(epn))
    c_txt = ['b*','r*']
    j = 0
    for k in epn:
        fdn  = rfdn + k + '/'

        ts_f  = np.load(fdn+'tracking'+'/ts_f.npy');l_t = len(ts_f)
        ts_f = ts_f - ts_f[0]
        idx_burst = np.zeros((2,l_t))>1
        spd_f = np.load(fdn+'tracking/'+'spd_f.npy')
        ii = 0
        n_b = np.empty((0,))
        for i in subset:
            chn = str(i//10)+str(i%10)
            T     = np.load(fdn + mode+'_'+str(fs_out)+'/ch.'+chn+'.burst.T.npy')
            index = np.load(fdn + mode+'_'+str(fs_out)+'/ch.'+chn+'.burst.npy')
            n_brst = len(T)
            rc  = int(np.sqrt(n_brst))+1
            fig = plt.figure(1, figsize=(rc*3, rc*3))
            gs  = GridSpec(rc , rc);
            n_b = np.append(n_b, n_brst)
            for l in np.arange(n_brst):
                i_brst = index[l];  T_brst = T[l]
                tmin = int(ts_f[i_brst]/60)
                t_txt = str(tmin)+':'+str(int(ts_f[i_brst]-tmin*60))
                if i_brst+T_brst<=l_t and i_brst-T_brst>=0:
                    idx_burst[ii,i_brst:i_brst+T_brst] = True
                    ax_idx = gs[l];  ax = fig.add_subplot(ax_idx);
                    ax.plot(ts_f[i_brst-T_spd:i_brst+T_spd], spd_f[i_brst-T_spd:i_brst+T_spd], 'g', lw = 1); 
                    ax.plot(ts_f[i_brst:i_brst+T_brst],      spd_f[i_brst:i_brst+T_brst], c_txt[ii]);
                    ax.text(ts_f[i_brst], 25,t_txt);
                    ax.set_ylim([-2,30])
            plt.savefig(rfdn+'pic2/'+k+'_burst_'+chn+'.png')
            plt.clf()
            ii = ii+1
        print(k,n_b)
            
#             ax.set_ylim([0,50])
#             ax.set_xlim([ts_f[0],ts_f[2000]]);
#             ax.tick_params(labelsize=20);
#             ax.set_title(k[10:21], fontsize = 20);  
            
def plot_avgspd_burst(rfdn, epn, fs_out, T_spd, subset, mode):
    r = int(len(epn))
    c_txt   = ['b','r']
    chn_txt = ['lesion', 'intact']
    j = 0
    for k in epn:
        fdn  = rfdn + k + '/'

        ts_f  = np.load(fdn+'tracking'+'/ts_f.npy');l_t = len(ts_f)
        ts_f = ts_f - ts_f[0]
        idx_burst = np.zeros((2,l_t))>1
        spd_f = np.load(fdn+'tracking/'+'spd_f.npy')

        fig = plt.figure(1, figsize=(8, 5))
        gs  = GridSpec(1, 1);
        ax_idx = gs[0];  ax = fig.add_subplot(ax_idx);
        
        n_b = np.empty((0,))
        ii = 0
        for i in subset:
            
            chn = str(i//10)+str(i%10)
            T     = np.load(fdn + mode+'_'+str(fs_out)+'/ch.'+chn+'.burst.T.npy')
            index = np.load(fdn + mode+'_'+str(fs_out)+'/ch.'+chn+'.burst.npy')
            n_brst = len(T)
            rc  = int(np.sqrt(n_brst))+1
            speed = np.empty((0,T_spd*2))
            for l in np.arange(n_brst):
                i_brst = index[l];  T_brst = T[l]
                tmin = int(ts_f[i_brst]/60)
                t_txt = str(tmin)+':'+str(int(ts_f[i_brst]-tmin*60))
                if i_brst+T_spd<=l_t and i_brst-T_spd>=0:
                    idx_burst[ii,i_brst:i_brst+T_brst] = True
                    speed = np.vstack((speed, spd_f[i_brst-T_spd:i_brst+T_spd]));
#                     ax.plot(np.arange(T_spd*2), spd_f[i_brst-T_spd:i_brst+T_spd], 'g', lw = 1);

            spd_m = np.mean(speed,axis=0)
            ax.plot(np.arange(T_spd*2), spd_m, c_txt[ii], lw = 1, label = chn_txt[ii]);
            ax.plot(T_spd,              spd_m[T_spd], c_txt[ii]+'*');
            
            n_b = np.append(n_b, (speed.shape)[0])
            ii = ii+1
        ax.set_ylim([-1,10])
        ax.legend()
        fig.suptitle(k[10:21]+' '+str(n_b[0])+' '+str(n_b[1]), fontsize=15)
        plt.savefig(rfdn+'pic2/'+k+'_avgburst.png')
        plt.clf()
def plot_turns(rfdn, epn, t_pre, test = True):
    txt_clr = ['b', 'g']
    total_l = 0;
    total_r = 0;
    fig = plt.figure(1, figsize=(15,5))
    gs  = GridSpec(1 , 3);
    d_tn_l = np.empty((0,))
    d_tn_r = np.empty((0,))
    
    for k in epn:
        fdn = rfdn + k + '/'

        if test:
            ts_f = np.load(fdn+'tracking'+'/time_frame.npy');
            XY = np.load(fdn+'tracking/'+'position2D_frame.npy');
            
            tn3   = np.load(fdn + 'tracking' +'/tn_idx.npy');
            T_tn3 = np.load(fdn + 'tracking' +'/T_tn_idx.npy');
            
            tn2   = np.load(fdn + 'tracking' + '/tn_2.npy');
            T_tn2 = np.load(fdn + 'tracking' + '/T_tn2.npy');
            
            tn   = np.load(fdn + 'tracking' + '/tn.npy');
            T_tn = np.load(fdn + 'tracking' + '/T_tn.npy');
        else:
            idx_clean = np.load(fdn+'tracking/'+'idx_clean.npy')
            ts_f      = np.load(fdn+'tracking'+'/ts_f.npy');    XY = np.load(fdn+'tracking/'+'XY_f.npy');
            tn        = np.load(fdn+'tracking'+'/tn_2_f.npy');T_tn = np.load(fdn+'tracking/'+'T_tn2_f.npy');
            
            tn_idx_r = np.empty((0,));
            l_t = len(ts_f);
            idx_t_r = (np.arange(l_t))[idx_clean]; 
            idx_t_r = idx_t_r[idx_t_r<l_t-t_pre]
        
        N = 3
        pos_Y = XY[:,1];             pos_Y = np.convolve(pos_Y, np.ones((N,))/N, mode='same')
        pos_X = XY[:,0];             pos_X = np.convolve(pos_X, np.ones((N,))/N, mode='same')
        v_ft_Y = np.gradient(pos_Y);v_ft_Y = gaussian_filter(v_ft_Y, sigma=1)
        v_ft_X = np.gradient(pos_X);v_ft_X = gaussian_filter(v_ft_X, sigma=1)
        n_tn = len(tn3)
        
        if n_tn>0:
            for i in np.arange(n_tn):
                TN   = T_tn3[i].astype('int')
                tn_i = tn3[i].astype('int')
                idx_t = np.arange(tn_i,tn_i+TN)
                ax_idx = gs[0,0];    ax = fig.add_subplot(ax_idx);
                ax.plot(pos_X[idx_t], pos_Y[idx_t],  '.', ms=1 );
                ax.plot(pos_X[idx_t[0]],   pos_Y[idx_t[0]], 'r*', ms=5 );
                ax.plot(pos_X[idx_t[-1]], pos_Y[idx_t[-1]], 'k^', ms=5 );
            n_tn = len(tn2)
            for i in np.arange(n_tn):
                TN   = T_tn2[i].astype('int')
                tn_i = tn2[i].astype('int')
                idx_t = np.arange(tn_i,tn_i+TN)
                ax_idx = gs[0,1];    ax = fig.add_subplot(ax_idx);
                ax.plot(pos_X[idx_t], pos_Y[idx_t],   '.', ms=1 );
                ax.plot(pos_X[idx_t[0]],   pos_Y[idx_t[0]], 'r*', ms=5 );
                ax.plot(pos_X[idx_t[-1]], pos_Y[idx_t[-1]], 'k^', ms=5 );
            n_tn = len(tn)
            for i in np.arange(n_tn):
                TN   = T_tn[i].astype('int')
                tn_i = tn[i].astype('int')
                idx_t = np.arange(tn_i,tn_i+TN)
                ax_idx = gs[0,2];    ax = fig.add_subplot(ax_idx);
                ax.plot(pos_X[idx_t], pos_Y[idx_t],  '.', ms=1 );
                ax.plot(pos_X[idx_t[0]],   pos_Y[idx_t[0]], 'r*', ms=5 );
                ax.plot(pos_X[idx_t[-1]], pos_Y[idx_t[-1]], 'k^', ms=5 );         
#     ax_idx = gs[0,0:2];    ax = fig.add_subplot(ax_idx);   
#     ax.set_xlim([50,550]); ax.set_ylim([225,475]); ax.tick_params(labelsize=8);
#     ax.set_title(rfdn, fontsize = 10);   
#     ax_idx = gs[1,0]; ax = fig.add_subplot(ax_idx);
#     ax.hist(d_tn_l);  ax.set_xlim([0,200]); ax.set_ylim([0,40]);ax.set_title('left', fontsize = 10);
#     ax_idx = gs[1,1]; ax = fig.add_subplot(ax_idx);
#     ax.hist(d_tn_r);  ax.set_xlim([0,200]); ax.set_ylim([0,40]);ax.set_title('right', fontsize = 10);
    plt.savefig('pic2/turns_'+rfdn[9:(len(rfdn)-1)]+'.png')
    plt.clf()
                
                
def plot_turnspeed(rfdn, epn, t_pre, test = True):
    
    r_p2cm = np.load('Rats/r_p2cm.npy')
    
    txt_clr = ['b', 'g']
    total_l = 0;
    total_r = 0;
    fig = plt.figure(1, figsize=(16, 8))
    gs  = GridSpec(1 , 1);
    d_tn_l = np.empty((0,))
    d_tn_r = np.empty((0,))
    
    for k in epn:
        fdn = rfdn + k + '/'
        if test:
            ts_f = np.load(fdn+'tracking'+'/time_frame.npy');
            XY = np.load(fdn+'tracking/'+'position2D_frame.npy');
            tn_idx   = np.load(fdn + 'tracking' + '/tn.npy'); 
            T_tn_idx = np.load(fdn + 'tracking' + '/T_tn.npy');
        else:
            
            ts_f      = np.load(fdn+'tracking'+'/ts_f.npy');    
            v_ft_Y = np.load(fdn+'tracking/' + 'spd_Y_f.npy'); pos_Y = np.load(fdn+'tracking/'+'pos_Y_f.npy')
            v_ft_X = np.load(fdn+'tracking/' + 'spd_X_f.npy'); pos_X = np.load(fdn+'tracking/'+'pos_X_f.npy')
            
            tn_idx   = np.load(fdn+'tracking'+'/tn_idx_f.npy');
            T_tn_idx = np.load(fdn+'tracking/'+'T_tn_idx_f.npy');
            idx_tn   = np.load(fdn+'tracking'+'/idx_tn_f.npy');
#             print(len(tn_idx),len(T_tn_idx),len(idx_tn))

        n_tn = len(tn_idx)
        if n_tn>0:
            for i in np.arange(n_tn):
         
                TN   = T_tn_idx[i].astype('int')
                tn_i = tn_idx[i].astype('int')
                idx_tn_i = idx_tn[i].astype('int')
                idx_t = np.arange(tn_i,tn_i+TN)
                tn_mid_i = tn_i+int(TN/2)
                ax_idx = gs[0,0];    ax = fig.add_subplot(ax_idx);

                ax.plot(pos_X[idx_t],  pos_Y[idx_t],  txt_clr[idx_tn_i]+'.', ms=1 );
#                 ax.plot(pos_X[tn_i],     pos_Y[tn_i],     'r*', ms=5);
#                 ax.plot(pos_X[tn_i+TN],  pos_Y[tn_i+TN],  'k^', ms=5);
                ax.plot(pos_X[tn_mid_i], pos_Y[tn_mid_i], 'ro', ms=5);   
    ax_idx = gs[0,0];    ax = fig.add_subplot(ax_idx);     
#     ax.set_xlim([0,150]); ax.set_ylim([50,150]); ax.tick_params(labelsize=8);
    ax.set_title(rfdn, fontsize = 10);
    plt.savefig('pic2/turns_'+rfdn[9:(len(rfdn)-1)]+'.png')
    plt.clf()

def ac_de_lfp(rfdn, epn, fps, fs_out, subset, freqs, fl, fh, buffer, mode):

    txt_tn = ['lesion','intact']
    txt_line = ['r-', 'b-', 'r--']
    Toi1 = np.load('Toi1.npy')*fps;
    Toi2 = np.load('Toi2.npy')*fps;

    f_win = np.ones((buffer,))/buffer

    fig = plt.figure(1, figsize=(8,5))
#     fig.set_figheight(10); fig.set_figwidth(7)
    gs = GridSpec(1, 1);
    
    N_acl = 0; N_dcl = 0;
    jj = 0
    for k in epn:
        fdn = rfdn + k + '/'
        idx_clean_fft = np.load(fdn+'tracking/'+'idx_clean_fft.npy')
        idx_clean_f = np.load(fdn+'tracking/'+'idx_clean_f.npy')
        idx_clean = np.logical_and(idx_clean_f,   idx_clean_fft)
        ts_f = np.load(fdn+'tracking/'+'ts_f.npy');
        v_ft = np.load(fdn+'tracking/'+'spd_f.npy');
        v_ft = np.convolve(v_ft, f_win, mode='same')

        idx_peak_acl   = np.load(fdn + 'FFT_'+str(fs_out) + '/t_acc_f.npy');
        idx_peak_dcl   = np.load(fdn + 'FFT_'+str(fs_out) + '/t_dec_f.npy');

        idx_ac = np.load(fdn + 'FFT_' +str(fs_out)+'/idx_ac_f.npy').astype('int');
        idx_dc = np.load(fdn + 'FFT_' +str(fs_out)+'/idx_dc_f.npy').astype('int');


        if len(idx_peak_acl)>0:
            for i in np.arange(len(idx_peak_acl)):
                t_ref = idx_peak_acl[i]
                lines = txt_line[idx_ac[i]]
                idx_t_spec = np.arange(t_ref,t_ref+Toi1+Toi2).astype('int')
                if np.sum(idx_clean[idx_t_spec])!=Toi1+Toi2:
                    print('artifact used', np.sum(idx_clean[idx_t_spec]),Toi1+Toi2)
                ax_idx = gs[0,0];  ax = fig.add_subplot(ax_idx);

                idx_t_spec = np.arange(t_ref,t_ref+Toi2-9).astype('int')
                ax.plot((idx_t_spec-t_ref)/30 ,v_ft[idx_t_spec], txt_line[0], lw = 0.5)
                
                idx_t_spec = np.arange(t_ref+Toi2-9,t_ref+Toi2).astype('int')
                ax.plot((idx_t_spec-t_ref)/30 ,v_ft[idx_t_spec], txt_line[2], lw = 0.5)
                
                idx_t_spec = np.arange(t_ref+Toi2,t_ref+Toi2+Toi1).astype('int')
                ax.plot((idx_t_spec-t_ref)/30 ,v_ft[idx_t_spec], txt_line[1], lw = 0.5)
                
                N_acl = N_acl+1


#         if len(idx_peak_dcl)>0:
#             idx_t_spec0 = np.arange(buffer,           Toi1-buffer).astype('int')
#             idx_t_spec2 = np.arange(Toi1+buffer, Toi1+Toi2-buffer).astype('int')

#             for i in np.arange(len(idx_peak_dcl)):
#                 t_ref = idx_peak_dcl[i]
#                 lines = txt_line[idx_dc[i]]
#                 idx_t_spec = np.arange(t_ref,t_ref+Toi1+Toi2).astype('int')
#                 if np.sum(idx_clean[idx_t_spec])!=Toi1+Toi2:
#                     print('artifact used', np.sum(idx_clean[idx_t_spec]),Toi1+Toi2)
#                 ax_idx = gs[1,1];  ax = fig.add_subplot(ax_idx);
                
#                 idx_t_spec = np.arange(t_ref,t_ref+Toi1).astype('int')
#                 ax.plot(idx_t_spec-t_ref ,v_ft[idx_t_spec], txt_line[0], lw = 0.5)
                
#                 idx_t_spec = np.arange(t_ref+Toi1+9,t_ref+Toi2+Toi1).astype('int')
#                 ax.plot(idx_t_spec-t_ref ,v_ft[idx_t_spec], txt_line[1], lw = 0.5)

    fig.suptitle(rfdn+' acceleration cases', fontsize=20)
#     ax_idx = gs[2,0];  ax = fig.add_subplot(ax_idx); ax.set_xlim([50, 550]);     ax.set_ylim([150, 600]);
#     ax_idx = gs[2,1];  ax = fig.add_subplot(ax_idx); ax.set_xlim([50, 550]);     ax.set_ylim([150, 600]);
    ax_idx = gs[0,0];  ax = fig.add_subplot(ax_idx);
    ax.set_xlabel('t(sec)');
    ax.set_ylabel('cm/s');
#     ax.set_xlim([0, Toi1+Toi2]); 
#     ax.set_ylim([0, 3]);
#     ax_idx = gs[1,1];  ax = fig.add_subplot(ax_idx); 
#     ax.set_xlim([0, Toi1+Toi2]);
#     ax.set_ylim([0, 3]);
    plt.savefig('pic2/ac_dc_'+rfdn[9:len(rfdn)-1]+'.png')
    plt.clf()