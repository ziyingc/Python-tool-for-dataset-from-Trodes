import struct
import numpy as np
import subprocess
from itertools import combinations
from scipy.signal import butter, bessel, filtfilt, freqz, firwin
import scipy.signal as signal
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.cm as cm; import matplotlib.mlab as mlab
from sklearn.decomposition import PCA
import os
import math
import read_videoPosTrack as VPT
import nelpy as nel
from scipy.ndimage.filters import gaussian_filter;


def angularchange(v_ft_X1, v_ft_Y1, v_ft_X2, v_ft_Y2):
    x1 = math.atan(v_ft_Y1/v_ft_X1)*180/np.pi;
    x2 = math.atan(v_ft_Y2/v_ft_X2)*180/np.pi;
    if v_ft_X1 < 0: x1 = x1+180
    if v_ft_X2 < 0: x2 = x2+180
    delta = x2 - x1
    return delta, x1, x2

def read_video(rfdn, epn, fs, fps, sigma):
    j = 0
    for k in epn: 
        fn = rfdn + k + '/' + k + '.1.videoPositionTracking'
        timestamps, x1, y1, x2, y2 = VPT.read_VPT(fn)
        fdn = rfdn + k + '/'
        pos = nel.PositionArray(np.vstack((x1, y1)), timestamps=np.array(timestamps)/fs, fs = fps, merge_sample_gap = 100)
        print(fs, fps, fn)
        pos = pos.smooth(sigma = sigma, Kalman=False)
        timestamps = pos.time;
        X = pos._ydata_colsig   # we access the
        pca = PCA(n_components=2);   XY = pca.fit_transform(X);    XY = pca.inverse_transform(XY)
        XY[:,1] = 480 - XY[:,1];
        ##################
        if not os.path.exists(fdn+'tracking'): 
            os.makedirs(fdn+'tracking')
        ##################
        np.save(fdn+'tracking/position2D_frame.npy', XY);
        np.save(fdn+'tracking/time_frame.npy', timestamps);
        pos_ft_X = XY[:,0];  pos_X = nel.PositionArray(XY[:,0], timestamps=pos.time, support=pos.support, fs=fps)
        pos_ft_Y = XY[:,1];  pos_Y = nel.PositionArray(XY[:,1], timestamps=pos.time, support=pos.support, fs=fps)
        position_X, spd_X  = pos_X._kalman_smoother(n_iter=20); np.save(fdn+'tracking/spd_X_kal.npy', spd_X);
        position_Y, spd_Y  = pos_Y._kalman_smoother(n_iter=20); np.save(fdn+'tracking/spd_Y_kal.npy', spd_Y);
        np.save(fdn+'tracking/pos_Y.npy', position_Y);
        np.save(fdn+'tracking/pos_X.npy', position_X);

def measurement(rfdn, epn, epn_id, T):
    r_p2cm_w = np.empty((0,))
    r_p2cm_l = np.empty((0,))
    r_p2cm = np.empty((0,))
    kk = 0
    for k in epn:
        fdn = rfdn + k + '/'
        XY  = np.load(fdn+'tracking/'+'position2D_frame.npy');
        pos_Y = XY[:,1];   pos_X = XY[:,0];
        Y_max = np.mean(pos_Y[pos_Y.argsort()[-T*30:][::-1]])
        Y_min = np.mean(pos_Y[pos_Y.argsort()[0:T*30]])
        X_max = np.mean(pos_X[pos_X.argsort()[-T*30:][::-1]])
        X_min = np.mean(pos_X[pos_X.argsort()[0:T*30]])
        w = Y_max-Y_min
        l = X_max-X_min
        
        if epn_id[kk]==1:
            if l > 410 and w > 180:
                r_p2cm_w = np.append(r_p2cm_w, 20/w*2.54)
                r_p2cm_l = np.append(r_p2cm_l, 43/l*2.54)
        elif epn_id[kk]==2:
            if w > 410 and l > 180:
                r_p2cm_w = np.append(r_p2cm_w, 43/w*2.54)
                r_p2cm_l = np.append(r_p2cm_w, 20/l*2.54)
        elif epn_id[kk]==0:
            LT_length = np.sqrt(np.square(w)+np.square(l))
            if LT_length>390:
                LT_length = LT_length+20
                r_p2cm = np.append(r_p2cm, 46/LT_length*2.54)
        kk += 1
    return r_p2cm_w, r_p2cm_l, r_p2cm

def raw2mp4(rfdn, epn, fps):
    if not os.path.exists(rfdn+'videos'):
        os.makedirs(rfdn+'videos')
    for k in epn: 
        fn = rfdn + k + '/' + k
        subprocess.call(['ffmpeg', '-r', str(fps),
                         '-i', fn + '.1.h264',
                         '-codec', 'copy', rfdn+'videos/'+k + '.1.mp4'])

def match_fps2fs_out(rfdn, epn, epn_id, fps, fs_out, N):
    r_p2cm   = np.load('Rats/r_p2cm.npy')
    r_p2cm_w = np.load('Rats/r_p2cm_w.npy')
    r_p2cm_l = np.load('Rats/r_p2cm_l.npy')
    kk = 0
    for k in epn:
        fdn = rfdn + k + '/'
        timestamps = np.load(fdn+'tracking/'+'time_frame.npy')
        ts_ds      = np.load(fdn+'NPY_'+str(fs_out)+'/ts_ds.npy');l_t = len(ts_ds)

        idx = np.empty([0,])
        for i in timestamps:
            idx_temp = np.searchsorted(ts_ds, i);
            if idx_temp > 0 and idx_temp < l_t:
                d_ts_1 = i-ts_ds[idx_temp-1]
                d_ts_2 = ts_ds[idx_temp]-i
                if d_ts_1>d_ts_2:
                    idx = np.append(idx, idx_temp);
                else:
                    idx = np.append(idx, idx_temp-1);
            else:
                idx = np.append(idx, idx_temp);
        idx_vb = np.logical_and(idx>N, idx<l_t-N);
        np.save(fdn+'tracking/'+'idx_vb.npy',  idx_vb)
        np.save(fdn+'tracking/'+'idx_ds2f.npy',idx.astype('int'))
        #################################

        idx = idx[idx_vb]
        idx = idx.astype('int');
        if epn_id[kk]>0:
            pos_Y     = (np.load(fdn+'tracking/' + 'pos_Y.npy').squeeze())*r_p2cm_w;
            pos_X     = (np.load(fdn+'tracking/' + 'pos_X.npy').squeeze())*r_p2cm_l;
            spd_Y_kal = np.load(fdn+'tracking/' + 'spd_Y_kal.npy')*r_p2cm_w;
            spd_X_kal = np.load(fdn+'tracking/' + 'spd_X_kal.npy')*r_p2cm_l;
            spd_kal = np.sqrt(np.square(spd_X_kal)+np.square(spd_Y_kal));
        elif epn_id[kk]==0:
            pos_Y     = (np.load(fdn+'tracking/' + 'pos_Y.npy').squeeze())*r_p2cm;
            pos_X     = (np.load(fdn+'tracking/' + 'pos_X.npy').squeeze())*r_p2cm;
            spd_Y_kal = np.load(fdn+'tracking/' + 'spd_Y_kal.npy');
            spd_X_kal = np.load(fdn+'tracking/' + 'spd_X_kal.npy');
            spd_kal = np.sqrt(np.square(spd_X_kal)+np.square(spd_Y_kal))*r_p2cm;
        
        
        np.save(fdn+'tracking/' + 'spd_kal.npy', spd_kal);

#         N_win = 5
#         f_win = np.hanning(N_win)
#         f_win = f_win/np.sum(f_win)
#         spd_Y = np.gradient(pos_Y)/(1/fps); spd_Y = np.convolve(spd_Y, f_win, mode='same')
#         spd_X = np.gradient(pos_X)/(1/fps); spd_X = np.convolve(spd_X, f_win, mode='same')
#         spd = np.sqrt (np.square(spd_X) + np.square(spd_Y));
#         np.save(fdn+'tracking/'+'spd_X.npy', spd_X);   
#         np.save(fdn+'tracking/'+'spd_Y.npy', spd_Y)
#         np.save(fdn+'tracking/'+'spd.npy',   spd)
#         spd_Y_f    = spd_Y[idx_vb];      np.save(fdn+'tracking/'+'spd_Y_f.npy', spd_Y_f)
#         spd_X_f    = spd_X[idx_vb];      np.save(fdn+'tracking/'+'spd_X_f.npy', spd_X_f)
#         spd_f      = spd[idx_vb];        np.save(fdn+'tracking/'+'spd_f.npy', spd_f)
        
        ts_f       = timestamps[idx_vb]; np.save(fdn+'tracking/'+'ts_f.npy', ts_f)
        pos_Y_f    = pos_Y[idx_vb];      np.save(fdn+'tracking/'+'pos_Y_f.npy', pos_Y_f)
        pos_X_f    = pos_X[idx_vb];      np.save(fdn+'tracking/'+'pos_X_f.npy', pos_X_f)
        spd_kal_f  = spd_kal[idx_vb];    np.save(fdn+'tracking/'+'spd_kal_f.npy', spd_kal_f)
        kk = kk + 1
        
def ac_de_transit(rfdn, epn, fps, fs_out, v_max_thr, v_max2_thr, v_min_thr, dispm_min_thr, dispm_max_thr, Toi1, Toi2, buffer, test):

    f_win = np.ones((buffer,))/buffer    
    T_seg = np.array([Toi1+Toi2]).astype('int')
    
    j = 0;
    total_acc = 0;
    total_dec = 0;
    dispm_ac_Toi2 = np.empty([0,])
    dispm_de_Toi2 = np.empty([0,])
    for k in epn:
        fdn   = rfdn + k + '/'
        if test:
            ts_f  = np.load(fdn+'tracking/'+'time_frame.npy');
            spd_f = np.load(fdn+'tracking/'+'spd.npy');
            pos_Y = np.load(fdn+'tracking/'+'pos_Y.npy')
            pos_X = np.load(fdn+'tracking/'+'pos_X.npy')
            l_t = len(ts_f); idx_t = np.ones((l_t,))>0;
        else:
            ts_f  = np.load(fdn+'tracking/'+'ts_f.npy');
            spd_f = np.load(fdn+'tracking/'+'spd_f.npy'); 
            spd_f = np.convolve(spd_f, f_win, mode='same')
            
            pos_Y = np.load(fdn+'tracking/'+'pos_Y_f.npy')
            pos_X = np.load(fdn+'tracking/'+'pos_X_f.npy')
            
            idx_clean_fft = np.load(fdn+'tracking/'+'idx_clean_fft.npy')
            idx_clean_f = np.load(fdn+'tracking/'+'idx_clean_f.npy')
            idx_clean = np.logical_and(idx_clean_f, idx_clean_fft)
            idx_ds2f  = np.load(fdn+'tracking/'+'idx_ds2f.npy')

            l_t = len(ts_f); idx_t = np.ones((l_t,))>0;
            idx_t_r = (np.arange(l_t))[idx_clean]; idx_t_r = idx_t_r[idx_t_r<l_t-Toi1-Toi2]

            t_ds_acc1 = np.empty((0,0));  t_ds_acc2 = np.empty((0,0));
            t_ds_dec1 = np.empty((0,0));  t_ds_dec2 = np.empty((0,0));
            
            
        t_acc = np.empty((0,0)); t_dec = np.empty((0,0));
        d_acc = np.empty((0,0)); d_dec = np.empty((0,0));
        for TN in T_seg:
            i = 0
            while i < len(pos_X)-TN:
                if test:
                    n_clean = Toi1+Toi2
                else:
                    n_clean = np.sum(idx_clean[i:(i+Toi1+Toi2)])
                if n_clean == Toi1+Toi2:
                    d2d = [np.max(pos_Y[i:(i+Toi1+Toi2)])-np.min(pos_Y[i:(i+Toi1+Toi2)]),
                           np.max(pos_X[i:(i+Toi1+Toi2)])-np.min(pos_X[i:(i+Toi1+Toi2)])]
                    dmax = np.max(d2d); dmin = np.min(d2d)
                    if dmax >= dispm_max_thr:
                        v_ac_max2 = np.max (spd_f[i:(i+Toi2)]);
                        v_ac_max1 = np.max (spd_f[(i+Toi2):(i+Toi2+int(Toi1))])
                        v_ac_min1 = np.min (spd_f[(i+Toi2):(i+Toi2+int(Toi1))])
                        
                        d_X_max = np.max(pos_X[i:(i+Toi2)])- np.min(pos_X[i:(i+Toi2)]);   
                        d_Y_max = np.max(pos_Y[i:(i+Toi2)])- np.min(pos_Y[i:(i+Toi2)]);
                        d_ac    = np.max([d_X_max,d_Y_max])
                        
                        v_de_max2  = np.max (spd_f[(i+Toi1):(i+Toi1+Toi2)])
                        v_de_max1  = np.max (spd_f[(i):(i+Toi1)]);
                        v_de_min1  = np.min (spd_f[(i):(i+Toi1)]);
                        
                        d_X_max = np.max(pos_X[(i+Toi1):(i+Toi1+Toi2)])- np.min(pos_X[(i+Toi1):(i+Toi1+Toi2)]);   
                        d_Y_max = np.max(pos_Y[(i+Toi1):(i+Toi1+Toi2)])- np.min(pos_Y[(i+Toi1):(i+Toi1+Toi2)]);
                        d_de    = np.max([d_X_max, d_Y_max])

                        if   v_ac_min1>v_min_thr and v_ac_max1>v_max2_thr and v_ac_max2<=v_max_thr and d_ac <= dispm_min_thr:
                            idx_t[i:i+TN] = False
                            t_acc = np.append(t_acc, i);

                            if not(test):
                                t_ds_acc1 = np.append(t_ds_acc1, idx_ds2f[i]);
                                t_ds_acc2 = np.append(t_ds_acc2, idx_ds2f[i+TN]);
                            t_seg = np.arange(i,i+TN).astype('int')
                            dispm_ac_Toi2 = np.append(dispm_ac_Toi2, d_ac)
                            i += TN
                            
                        elif v_de_min1>v_min_thr and v_de_max1>v_max2_thr and v_de_max2<=v_max_thr and d_de <= dispm_min_thr:
                            idx_t[i:i+TN] = False
                            t_dec = np.append(t_dec, i);
                            if not(test):
                                t_ds_dec1 = np.append(t_ds_dec1, idx_ds2f[i])
                                t_ds_dec2 = np.append(t_ds_dec2, idx_ds2f[i+TN])
                            t_seg = np.arange(i,i+TN).astype('int')
                            dispm_de_Toi2 = np.append(dispm_de_Toi2, d_de)
                            i += TN

                        else:
                            i += 1
                    else:
                        i += 1
                else:
                    i += 1


        t_acc = t_acc.astype('int'); l_acl = len(t_acc); total_acc = total_acc + l_acl
        t_dec = t_dec.astype('int'); l_dcl = len(t_dec); total_dec = total_dec + l_dcl
        if test:
            np.save(fdn + 'tracking/t_acc_f.npy', t_acc);
            np.save(fdn + 'tracking/t_dec_f.npy', t_dec);
        else:
            t_ds_acc1 = t_ds_acc1.astype('int');    
            t_ds_dec1 = t_ds_dec1.astype('int');
            np.save(fdn + 'FFT_' +str(fs_out)+'/t_acc_f.npy', t_acc);
            np.save(fdn + 'FFT_' +str(fs_out)+'/t_dec_f.npy', t_dec);
            
#             t_acc_r = t_acc_r.astype('int');t_ds_acc2 = t_ds_acc2.astype('int')
#             t_dec_r = t_dec_r.astype('int');t_ds_dec2 = t_ds_dec2.astype('int')
#             i = 0
#             while i < l_acl*50:
#                 t_ref = np.random.choice(idx_t_r, 1)[0]
#                 idx_t_spec = np.arange(t_ref,t_ref+Toi1+Toi2).astype('int')
#                 if np.sum(idx_clean[idx_t_spec])==Toi1+Toi2:
#                     t_acc_r = np.append(t_acc_r, t_ref);
#                     i = i+1
#             i = 0
#             while i < l_dcl*50:
#                 t_ref = np.random.choice(idx_t_r, 1)[0]
#                 idx_t_spec = np.arange(t_ref,t_ref+Toi1+Toi2).astype('int')
#                 if np.sum(idx_clean[idx_t_spec])==Toi1+Toi2:
#                     t_dec_r = np.append(t_dec_r, t_ref);
#                     i = i+1
#             np.save(fdn + 'FFT_' +str(fs_out)+'/t_ds_acc1_f.npy', t_ds_acc1);
#             np.save(fdn + 'FFT_' +str(fs_out)+'/t_ds_dec1_f.npy', t_ds_dec1);
#             np.save(fdn + 'FFT_' +str(fs_out)+'/t_ds_acc2_f.npy', t_ds_acc2);
#             np.save(fdn + 'FFT_' +str(fs_out)+'/t_ds_dec2_f.npy', t_ds_dec2);
#             np.save(fdn + 'FFT_' +str(fs_out)+'/t_acc_r_f.npy', t_acc_r);
#             np.save(fdn + 'FFT_' +str(fs_out)+'/t_dec_r_f.npy', t_dec_r);
    print('acc:', total_acc, ', dec:', total_dec)
    
    
    
def ac_de_transit_filter(rfdn, epn, fps, fs_out, Toi1, Toi2, buffer, test):
    
    idx_t_spec = np.arange(buffer).astype('int');
    j = 0;
    total_acc = 0;
    total_dec = 0;
    dispm_ac_Toi2 = np.empty([0,])
    dispm_de_Toi2 = np.empty([0,])
    for k in epn:
        fdn   = rfdn + k + '/'
        if test:
            idx_peak_acl   = np.load(fdn + 'tracking/t_acc_f.npy');
            idx_peak_dcl   = np.load(fdn + 'tracking/t_dec_f.npy');
            pos_Y       = np.load(fdn+'tracking/'+'pos_Y.npy')
            pos_X       = np.load(fdn+'tracking/'+'pos_X.npy')
        else:
            idx_peak_acl   = np.load(fdn + 'FFT_'+str(fs_out) + '/t_acc_f.npy');
            idx_peak_dcl   = np.load(fdn + 'FFT_'+str(fs_out) + '/t_dec_f.npy');
            pos_Y       = np.load(fdn+'tracking/'+'pos_Y_f.npy')
            pos_X       = np.load(fdn+'tracking/'+'pos_X_f.npy')
            v_ft_Y = np.load(fdn+'tracking/'+'spd_Y_f.npy')
            v_ft_X = np.load(fdn+'tracking/'+'spd_X_f.npy')

        idx_ac = np.empty((0,));   
        idx_dc = np.empty((0,));
        if len(idx_peak_acl)>0:
            for i in np.arange(len(idx_peak_acl)):
                t_ref       = idx_peak_acl[i]
                idx_t_spec0 = idx_t_spec + t_ref + Toi2
                idx_t_spec1 = t_ref + Toi2 + Toi1 - idx_t_spec
                v_x1 = np.mean(v_ft_X[idx_t_spec0]);   v_x2 = np.mean(v_ft_X[idx_t_spec1])
                v_y1 = np.mean(v_ft_Y[idx_t_spec0]);   v_y2 = np.mean(v_ft_Y[idx_t_spec1])
                delta, x1, x2 = angularchange(v_x1, v_y1, v_x2, v_y2)
                if delta < 20 and delta > -20:
                    idx_ac = np.append(idx_ac,1)
                elif delta > 70 or delta < -70:
                    idx_ac = np.append(idx_ac,0)
                else:
                    idx_ac = np.append(idx_ac,2)
        if len(idx_peak_dcl)>0:
            for i in np.arange(len(idx_peak_dcl)):
                t_ref       = idx_peak_dcl[i]
                idx_t_spec0 = idx_t_spec + t_ref
                idx_t_spec1 = t_ref + Toi1 - idx_t_spec 
                v_x1 = np.mean(v_ft_X[idx_t_spec0]);   v_x2 = np.mean(v_ft_X[idx_t_spec1])
                v_y1 = np.mean(v_ft_Y[idx_t_spec0]);   v_y2 = np.mean(v_ft_Y[idx_t_spec1])
                delta, x1, x2 = angularchange(v_x1, v_y1, v_x2, v_y2)
                if delta < 20 and delta > -20:
                    idx_dc = np.append(idx_dc,1)
                elif delta > 70 or delta < -70:
                    idx_dc = np.append(idx_dc,0)
                else:
                    idx_dc = np.append(idx_dc,2)
        if test:
            np.save(fdn + 'tracking/idx_ac_f.npy', idx_ac);
            np.save(fdn + 'tracking/idx_dc_f.npy', idx_dc);
        else:
            np.save(fdn + 'FFT_' +str(fs_out)+'/idx_ac_f.npy', idx_ac);
            np.save(fdn + 'FFT_' +str(fs_out)+'/idx_dc_f.npy', idx_dc);