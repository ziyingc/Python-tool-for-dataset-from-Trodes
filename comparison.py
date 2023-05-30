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

def ac_de_lfp(rfdn, epn, fps, fs_out, subset, freqs, fl, fh, buffer, mode):
    if not os.path.exists(rfdn+'test'):
        os.makedirs(rfdn+'test')
    n_f = len(freqs)
    txt_tn = ['lesion','intact']
    txt_line = ['r-', 'b-', 'g-']
    freqs = np.load('Rats/freqs_FFT_'+str(fs_out)+'.npy');
    idx_beta = np.logical_and(freqs>=fl, freqs<=fh)
    f_beta = np.round(freqs[idx_beta],1)
    l_f = len(freqs[idx_beta])
    Toi1 = np.load('Toi1.npy')*fps;
    Toi2 = np.load('Toi2.npy')*fps;
    x_fit = np.log10(freqs)
    
    
    fig = plt.figure(1, figsize=(10,20))
    fig.set_figheight(20); fig.set_figwidth(10)
    gs = GridSpec(5, 2);
    
    if mode == '1/f': 
        slope_l, intercept_l, slope_i, intercept_i, intercept_li = \
        t2f.flicker_fit_fft(rfdn, epn, fs_out, subset, freqs, fl, fh)
    v_acl = np.empty([0,l_f]);
    v_dcl = np.empty([0,l_f]);
    l_0 = int(Toi2)-2*buffer;
    l_2 = int(Toi1)-2*buffer;
    print(l_0, l_2, l_f)
    S_l_acl0 = np.empty([0,l_f]); S_i_acl0 = np.empty([0,l_f])
    S_l_acl2 = np.empty([0,l_f]); S_i_acl2 = np.empty([0,l_f])
    S_l_dcl0 = np.empty([0,l_f]); S_i_dcl0 = np.empty([0,l_f])
    S_l_dcl2 = np.empty([0,l_f]); S_i_dcl2 = np.empty([0,l_f])

    S_l_acl0_r = np.empty([0,l_f]); S_i_acl0_r = np.empty([0,l_f])
    S_l_acl2_r = np.empty([0,l_f]); S_i_acl2_r = np.empty([0,l_f])
    S_l_dcl0_r = np.empty([0,l_f]); S_i_dcl0_r = np.empty([0,l_f])
    S_l_dcl2_r = np.empty([0,l_f]); S_i_dcl2_r = np.empty([0,l_f])

    idx_ac_all = np.empty((0,));
    idx_dc_all = np.empty((0,));
    N_acl = 0; N_dcl = 0;
    jj = 0
    for k in epn:
        fdn = rfdn + k + '/'
        idx_clean_fft = np.load(fdn+'tracking/'+'idx_clean_fft.npy')
        idx_clean_f = np.load(fdn+'tracking/'+'idx_clean_f.npy')
        idx_clean = np.logical_and(idx_clean_f,   idx_clean_fft)
        ts_f = np.load(fdn+'tracking/'+'ts_f.npy');
        i = subset[0]; chn = str(i//10)+str(i%10);
        Sxx_ft_l = np.load(fdn + 'FFT_'+str(fs_out)+'/ch.'+chn+'.fps.npy')
        i = subset[1]; chn = str(i//10)+str(i%10);
        Sxx_ft_i = np.load(fdn + 'FFT_'+str(fs_out)+'/ch.'+chn+'.fps.npy')
        
        if mode == '1/f':
            y_fit = 10**(x_fit*slope_i[jj] + intercept_i[jj]); jj = jj+1
            Sxx_ft = (Sxx_ft_l/y_fit).T;    Sxx_ft_l = Sxx_ft[idx_beta,:]
            Sxx_ft = (Sxx_ft_i/y_fit).T;    Sxx_ft_i = Sxx_ft[idx_beta,:]
        elif mode == 'z':
            S_m = np.mean(Sxx_ft_i[idx_clean,:], axis=0);
            S_s = np.std (Sxx_ft_i[idx_clean,:], axis=0);
            Sxx_ft_i = (((Sxx_ft_i-S_m)/S_s).T)[idx_beta,:]
            Sxx_ft_l = (((Sxx_ft_l-S_m)/S_s).T)[idx_beta,:]
        
        XY   = np.load(fdn+'tracking/'+'XY_f.npy');
        v_ft = np.load(fdn+'tracking/'+'spd_f.npy');

        idx_peak_acl   = np.load(fdn + 'FFT_'+str(fs_out) + '/t_acc_f.npy');
        idx_peak_dcl   = np.load(fdn + 'FFT_'+str(fs_out) + '/t_dec_f.npy');
        idx_peak_acl_r = np.load(fdn + 'FFT_'+str(fs_out) + '/t_acc_r_f.npy');
        idx_peak_dcl_r = np.load(fdn + 'FFT_'+str(fs_out) + '/t_dec_r_f.npy');
        
        idx_ac = np.load(fdn + 'FFT_' +str(fs_out)+'/idx_ac_f.npy').astype('int');
        idx_dc = np.load(fdn + 'FFT_' +str(fs_out)+'/idx_dc_f.npy').astype('int');
        
        idx_ac_all = np.append(idx_ac_all, idx_ac);
        idx_dc_all = np.append(idx_dc_all, idx_dc);

        if len(idx_peak_acl)>0:
            idx_t_spec0 = np.arange(buffer,      int(Toi2)-buffer).astype('int')
            idx_t_spec2 = np.arange(Toi2+buffer, Toi2+Toi1-buffer).astype('int')
            for i in np.arange(len(idx_peak_acl)):
                t_ref = idx_peak_acl[i]
                lines = txt_line[idx_ac[i]]
                idx_t_spec = np.arange(t_ref,t_ref+Toi1+Toi2).astype('int')
                if np.sum(idx_clean[idx_t_spec])!=Toi1+Toi2:
                    print('artifact used', np.sum(idx_clean[idx_t_spec]),Toi1+Toi2)
                ax_idx = gs[1,0];  ax = fig.add_subplot(ax_idx);
                ax.plot(idx_t_spec-idx_t_spec[0] ,v_ft[idx_t_spec])
                
                idx_t_spec = idx_t_spec0+t_ref
                S_l_acl0 = np.vstack((S_l_acl0, np.mean(Sxx_ft_l[:,idx_t_spec],axis=1)))
                S_i_acl0 = np.vstack((S_i_acl0, np.mean(Sxx_ft_i[:,idx_t_spec],axis=1)))
                idx_t_spec = idx_t_spec2+t_ref
                S_l_acl2 = np.vstack((S_l_acl2, np.mean(Sxx_ft_l[:,idx_t_spec],axis=1)))
                S_i_acl2 = np.vstack((S_i_acl2, np.mean(Sxx_ft_i[:,idx_t_spec],axis=1)))
                
                N_acl = N_acl+1
            for i in np.arange(len(idx_peak_acl_r)):
                t_ref = idx_peak_acl_r[i]
                idx_t_spec = idx_t_spec0+t_ref
                S_l_acl0_r = np.vstack((S_l_acl0_r, np.mean(Sxx_ft_l[:,idx_t_spec],axis=1)))
                S_i_acl0_r = np.vstack((S_i_acl0_r, np.mean(Sxx_ft_i[:,idx_t_spec],axis=1)))
                idx_t_spec = idx_t_spec2+t_ref
                S_l_acl2_r = np.vstack((S_l_acl2_r, np.mean(Sxx_ft_l[:,idx_t_spec],axis=1)))
                S_i_acl2_r = np.vstack((S_i_acl2_r, np.mean(Sxx_ft_i[:,idx_t_spec],axis=1)))

        if len(idx_peak_dcl)>0:
            idx_t_spec0 = np.arange(buffer,           Toi1-buffer).astype('int')
            idx_t_spec2 = np.arange(Toi1+buffer, Toi1+Toi2-buffer).astype('int')

            for i in np.arange(len(idx_peak_dcl)):
                t_ref = idx_peak_dcl[i]
                lines = txt_line[idx_dc[i]]
                idx_t_spec = np.arange(t_ref,t_ref+Toi1+Toi2).astype('int')
                if np.sum(idx_clean[idx_t_spec])!=Toi1+Toi2:
                    print('artifact used', np.sum(idx_clean[idx_t_spec]),Toi1+Toi2)
                ax_idx = gs[1,1];  ax = fig.add_subplot(ax_idx);
                ax.plot(idx_t_spec-idx_t_spec[0] ,v_ft[idx_t_spec])
                
                idx_t_spec = idx_t_spec0+t_ref
                S_l_dcl0 = np.vstack((S_l_dcl0, np.mean(Sxx_ft_l[:,idx_t_spec],axis=1)))
                S_i_dcl0 = np.vstack((S_i_dcl0, np.mean(Sxx_ft_i[:,idx_t_spec],axis=1)))
                idx_t_spec = idx_t_spec2+t_ref
                S_l_dcl2 = np.vstack((S_l_dcl2, np.mean(Sxx_ft_l[:,idx_t_spec],axis=1)))
                S_i_dcl2 = np.vstack((S_i_dcl2, np.mean(Sxx_ft_i[:,idx_t_spec],axis=1)))
                N_dcl = N_dcl+1
                
            for i in np.arange(len(idx_peak_dcl_r)):
                t_ref = idx_peak_dcl_r[i]
                idx_t_spec = idx_t_spec0+t_ref
                S_l_dcl0_r = np.vstack((S_l_dcl0_r, np.mean(Sxx_ft_l[:,idx_t_spec],axis=1)))
                S_i_dcl0_r = np.vstack((S_i_dcl0_r, np.mean(Sxx_ft_i[:,idx_t_spec],axis=1)))
                idx_t_spec = idx_t_spec2+t_ref
                S_l_dcl2_r = np.vstack((S_l_dcl2_r, np.mean(Sxx_ft_l[:,idx_t_spec],axis=1)))
                S_i_dcl2_r = np.vstack((S_i_dcl2_r, np.mean(Sxx_ft_i[:,idx_t_spec],axis=1)))

    ax_idx = gs[3,0];  ax = fig.add_subplot(ax_idx);
    d_acl0 = S_l_acl0-S_i_acl0;
    d_acl0_m  = np.mean(d_acl0, axis=0);  d_acl0_s  = np.std(d_acl0, axis=0)
    ax.plot(freqs[idx_beta], d_acl0_m, 'r',   label = 'phase 0')
    ax.fill_between(freqs[idx_beta], d_acl0_m+d_acl0_s, d_acl0_m-d_acl0_s, facecolor='red', alpha=0.5)
    d_acl2 = S_l_acl2-S_i_acl2;
    d_acl2_m  = np.mean(d_acl2, axis=0);  d_acl2_s  = np.std(d_acl2, axis=0)
    ax.plot(freqs[idx_beta], d_acl2_m, 'g',   label = 'phase 2')
    ax.fill_between(freqs[idx_beta], d_acl2_m+d_acl2_s, d_acl2_m-d_acl2_s, facecolor='green', alpha=0.5)
    ax.set_ylim([-1, 2])
    ax.legend(loc = 'upper right')

    ax_idx = gs[3,1];  ax = fig.add_subplot(ax_idx);
    d_dcl0 = S_l_dcl0-S_i_dcl0;
    d_dcl0_m  = np.mean(d_dcl0, axis=0);  d_dcl0_s  = np.std(d_dcl0, axis=0)
    ax.plot(freqs[idx_beta], d_dcl0_m, 'r',   label = 'phase 0')
    ax.fill_between(freqs[idx_beta], d_dcl0_m+d_dcl0_s, d_dcl0_m-d_dcl0_s, facecolor='red', alpha=0.5)
    d_dcl2 = S_l_dcl2-S_i_dcl2;
    d_dcl2_m  = np.mean(d_dcl2, axis=0);  d_dcl2_s  = np.std(d_dcl2, axis=0)
    ax.plot(freqs[idx_beta], d_dcl2_m, 'g',   label = 'phase 2')
    ax.fill_between(freqs[idx_beta], d_dcl2_m+d_dcl2_s, d_dcl2_m-d_dcl2_s, facecolor='green', alpha=0.5)
    ax.set_ylim([-1, 2])
    ax.legend(loc = 'upper right')


    ax_idx = gs[0,0];  ax = fig.add_subplot(ax_idx);
    ax.set_title('acceleration'+' '+str(S_l_acl0.shape[0]));
    ax.plot(freqs[idx_beta], np.mean(S_l_acl0, axis=0), 'r',   label = 'Lesion phase 0')
#     ax.plot(freqs[idx_beta], np.mean(S_l_acl1, axis=0), 'b',   label = 'Lesion phase 1')
    ax.plot(freqs[idx_beta], np.mean(S_l_acl2, axis=0), 'g',   label = 'Lesion phase 2')
    ax.plot(freqs[idx_beta], np.mean(S_i_acl0, axis=0), 'r--')
#     ax.plot(freqs[idx_beta], np.mean(S_i_acl1, axis=0), 'b--')
    ax.plot(freqs[idx_beta], np.mean(S_i_acl2, axis=0), 'g--')
    ax.plot(freqs[idx_beta], np.mean(S_l_acl2_r, axis=0), 'k')
    ax.plot(freqs[idx_beta], np.mean(S_i_acl2_r, axis=0), 'k--')
    ax.legend(loc = 'upper right');ax.set_ylim([-0.5, 2])
    
    ax_idx = gs[0,1];  ax = fig.add_subplot(ax_idx);
    ax.set_title('deceleration'+' '+str(S_l_dcl0.shape[0]));
    ax.plot(freqs[idx_beta], np.mean(S_l_dcl0, axis=0), 'r',   label = 'Lesion phase 0')
#     ax.plot(freqs[idx_beta], np.mean(S_l_dcl1, axis=0), 'b',   label = 'Lesion phase 1')
    ax.plot(freqs[idx_beta], np.mean(S_l_dcl2, axis=0), 'g',   label = 'Lesion phase 2')
    ax.plot(freqs[idx_beta], np.mean(S_i_dcl0, axis=0), 'r--')
#     ax.plot(freqs[idx_beta], np.mean(S_i_dcl1, axis=0), 'b--')
    ax.plot(freqs[idx_beta], np.mean(S_i_dcl2, axis=0), 'g--')
    ax.plot(freqs[idx_beta], np.mean(S_l_dcl2_r, axis=0), 'k')
    ax.plot(freqs[idx_beta], np.mean(S_i_dcl2_r, axis=0), 'k--')
    ax.legend(loc = 'upper right');ax.set_ylim([-0.5, 2])
    
    plt.savefig(rfdn+'pic2/speed_method2_'+str(Toi1)+'.png')
    print(S_l_acl0.shape, S_l_dcl0.shape)
    fig.suptitle(rfdn+' z score normalization', fontsize=20)
#     ax_idx = gs[2,0];  ax = fig.add_subplot(ax_idx); ax.set_xlim([50, 550]);     ax.set_ylim([150, 600]);
#     ax_idx = gs[2,1];  ax = fig.add_subplot(ax_idx); ax.set_xlim([50, 550]);     ax.set_ylim([150, 600]);
#     ax_idx = gs[1,0];  ax = fig.add_subplot(ax_idx); ax.set_xlim([0, Toi1+Toi2]);ax.set_ylim([0, 3]);
#     ax_idx = gs[1,1];  ax = fig.add_subplot(ax_idx); ax.set_xlim([0, Toi1+Toi2]);ax.set_ylim([0, 3]);
    plt.savefig('pic2/ac_dc_'+rfdn[9:len(rfdn)-1]+'.png')
    plt.clf()
    
    
#     fig.set_figheight(10)
#     fig.set_figwidth(18)

#     gs = GridSpec(2,3);
#     ax_idx = gs[0,0];  ax = fig.add_subplot(ax_idx); ax.set_ylabel('PWR differece')
#     f_l_ac0_max = np.zeros((N_acl,)).astype('int');
#     f_l_ac1_max = np.zeros((N_acl,)).astype('int');
#     f_l_ac2_max = np.zeros((N_acl,)).astype('int');
#     d_ac0_max = np.zeros((N_acl,));
#     d_ac1_max = np.zeros((N_acl,));
#     d_ac2_max = np.zeros((N_acl,));

#     for i in np.arange(N_acl):
#         d_acl0 = S_l_acl0[i,:]-S_i_acl0[i,:]; ax.plot(freqs[idx_beta], d_acl0,'r', lw = 0.5)
#         d_acl2 = S_l_acl2[i,:]-S_i_acl2[i,:]; ax.plot(freqs[idx_beta], d_acl2,'g', lw = 0.5)

#         peaks, _ = signal.find_peaks(d_acl0,height=np.mean(d_acl0))
#         if len(peaks)==1:
#             f_l_ac0_max[i] = peaks
#         elif len(peaks)>1:
#             f_l_ac0_max[i] = peaks[np.argmax(d_acl0[peaks])]
#         idx_f = f_l_ac0_max[i]
#         d_ac0_max[i] = d_acl0[idx_f]

#         peaks, _ = signal.find_peaks(d_acl2,height=np.mean(d_acl2))
#         if len(peaks)==1:
#             f_l_ac2_max[i] = peaks

#         elif len(peaks)>1:
#             f_l_ac2_max[i] = peaks[np.argmax(d_acl2[peaks])]
#         idx_f = f_l_ac2_max[i]
#         d_ac2_max[i] = d_acl2[idx_f]

#     t_ac,   p_ac   = stats.ttest_rel(f_l_ac0_max,   f_l_ac2_max)
#     ax.plot((freqs[idx_beta])[f_l_ac0_max], d_ac0_max,'r*', label = 'peak b/f accelerate') 
#     ax.plot((freqs[idx_beta])[f_l_ac2_max], d_ac2_max,'g*', label = 'peak during accelerate')
#     ax.legend();ax.set_ylim([-1, 7]);ax.set_xlabel('Hz');
#     ax.set_title(str(np.sum(f_l_ac2_max>f_l_ac0_max))+' out of '+str(N_acl)+' acc  '+'p = '+str(np.round(p_ac,6)))

#     ax_idx = gs[0,1];  ax = fig.add_subplot(ax_idx);
#     N_acl_r = S_l_acl2_r.shape[0]
#     f_l_ac0_r_max = np.zeros((N_acl_r,)).astype('int');
#     f_l_ac2_r_max = np.zeros((N_acl_r,)).astype('int');

#     d_ac0_r_max = np.zeros((N_acl_r,));
#     d_ac2_r_max = np.zeros((N_acl_r,));


#     for i in np.arange(N_acl_r):
#         d_acl0 = S_l_acl0_r[i,:]-S_i_acl0_r[i,:];ax.plot(freqs[idx_beta], d_acl0,'r', lw = 0.5)
#         d_acl2 = S_l_acl2_r[i,:]-S_i_acl2_r[i,:];ax.plot(freqs[idx_beta], d_acl2,'g', lw = 0.5 )
#     #     f_l_ac0_r_max[i] = int(measurements.center_of_mass(d_acl0-np.min(d_acl0))[0])
#     #     f_l_ac2_r_max[i] = int(measurements.center_of_mass(d_acl2-np.min(d_acl2))[0])

#         peaks, _ = signal.find_peaks(d_acl0,height=np.mean(d_acl0))
#         if len(peaks)==1:
#             f_l_ac0_r_max[i] = peaks
#         elif len(peaks)>1:
#             peak_acl0 = peaks[np.argmax(d_acl0[peaks])]
#             f_l_ac0_r_max[i] = peak_acl0
#         idx_f = f_l_ac0_r_max[i]
#         d_ac0_r_max[i] = d_acl0[idx_f]

#         peaks, _ = signal.find_peaks(d_acl2,height=np.mean(d_acl2))
#         if len(peaks)==1:
#             f_l_ac2_r_max[i] = peaks
#         elif len(peaks)>1:
#             peak_acl2 = peaks[np.argmax(d_acl2[peaks])]
#             f_l_ac2_r_max[i] = peak_acl2
#         idx_f = f_l_ac2_r_max[i]
#         d_ac2_r_max[i] = d_acl2[idx_f]
#     t_dc,   p_ac_r = stats.ttest_rel(f_l_ac0_r_max,  f_l_ac2_r_max)
#     ax.plot((freqs[idx_beta])[f_l_ac0_r_max], d_ac0_r_max,'r*', label = 'peak b/f random') 
#     ax.plot((freqs[idx_beta])[f_l_ac2_r_max], d_ac2_r_max,'g*', label = 'peak during random')
#     ax.legend();ax.set_ylim([-1, 7]);ax.set_xlabel('Hz');
#     ax.set_title(str(np.sum(f_l_ac2_r_max>f_l_ac0_r_max))+' out of '+str(N_acl_r)+' rand  '+'p = '+str(np.round(p_ac_r,6)))

#     ax_idx = gs[0,2];  ax = fig.add_subplot(ax_idx);
#     ax.set_ylabel('d Hz');
#     d_f   = f_beta[f_l_ac2_max]-f_beta[f_l_ac0_max]
#     d_f_r = f_beta[f_l_ac2_r_max]-f_beta[f_l_ac0_r_max]
#     ax.bar(1, np.mean(d_f),   yerr=np.std(d_f), color = 'green')
#     ax.bar(2, np.mean(d_f_r), yerr=np.std(d_f_r));ax.set_ylim([-15,15])
#     # ax.bar(1, np.mean(f_beta[f_l_ac2_max]),   yerr=np.std(f_beta[f_l_ac2_max]), color = 'green')
#     # ax.bar(2, np.mean(f_beta[f_l_ac2_r_max]), yerr=np.std(f_beta[f_l_ac2_r_max]));ax.set_ylim(f_beta[[0,-1]])
#     labels = [item.get_text() for item in ax.get_xticklabels()]
#     labels[2] = 'acceleration'
#     labels[6] = 'random'
#     ax.set_xticklabels(labels);ax.set_title('band drift')
#     fig.suptitle(rfdn+' z score normalization', fontsize=14)

    
    
#     t_dc,   p_ac_d = stats.ttest_ind(d_f, d_f_r, equal_var = False)
    
#     d1 = np.mean(d_f)+np.std(d_f)
#     d2 = np.mean(d_f_r)+np.std(d_f_r)
#     ax.plot([1, 1, 2, 2], [d1+1, d1+1.5, d1+1.5, d1+1],'k', lw=1.5)
#     ax.text(1, d1+2,  'p = '+str(np.round(p_ac_d,6)), fontsize=10)
    
#     print('ac', np.round(p_ac,4), np.round(p_ac_r,4), np.round(p_ac_d,4), 
#           '   turing:',len(idx_ac_all[idx_ac_all==0]),
#           'straight:',len(idx_ac_all[idx_ac_all==1]),
#           'none:',  len(idx_ac_all[idx_ac_all==2]))

#     ax_idx = gs[1,0];  ax = fig.add_subplot(ax_idx);
#     f_l_dc0_max = np.zeros((N_dcl,)).astype('int');
#     f_l_dc1_max = np.zeros((N_dcl,)).astype('int');
#     f_l_dc2_max = np.zeros((N_dcl,)).astype('int');
#     d_dc0_max = np.zeros((N_dcl,));
#     d_dc1_max = np.zeros((N_dcl,));
#     d_dc2_max = np.zeros((N_dcl,));

#     for i in np.arange(N_dcl):
#         d_dcl0 = S_l_dcl0[i,:]-S_i_dcl0[i,:]; ax.plot(freqs[idx_beta], d_dcl0,'r', lw = 0.5)
#         d_dcl2 = S_l_dcl2[i,:]-S_i_dcl2[i,:]; ax.plot(freqs[idx_beta], d_dcl2,'g', lw = 0.5)

#         peaks, _ = signal.find_peaks(d_dcl0,height=np.mean(d_dcl0))
#         if len(peaks)==1:
#             f_l_dc0_max[i] = peaks
#         elif len(peaks)>1:
#             f_l_dc0_max[i] = peaks[np.argmax(d_dcl0[peaks])]
#         idx_f = f_l_dc0_max[i]
#         d_dc0_max[i] = d_dcl0[idx_f]

#         peaks, _ = signal.find_peaks(d_dcl2,height=np.mean(d_dcl2))
#         if len(peaks)==1:
#             f_l_dc2_max[i] = peaks

#         elif len(peaks)>1:
#             f_l_dc2_max[i] = peaks[np.argmax(d_dcl2[peaks])]
#         idx_f = f_l_dc2_max[i]
#         d_dc2_max[i] = d_dcl2[idx_f]

#     t_dc,   p_dc   = stats.ttest_rel(f_l_dc0_max,   f_l_dc2_max)
#     ax.plot((freqs[idx_beta])[f_l_dc0_max], d_dc0_max,'r*', label = 'peak decelerate') 
#     ax.plot((freqs[idx_beta])[f_l_dc2_max], d_dc2_max,'g*', label = 'peak after decelerate')
#     ax.legend();ax.set_ylim([-1, 7]);ax.set_xlabel('Hz');
#     ax.set_title(str(np.sum(f_l_dc2_max<f_l_dc0_max))+' out of '+str(N_dcl)+' dec  '+'p = '+str(np.round(p_dc,6)))

#     ax_idx = gs[1,1];  ax = fig.add_subplot(ax_idx);
#     N_dcl_r = S_l_dcl0_r.shape[0]
#     f_l_dc0_r_max = np.zeros((N_dcl_r,)).astype('int');
#     f_l_dc2_r_max = np.zeros((N_dcl_r,)).astype('int');

#     d_dc0_r_max = np.zeros((N_dcl_r,));
#     d_dc2_r_max = np.zeros((N_dcl_r,));


#     for i in np.arange(N_dcl_r):
#         d_dcl0 = S_l_dcl0_r[i,:]-S_i_dcl0_r[i,:];ax.plot(freqs[idx_beta], d_dcl0,'r', lw = 0.5)
#         d_dcl2 = S_l_dcl2_r[i,:]-S_i_dcl2_r[i,:];ax.plot(freqs[idx_beta], d_dcl2,'g', lw = 0.5 )

#         peaks, _ = signal.find_peaks(d_dcl0,height=np.mean(d_dcl0))
#         if len(peaks)==1:
#             f_l_dc0_r_max[i] = peaks
#         elif len(peaks)>1:
#             peak_dcl0 = peaks[np.argmax(d_dcl0[peaks])]
#             f_l_dc0_r_max[i] = peak_dcl0
#         idx_f = f_l_dc0_r_max[i]
#         d_dc0_r_max[i] = d_dcl0[idx_f]

#         peaks, _ = signal.find_peaks(d_dcl2,height=np.mean(d_dcl2))
#         if len(peaks)==1:
#             f_l_dc2_r_max[i] = peaks
#         elif len(peaks)>1:
#             peak_dcl2 = peaks[np.argmax(d_dcl2[peaks])]
#             f_l_dc2_r_max[i] = peak_dcl2
#         idx_f = f_l_dc2_r_max[i]
#         d_dc2_r_max[i] = d_dcl2[idx_f]
#     t_dc,   p_dc_r = stats.ttest_rel(f_l_dc0_r_max, f_l_dc2_r_max)
#     ax.plot((freqs[idx_beta])[f_l_dc0_r_max], d_dc0_r_max,'r*', label = 'peak random') 
#     ax.plot((freqs[idx_beta])[f_l_dc2_r_max], d_dc2_r_max,'g*', label = 'peak after random')
#     ax.legend();ax.set_ylim([-1, 7]);ax.set_xlabel('Hz');
#     ax.set_title(str(np.sum(f_l_dc0_r_max>f_l_dc2_r_max))+' out of '+str(N_dcl_r)+' rand  '+'p = '+str(np.round(p_dc_r,6)))

#     ax_idx = gs[1,2];  ax = fig.add_subplot(ax_idx);
#     ax.set_ylabel('d Hz');
#     d_f   = f_beta[f_l_dc2_max]-f_beta[f_l_dc0_max]
#     d_f_r = f_beta[f_l_dc2_r_max]-f_beta[f_l_dc0_r_max]

#     ax.bar(1, np.mean(d_f),   yerr=np.std(d_f), color = 'green')
#     ax.bar(2, np.mean(d_f_r), yerr=np.std(d_f_r));ax.set_ylim([-15,15])
#     labels = [item.get_text() for item in ax.get_xticklabels()]
#     labels[2] = 'deceleration'
#     labels[6] = 'random'
#     ax.set_xticklabels(labels);ax.set_title('band drift')
    
    
#     t_dc, p_dc_d = stats.ttest_ind(d_f, d_f_r, equal_var = False)
    
#     d1 = np.mean(d_f)+np.std(d_f)
#     d2 = np.mean(d_f_r)+np.std(d_f_r)
#     ax.plot([1, 1, 2, 2], [d1+1, d1+1.5, d1+1.5, d1+1],'k', lw=1.5)
#     ax.text(1, d1+2,  'p = '+str(np.round(p_dc_d,6)), fontsize=10)

#     print('dc', np.round(p_dc,4), np.round(p_dc_r,4), np.round(p_dc_d,4), 
#           'turing:',  len(idx_dc_all[idx_dc_all==0]),
#           'straight:',len(idx_dc_all[idx_dc_all==1]),
#           'none:',    len(idx_dc_all[idx_dc_all==2]))
#     plt.savefig('pic2/ac_dc_stat_'+rfdn[9:len(rfdn)-1]+'.png')
#     plt.clf()
    
