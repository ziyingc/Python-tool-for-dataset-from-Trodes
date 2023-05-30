import struct
import numpy as np
from itertools import combinations
import scipy
import scipy.signal as signal
from scipy.stats.stats import pearsonr
from sklearn import linear_model
#import pycwt as wavelet
import os

def cwt(rfdn, epn, fs_out, freqs, subset, mother):
    dt = 1/fs_out
    for k in epn:
        fdn = rfdn + k + '/'
        if not os.path.exists(fdn+'CWT_'+str(fs_out)):
            os.makedirs(fdn+'CWT_'+str(fs_out))
        for i in subset:   
            chn = str(i//10)+str(i%10);   
            x = np.load(fdn+'NPY_'+str(fs_out)+'/ch.'+chn+'.npy');
            wave, scales, f, coi, fft, fftfreqs = wavelet.cwt(x, dt, wavelet = mother, freqs = freqs)
            Sxx = ((np.abs(wave)) ** 2)
            np.save(fdn + 'CWT_'+str(fs_out)+'/ch.'+chn+'.npy', Sxx)

def rm_cwt_artifact(rfdn, epn, fs_out, freqs, subset, N_std, fl, fh):
    idx_f_thr = np.logical_and(freqs>=fl, freqs<=fh)
    for k in epn:
        fdn = rfdn + k + '/'
        idx_clean = np.load(fdn + 'NPY_'+str(fs_out)+'/idx_clean.npy')
        l_t       = len(idx_clean)
        idx_t     = np.arange(l_t)
        idx_clean_cwt = idx_t>-1
        i = subset[0]; 
        for i in subset:
            chn = str(i//10)+str(i%10);   
            Sxx = np.load(fdn + 'CWT_'+str(fs_out)+'/ch.'+chn+'.npy');
            Sxx = Sxx[idx_f_thr,:];
            Sx_f_l  = np.mean(Sxx, axis=0);
            
            
            Sx_f_clean = Sx_f_l[idx_clean]
            
            Sx_f_l_ref = Sx_f_clean[Sx_f_clean.argsort()[0:int(l_t/2)]];
            
            Sx_f_md  = np.median(Sx_f_l_ref)
            Sx_f_m   = np.mean(Sx_f_l_ref);  
            Sx_f_std = np.std(Sx_f_l_ref);
            
            thr_l = Sx_f_md + Sx_f_std*N_std
            idx_arf = idx_t[Sx_f_l>thr_l]
            for ii in idx_arf:
                if idx_clean_cwt[ii]:
                    i_st = ii
                    while Sx_f_l[i_st]>Sx_f_md/2 + 0*Sx_f_std and i_st>=0:
                        idx_clean_cwt[i_st] = False
                        i_st = i_st-1 
                    i_st = ii
                    while Sx_f_l[i_st]>Sx_f_md/2 + 0*Sx_f_std and i_st<l_t-1:
                        idx_clean_cwt[i_st] = False
                        i_st = i_st+1
        np.save(fdn+'CWT_' +str(fs_out)+'/idx_clean.npy', idx_clean_cwt)
        print(np.round(np.sum(idx_clean_cwt)/len(idx_clean_cwt),3),
              np.round(np.sum(idx_clean)/len(idx_clean),3), k)

        
        
        
def cwt_frame(rfdn, epn, fs_out, freqs, subset, N_h):
    window = np.hanning(N_h*2);
    for k in epn:
        fdn = rfdn + k + '/'
        idx_clean = np.load(fdn+'CWT_' +str(fs_out)+'/idx_clean.npy')
        
        idx    = np.load(fdn+'tracking/'+'/idx_ds2f.npy');
        idx_vb = np.load(fdn+'tracking/'+'/idx_vb.npy');
        idx    = idx[idx_vb]
        
        idx_t = np.arange(len(idx))
        idx_clean_f_cwt = idx_t>-1
        jj = 0
        for j in idx:
            idx_clean_f_cwt[jj] = np.sum(idx_clean[j-N_h:j+N_h])==2*N_h;
            jj = jj+1;
        np.save(fdn+'tracking/'+'idx_clean_cwt.npy', idx_clean_f_cwt)
        
        l_f = len(freqs)
        l_t = len(idx)
        idx_id = np.arange(l_t)
        N =int(len(window)/2)
        Sxx_f = np.zeros((l_f, l_t))
        for i in subset:
            chn   = str(i//10)+str(i%10)
            Sxx = np.load(fdn + 'CWT_'+str(fs_out)+'/ch.'+chn+'.npy')
            Sxx_f = np.zeros((l_f, l_t))
            for j in idx_id:
                jj = idx[j]
                Sxx_f[:,j] = np.sum(Sxx[:,jj-N:jj+N] * window[None,:],axis=1)/np.sum(window)
            np.save(fdn + 'CWT_'+str(fs_out)+'/ch.'+chn+'.fps.npy', Sxx_f.T)
        
def fft_frame(rfdn, epn, subset, fs_out, N_h):
    window = np.hanning(N_h*2);
    for k in epn:
        fdn = rfdn + k + '/'
        if not os.path.exists(fdn+'FFT_'+str(fs_out)):
            os.makedirs(fdn+'FFT_'+str(fs_out))
        idx    = np.load(fdn+'tracking/'+'/idx_ds2f.npy');
        idx_vb = np.load(fdn+'tracking/'+'/idx_vb.npy');
        idx = idx[idx_vb]
        for i in subset:   
            chn   = str(i//10)+str(i%10);
            x_f = np.load(fdn+'NPY_'+str(fs_out)+'/ch.'+chn+'.npy')
            X_f = np.zeros((len(idx),N_h*2));
            r = 0
            for j in idx:
                X_f[r,:] = x_f[j-N_h:j+N_h];
                r = r+1
            X_f = X_f * window[None,:];
            Y_f = scipy.fftpack.fft(X_f);
            Y_f = 2.0/(N_h*2) * np.abs(Y_f[:,:(N_h*2)//2])
            np.save(fdn + 'FFT_'+str(fs_out)+'/ch.'+chn+'.fps.npy', Y_f)
            
def rm_fft_artifact(rfdn, epn, fs_out, freqs, subset, T_N_std, N_std, N_f, fl, fh, mode):
    idx_f_thr = np.logical_and(freqs>=fl, freqs<=fh)
    for k in epn:
        fdn = rfdn + k + '/'
        
        idx_vb = np.load(fdn+'tracking/'+'idx_vb.npy')
        idx    = np.load(fdn+'tracking/'+'idx_ds2f.npy')
        #################################
        idx_clean   = np.load(fdn+'NPY_'+str(fs_out)+'/idx_clean.npy')
        idx = idx[idx_vb]
        idx = idx.astype('int');
        jj=0
        idx_clean_f = idx>-1
        for j in idx:
            idx_clean_f[jj] = np.sum(idx_clean[j-N_f:j+N_f])==2*N_f;
            jj = jj+1;
        np.save(fdn+'tracking/'+'idx_clean_f.npy', idx_clean_f)
        
        idx_clean = idx_clean_f
        
        l_t       = len(idx_clean)
        idx_t     = np.arange(l_t)
        idx_clean_fft = idx_t>-1
        for i in subset:
            chn = str(i//10)+str(i%10);   
            Sxx = np.load(fdn + 'FFT_'+str(fs_out)+'/ch.'+chn+'.fps.npy').T
            Sxx = Sxx[idx_f_thr,:]; Sx_f_l  = np.mean(Sxx, axis=0);
            Sx_f_clean = Sx_f_l[idx_clean]
            Sx_f_l_ref = Sx_f_clean[Sx_f_clean.argsort()[0:int(T_N_std*l_t)]];

            Sx_f_md  = np.median(Sx_f_l_ref); Sx_f_m = np.mean(Sx_f_l_ref);  
            Sx_f_std = np.std(Sx_f_l_ref);

            thr_l = Sx_f_md + Sx_f_std*N_std
            idx_arf = idx_t[Sx_f_l>thr_l]

            for ii in idx_arf:
                if idx_clean_fft[ii]:
                    i_st = ii
                    while Sx_f_l[i_st]>Sx_f_m + 0*Sx_f_std and i_st>=0:
                        idx_clean_fft[i_st] = False
                        i_st = i_st-1 
                    i_st = ii
                    while Sx_f_l[i_st]>Sx_f_m + 0*Sx_f_std and i_st<l_t-1:
                        idx_clean_fft[i_st] = False
                        i_st = i_st+1
        np.save(fdn+'tracking/'+'idx_clean_'+mode+'.npy', idx_clean_fft)
        rto = np.sum(idx_clean_fft)/len(idx_clean_fft)
        if rto <0.75:
            print(np.round(np.sum(idx_clean_fft)/len(idx_clean_fft),3),
                  np.round(np.sum(idx_clean)/len(idx_clean),3), k[10:21])

def flicker_fit_fft(rfdn, epn, fs_out, subset, f, idx_f, mode):
    Psd   = np.zeros((len(epn)*2, len(f)));
    jj = 0
    for k in epn:
        fdn = rfdn + k + '/'
        idx_clean_f   = np.load(fdn+'tracking/'+'idx_clean_f.npy');
        idx_clean_fft = np.load(fdn+'tracking/'+'idx_clean_'+mode+'.npy')
        idx_clean     = np.logical_and(idx_clean_f, idx_clean_fft)
        ii = 0
        for i in subset: 
            chn = str(i//10)+str(i%10);   
            Y_f = np.load(fdn + 'fft_'+str(fs_out)+'/ch.'+chn+'.fps.npy')
            psd = np.mean(Y_f[idx_clean,:],axis=0)
            Psd[jj*2+ii,:] = psd
            ii = ii+1
        jj=jj+1
    Psd_l = Psd[::2];     Psd_i = Psd[1::2]
    f_fit = f[idx_f];
    
    regr = linear_model.LinearRegression()  
    regr.fit(np.log10(f_fit).reshape(-1,1), np.log10(Psd_l[:,idx_f]).T)
    slope_l = (regr.coef_).squeeze();       intercept_l = (regr.intercept_).squeeze()

    regr = linear_model.LinearRegression()  
    regr.fit(np.log10(f_fit).reshape(-1,1), np.log10(Psd_i[:,idx_f]).T)
    slope_i = (regr.coef_).squeeze();       intercept_i = (regr.intercept_).squeeze()

    intercept = np.log10(Psd_l[:,idx_f]) - np.dot(slope_i.reshape(-1,1), np.log10(f_fit).reshape(1,-1));
    intercept_li = np.mean(intercept, axis=1)
    jj = 0
    for k in epn:
        fdn = rfdn + k + '/'
        idx_clean_f  = np.load(fdn+'tracking/'+'idx_clean_f.npy');
        np.save(fdn+'tracking/slope_l.npy',     slope_l[jj]);
        np.save(fdn+'tracking/intercept_l.npy', intercept_l[jj]);
        np.save(fdn+'tracking/slope_i.npy',     slope_i[jj]);
        np.save(fdn+'tracking/intercept_i.npy', intercept_i[jj]);
        jj=jj+1
        
    
    return slope_l, intercept_l, slope_i, intercept_i, intercept_li


