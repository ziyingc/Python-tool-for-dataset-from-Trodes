import os
import warnings; warnings.filterwarnings("ignore")
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm # colormap module
import matplotlib.mlab as mlab 
from matplotlib.gridspec import GridSpec
import struct
import numpy as np
import nelpy as nel
import scipy
import scipy.signal as signal
from scipy.ndimage.filters import gaussian_filter
import jagular as jag
from sklearn import linear_model
from scipy.signal import butter, lfilter

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def rec2raw(rfdn, epn, subset):
    for fn in epn:
        fdn = rfdn + fn + '/'
        if not os.path.exists(fdn+'Raw'):
            os.makedirs(fdn+'Raw')
        jfm = jag.io.JagularFileMap(fdn+ fn +'.rec')
        jag.utils.extract_channels(jfm=jfm, ch_out_prefix=fdn+'Raw/', ts_out = fdn+'timestamps.raw', subset=subset)

def raw2npy_timestamp(rfdn, epn, subset, fs, fs_out):

    for k in epn:
        fdn = rfdn + k + '/'
        if not os.path.exists(fdn+'Raw'):
            os.makedirs(fdn+'Raw')
        jfm = jag.io.JagularFileMap(fdn+ k +'.rec')
        jag.utils.extract_channels(jfm=jfm, ch_out_prefix=fdn+'Raw/', ts_out = fdn+'timestamps.raw', subset=subset)
        
        ts = np.fromfile(fdn + 'timestamps.raw', dtype=np.uint32)
        if not os.path.exists(fdn+'NPY_'+str(fs_out)):
            os.makedirs(fdn+'NPY_'+str(fs_out))
        i = subset[0]
        chn   = str(i//10)+str(i%10)
        fn    = 'Raw/ch.'+chn+'.raw'
        x     = np.fromfile(fdn+fn, dtype=np.int16)
        sig   = nel.AnalogSignalArray(data=x, timestamps=ts/fs, fs=fs)
        lfp_f = sig.downsample(fs_out=fs_out)

        t = lfp_f.time
        np.save(fdn+'NPY_'+str(fs_out)+'/ts_ds.npy', t)
        np.save(fdn+'NPY_'+str(fs_out)+'/fs_out.npy', fs_out)
        
        
def raw2npy(rfdn, epn, subset, fs, fl, fh, fs_out, check):
    f_band_check = np.array([fl,fh])
    for k in epn:
        fdn = rfdn + k + '/'
        if not(check):
            ts = np.fromfile(fdn + 'timestamps.raw', dtype=np.uint32)
            if not os.path.exists(fdn+'NPY_'+str(fs_out)):
                os.makedirs(fdn+'NPY_'+str(fs_out))
            ls = len(ts)
            for i in subset:
                chn   = str(i//10)+str(i%10)
                fn    = 'Raw/ch.'+chn+'.raw'
                x     = np.fromfile(fdn+fn, dtype=np.int16)
                sig   = nel.AnalogSignalArray(data=x, timestamps=ts/fs, fs=fs)
                lfp_f = nel.filtering.sosfiltfilt(sig, fl=fl, fh=fh)
                lfp_f = lfp_f.downsample(fs_out=fs_out)
                x_f   = lfp_f.ydata[0]
                np.save(fdn+'NPY_'+str(fs_out)+'/ch.'+chn+'.npy', x_f);

            t = lfp_f.time
            np.save(fdn+'NPY_'+str(fs_out)+'/ts_ds.npy', t)
            np.save(fdn+'NPY_'+str(fs_out)+'/fs_out.npy', fs_out)
            np.save(fdn+'NPY_'+str(fs_out)+'/filter.npy', np.array([fl,fh]))
        else:
            try:
                f_band = np.load(fdn+'NPY_'+str(fs_out)+'/filter.npy')
                if (f_band_check[0] != f_band[0]) and (f_band_check[1] != f_band[1]):
                    print(k[10:21],'wrong filter')
                else:
                    xxxx = 0
                    for i in subset:
                        chn = str(i//10)+str(i%10)
                        try:
                            x_f = np.load(fdn+'NPY_'+str(fs_out)+'/ch.'+chn+'.npy');
                        except:
                            xxxx = 1
                    if xxxx == 1:
                        print(k[10:21],'lack channels')  
            except:
                print(k[10:21],'no filters')

def rm_artifact(rfdn, epn, fs_out, subset, N_std):
    for k in epn:
        fdn = rfdn + k + '/'
        ts_ds = np.load(fdn+'NPY_'+str(fs_out)+'/ts_ds.npy');
        ###############remove large value abnormaly from recording##################
        idx_clean = ts_ds>-1; l_t = len(ts_ds); idx_t = np.arange(l_t)
        max_x = 0
        for jj in np.arange(2):
            for i in subset:   
                chn = str(i//10)+str(i%10); x = np.load(fdn+'NPY_'+str(fs_out)+'/ch.'+chn+'.npy');
                if np.max(x)>max_x:
                    max_x = np.max(x)
                sigma_x = np.std(x[idx_clean]);mu_x = np.mean(x[idx_clean])
                x_abs = np.absolute(x-mu_x)
                x_thr = np.min([6000, sigma_x*N_std])
                idx_arf = idx_t[x_abs>x_thr]
                for ii in idx_arf:
                    if idx_clean[ii]:
                        i_st = ii
                        while x_abs[i_st]>sigma_x/2 and i_st>0:
                            idx_clean[i_st] = False
                            i_st = i_st-1
                        i_st = ii
                        while x_abs[i_st]>sigma_x/2 and i_st<l_t-1:
                            idx_clean[i_st] = False
                            i_st = i_st+1
        np.save(fdn + 'NPY_'+str(fs_out)+'/idx_clean.npy', idx_clean)


        