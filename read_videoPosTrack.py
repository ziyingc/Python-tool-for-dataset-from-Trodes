import struct
import numpy as np
from itertools import combinations
from scipy.signal import butter, bessel, filtfilt, freqz, firwin
import scipy.signal as signal
from scipy.stats.stats import pearsonr
import os

import math

def dotproduct(v1, v2):
      return sum((a*b) for a, b in zip(v1, v2))

def length(v):
      return math.sqrt(dotproduct(v, v))

def angle(v1, v2):
      return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))

def angularchange(v_ft_X1, v_ft_Y1, v_ft_X2, v_ft_Y2):
    x1 = math.atan(v_ft_Y1/v_ft_X1)*180/np.pi;
    x2 = math.atan(v_ft_Y2/v_ft_X2)*180/np.pi;
    if v_ft_X1<0:
        x1 = x1+180
    if v_ft_X2<0:
        x2 = x2+180
    delta = x2 - x1
    return delta, x1, x2
def read_VPT(fn):
    
    n_packets = 500000
    timestamps = [];x1 = [];y1 = [];x2 = [];y2 = []
    ii = 0
    with open(fn, 'rb') as fileobj:
        instr = fileobj.readline()
        n_max_header_lines = 50
        hh = 0
        while (instr != b'<End settings>\n') :
            hh +=1
            instr = fileobj.readline()
            if hh > n_max_header_lines:
                print('End of header not found! Aborting...')
                break
        for packet in iter(lambda: fileobj.read(12), ''):
            if packet:
                ts_ = struct.unpack('<L', packet[0:4])[0]
                x1_ = struct.unpack('<H', packet[4:6])[0]
                y1_ = struct.unpack('<H', packet[6:8])[0]
                x2_ = struct.unpack('<H', packet[8:10])[0]
                y2_ = struct.unpack('<H', packet[10:12])[0]
                timestamps.append(ts_)
                x1.append(x1_)
                y1.append(y1_)
                x2.append(x2_)
                y2.append(y2_)
            else:
                break
            if ii >= n_packets:
                print('Stopped before reaching end of file')
                break
            
    return timestamps, x1, y1, x2, y2

def read_epochs_turning(position, order, loc_range, loc_diff_range):
    l = len(position)
    all_idx = np.arange(l)
    indexes_loc = all_idx[np.logical_and(position>=loc_range[0],position<=loc_range[1])]
    indexes_1 = np.array(signal.argrelextrema(position, np.greater, order = order))
    indexes_2 = np.array(signal.argrelextrema(np.max(position)-position, np.greater, order = order))
    indexes = np.unique(np.concatenate((indexes_1[0],indexes_2[0]),0))
    indexes = np.intersect1d(indexes_loc,indexes)
    indexes_loc_diff = np.zeros([1,len(indexes)])
    indexes_loc_diff = indexes_loc_diff>0
    indexes_loc_diff = indexes_loc_diff.squeeze()
    j = 0
    for i in indexes:
        i1 = i+order
        i2 = i-order
        if i1 < l:
            d1 = np.absolute(position[i1]-position[i])
        else:
            d1 = np.absolute(position[l-1]-position[i])
            
        if i2 >= 0:
            d2 = np.absolute(position[i2]-position[i])
        else:
            d2 = np.absolute(position[0]-position[i])
        if d1>=loc_diff_range[0] or d2>=loc_diff_range[0]:
            indexes_loc_diff[j] = True
        j = j+1
                    
    indexes = indexes[indexes_loc_diff]
    return indexes


def read_epochs_trans(speed, order1, thr_peak, order):
    
    indexes = np.array(signal.argrelextrema(speed, np.greater, order = order1))[0]
    j = 0
    idx_1 = indexes>0
    for i in np.arange(len(indexes)-1):
        t1 = indexes[i]
        t2 = indexes[i+1]
        if t2-t1 < order:
            if speed[t1]<speed[t2]:
                idx_1[i] = False
            else:
                idx_1[i+1] = False
    indexes = indexes[idx_1]
    spd_idx = speed[indexes]
    idx_2 = spd_idx>=thr_peak
    indexes = indexes[idx_2]
    
    return indexes


def read_epochs_peaks(data, order1, thr_peak, order2, drop, order3):

    indexes = np.array(signal.argrelextrema(data, np.greater, order = order1))[0]
    spd_idx = data[indexes]
    idx_1 = spd_idx>=thr_peak
    indexes = indexes[idx_1]
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
    ################################################
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
        spd_idx1[j] = np.min(data[t1:i])<(peaks[j]*drop) and np.min(data[(i+1):t2])<(peaks[j]*drop)
        j = j+1
    
    indexes = indexes[spd_idx1]
    return indexes


# def read_accel(data, ):
    
#     return indexes