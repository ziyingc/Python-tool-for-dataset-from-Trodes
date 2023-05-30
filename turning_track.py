import struct
import numpy as np
from itertools import combinations
from scipy.signal import butter, bessel, filtfilt, freqz, firwin
import scipy.signal as signal
from scipy.stats.stats import pearsonr
from scipy.ndimage.filters import gaussian_filter;
import os
import math
import philters
fps = 30
def angularchange(v_ft_X1, v_ft_Y1, v_ft_X2, v_ft_Y2):
    x1 = math.atan(v_ft_Y1/v_ft_X1)*180/np.pi;
    x2 = math.atan(v_ft_Y2/v_ft_X2)*180/np.pi;
    if v_ft_X1<0: x1 = x1+180
    if v_ft_X2<0: x2 = x2+180
    delta = x2 - x1
    return delta, x1, x2


def max_distance_k(XY, d, dT):
    l = XY.shape[0];
    p_0 = XY[0,:]
    d_max = False
    dis = 0
    for i in np.arange(dT, l):
        p_i = XY[i,:]
        d_ji = np.sqrt(np.matmul((p_0-p_i),(p_0-p_i).T))
        if d_ji > dis:
            dis  = d_ji
    if dis > d:
        d_max = True
    return d_max, dis
        
def max_distance(XY, d, dT):
    
    
    
    l = XY.shape[0];
    dis  = 0
    d_max = False
    for i in np.arange(l-1):
        p_i = XY[i,:]
        i_s = np.max([dT, i]);
        for j in np.arange(i_s+1,l):
            p_j  = XY[j,:]
            d_ji = np.sqrt(np.matmul((p_j-p_i),(p_j-p_i).T))
            if d_ji > dis:
                dis   = d_ji
                dis_j = j
                dis_i = i
            if d_ji > d:
                d_max = True
                break
        if d_max:
            break
    return d_max, dis_i, dis_j, dis


def displacement_k(fdn, T1, T2, d_min, d_max):
    ts_f  = np.load(fdn+'tracking/'+'ts_f.npy');l_t = len(ts_f);
    pos_Y = np.load(fdn+'tracking/'+'pos_Y_f.npy')
    pos_X = np.load(fdn+'tracking/'+'pos_X_f.npy')
    pos_Y = np.convolve(pos_Y, np.ones((6,))/6, mode='same')
    pos_X = np.convolve(pos_X, np.ones((6,))/6, mode='same')
    XY    = np.array([pos_X, pos_Y]).T;
    try:
        idx_clean_f   = np.load(fdn+'tracking/'+'idx_clean_f.npy');
        idx_clean_fft = np.load(fdn+'tracking/'+'idx_clean_fft.npy')
        idx_clean     = np.logical_and(idx_clean_f, idx_clean_fft)
    except:
        idx_clean = np.ones((l_t,))>0; print('not ephys')

    dT = T2-T1
    
    d_T12 = np.empty((0,)).astype('int');
    
    i = 0
    T_step1 = 0
    T_step2 = 0
    while i < l_t-T2-dT:
        d_max_T1, d_i, d_j, dis     = max_distance(XY[i:i+T1,:],    d_min, T_step1);
        if not(d_max_T1):
#             d_max_T2, d_j, dis = max_distance  (XY[i+T1:i+T2,:], d_max, T_step2);
            d_max_T2, dis = max_distance_k(XY[i+T1:i+T2,:], d_max, T_step2);
            if d_max_T2:
                d_T12 = np.append(d_T12, i);
#                 print(np.round(d_max_T2,2), len(d_max_T2), d_i, d_j)
                i = i + T2
                T_step1 = 0;    T_step2 = 0
            else:
                i = i+1
                T_step1 = T1-2; T_step2 = dT-2
        else:
            i = i+d_j
            T_step1 = 0; T_step2 = 0
    index_clean = np.zeros((len(d_T12),))>1
    ii = 0
    for i in d_T12:
        
        if np.sum(idx_clean[i:i+T2+dT]) == T2+dT:
            index_clean[ii] = True
        ii = ii+1
    return d_T12[index_clean]

def displacement(rfdn, epn, T1, T2, d_min, d_max):
    for k in epn:
        fdn = rfdn + k + '/'
        d_T12 = displacement_k(fdn, T1, T2, d_min, d_max)
        np.save(fdn+'tracking/idx_acc_f.npy', d_T12);
        print(k, d_T12.shape)


def displacement_rand(rfdn, epn, fs_out, T1, T2, N):
    for k in epn:
        fdn = rfdn + k + '/'
        ts_f = np.load(fdn+'tracking/'+'ts_f.npy'); l_t = len(ts_f)
        n_rand = int(l_t/fps/60)*N
        try:
            idx_clean_f   = np.load(fdn+'tracking/'+'idx_clean_f.npy');
            idx_clean_fft = np.load(fdn+'tracking/'+'idx_clean_fft.npy')
            idx_clean     = np.logical_and(idx_clean_f, idx_clean_fft)
        except:
            idx_clean = np.ones((l_t,))>0; print('not ephys')
        
        
        idx_t_r = np.arange(l_t-T2)
        idx_t_r = idx_t_r[idx_clean[0:l_t-T2]]
        idx_acc_f   = np.load(fdn+'tracking/'+'idx_acc_f.npy')
        idx_acc_f_r = np.empty((0,))
        l_acc = len(idx_acc_f)
        i = 0
        while i < n_rand:
            t_ref = np.random.choice(idx_t_r, 1)[0]  
            idx_t_spec = np.arange(t_ref,t_ref+T2).astype('int')
            if np.sum(idx_clean[idx_t_spec])==T2:
                idx_acc_f_r = np.append(idx_acc_f_r, t_ref);
                i = i+1
        np.save(fdn+'tracking/idx_acc_f_r.npy', idx_acc_f_r);
        
        
def accelerations(rfdn, epn, T1, T2, N_win, d_min_thr, d_max_thr, spd_m_thr, spd_min_thr):

    total_ac  = 0
    total_ref = 0
    total_l_t = 0
    for k in epn:
        fdn = rfdn + k + '/'
        idx_clean_f   = np.load(fdn+'tracking/'+'idx_clean_f.npy');
        idx_clean_fft = np.load(fdn+'tracking/'+'idx_clean_fft.npy')
        idx_clean     = np.logical_and(idx_clean_f, idx_clean_fft)
        ts_f = np.load(fdn+'tracking/'+'ts_f.npy');
        XY    = np.array([pos_X,pos_Y]).T;
        
        displacement = np.load(fdn+'tracking/displacement.npy').astype('int')

        data_seg = np.zeros((len(spd_f),)); data_seg[displacement] = 2
        
        index, T = philters.read_burst(data_seg, 0.5, 1, idx_clean, 0)
        
        np.save(fdn+'tracking/T_acc_f.npy', T);
        np.save(fdn+'tracking/idx_acc_f.npy', index);
        
        total_ac  = total_ac  + len(index)
        total_l_t = total_l_t + len(spd_f)
        
        idx_pre_acc = np.zeros((len(spd_f),))<-1
        for i in np.arange(len(T)):
            i_acc = index[i];
            idx_pre_acc[i_acc-60:i_acc] = True;
        np.save(fdn+'tracking/idx_pre_acc.npy', idx_pre_acc);

    return total_ac, total_l_t

# def align_acc(v_seg, T_lag_max, T_seg, T_s):
#     n_ac, T = v_seg.shape
#     idx =  = np.arange(0,T);
#     xcor_v = np.zeros((n_ac,n_ac))
#     for i in np.arange(n_ac-1):
#         for j in np.arange(i+1,n_ac):
#             v_ij_cor = np.correlate(v_seg[i, T-T_seg:T-T_s], v_seg[j,T-T_seg:T-T_s],"full")[T_seg-T_s-T_lag_max:T_seg-T_s+T_lag_max+1];
#             xcor_v[i,j] = np.argmax(v_ij_cor)-T_lag_max # minus is i leading, plus is i lagging
#             xcor_v[j,i] = -(np.argmax(v_ij_cor)-T_lag_max)

#     xcor_v_m = np.absolute(np.sum(xcor_v, axis=0)/(n_ac-1))
#     xcor_v_s = np.std(xcor_v, axis=0)
#     xcor_v_max = np.max(np.absolute(xcor_v), axis=0)
    
#     lag_m = xcor_v[xcor_v_m.argmin(),:]
#     lag_s = xcor_v[xcor_v_s.argmin(),:]
#     return lag_m, lag_s

def align_acc(v_seg, T_lag_max, T1, T2):
    n_ac, T = v_seg.shape
    idx  = np.arange(T1,T2);
    xcor_v = np.zeros((n_ac,n_ac))
    for i in np.arange(n_ac-1):
        for j in np.arange(i+1,n_ac):
            v_ij_cor = np.correlate(v_seg[i, idx], v_seg[j, idx],"full")[T2-T1-T_lag_max:T2-T1+T_lag_max+1];
            xcor_v[i,j] = np.argmax(v_ij_cor)-T_lag_max # minus is i leading, plus is i lagging
            xcor_v[j,i] = -(np.argmax(v_ij_cor)-T_lag_max)
    
    xcor_v_m = np.absolute(np.sum(xcor_v, axis=0)/(n_ac-1))
    xcor_v_s = np.std(xcor_v, axis=0)
    xcor_v_max = np.max(np.absolute(xcor_v), axis=0)
    
    lag_m = xcor_v[xcor_v_m.argmin(),:]
    lag_s = xcor_v[xcor_v_s.argmin(),:]
    print(xcor_v_m.min())
    return lag_m, lag_s, xcor_v_m.argmin()


def event_rate(rfdn, rats, epn, fs_out):

    subset    = np.load(rfdn+'tracking/subset_ephy_li.npy');
    slope     = np.load(rfdn+'tracking/slope_epn_ephy'+rats+'.npy')
    intercept = np.load(rfdn+'tracking/intercept_epn_ephy'+rats+'.npy')

    n_k = len(epn)

    n_acc_total = 0
        
    r_ac = np.empty((0,));
    T_k  = np.empty((0,));
    dis_ac_rats  = np.empty((0,));
    idx_acc_f_T = np.load(rfdn+'tracking/idx_acc_f_T.npy');
    kk = 0
    for k in epn:
        fdn = rfdn + k + '/'
        idx_clean_f   = np.load(fdn+'tracking/'+'idx_clean_f.npy');
        idx_clean_fft = np.load(fdn+'tracking/'+'idx_clean_fft.npy')
        idx_clean     = np.logical_and(idx_clean_f, idx_clean_fft)
        l_t = np.sum(idx_clean)/fps/60
        
#         print(np.round(np.sum(idx_clean)/len(idx_clean),2))
#         l_t = len(idx_clean)/fps/60

        ts_f        = np.load(fdn+'tracking/ts_f.npy');
        idx_acc_f   = np.load(fdn+'tracking/idx_acc_f.npy');
        idx_acc_f_k = np.load(fdn+'tracking/idx_acc_f_k.npy');
        dis_ac      = np.load(fdn+'tracking/dis_ac.npy');
        n_acc = len(idx_acc_f)
        if n_acc>0:
            idx_acc_f_T_k = idx_acc_f_T[idx_acc_f_k];
            idx_acc_f     = idx_acc_f[idx_acc_f_T_k]
            dis_ac        = dis_ac[idx_acc_f_T_k]
            n_acc = len(idx_acc_f)
            r_ac = np.append(r_ac, n_acc/l_t)
            
            dis_ac_rats = np.append(dis_ac_rats, dis_ac)
            
#             r_ac = np.append(r_ac, n_acc)
        else:
            r_ac = np.append(r_ac, 0)

        T_k  = np.append(T_k, l_t)
        kk += 1
    np.save(rfdn+'tracking/r_ac'+rats+'.npy', r_ac)
    np.save(rfdn+'tracking/T_k'+rats+'.npy',  T_k)
    np.save(rfdn+'tracking/dis_ac'+rats+'.npy',  dis_ac_rats)

def XY_displacement(rfdn, epn, dmax, dmin, T_seg, t_start, test = True):
    total = 0;
    for k in epn:
        fdn = rfdn + k + '/'
        if test:
            ts_f = np.load(fdn+'tracking/'+'time_frame.npy');
            XY   = np.load(fdn+'tracking/'+'position2D_frame.npy');
        else:
            ts_f = np.load(fdn+'tracking/'+'ts_f.npy');
            pos_Y = np.load(fdn+'tracking/'+'pos_Y_f.npy')
            pos_X = np.load(fdn+'tracking/'+'pos_X_f.npy')
            XY    = np.array([pos_X,pos_Y]).T;
            
        pos_X = XY[:,0];
        pos_Y = XY[:,1];
        T_tn  = np.empty((0,0));
        tn    = np.empty((0,0));
        idx_t = np.ones((len(ts_f),))>0
        
        for TN in T_seg:
            i = t_start
            while i < len(pos_X)-TN:
                dy = np.absolute(pos_Y[i+TN] - pos_Y[i])
                dx = np.absolute(pos_X[i+TN] - pos_X[i])
                if dx >= dmin and dy >= dmin and dx <= dmax and dy <= dmax:
                    if np.max([dy,dx])/np.min([dy,dx])<1.05:
                        if np.sum(idx_t[i:i+TN]) == len(idx_t[i:i+TN]):
                            idx_t[i:i+TN] = False
                            tn  = np.append(tn, i);
                            T_tn  = np.append(T_tn, TN);
                            
                            i += TN
                        else:
                            i += 1
                    else:
                        i += 1
                else:
                    i += 1
        tn    = tn.astype('int')
        total = total + len(tn);
        seq   = np.argsort(tn);   T_tn = T_tn[seq];    tn = tn[seq]
        if test:
            np.save(fdn + 'tracking' + '/tn.npy', tn);
            np.save(fdn + 'tracking' + '/T_tn.npy', T_tn);
        else:
            np.save(fdn + 'tracking' + '/tn_f.npy', tn);
            np.save(fdn + 'tracking' + '/T_tn_f.npy', T_tn);
            print(len(tn), k[10:21])
    print('total:', total)
            
def XY_disp_filter(rfdn, epn, dmax, dmin, T_min, test = True):
    total = 0;
    total_b4 = 0
    for k in epn:
        fdn = rfdn + k + '/'
        if test:
            ts_f = np.load(fdn + 'tracking' + '/time_frame.npy')
            XY   = np.load(fdn + 'tracking' + '/position2D_frame.npy');
            tn   = np.load(fdn + 'tracking' + '/tn.npy');
            T_tn = np.load(fdn + 'tracking' + '/T_tn.npy');
        else:
            ts_f = np.load(fdn + 'tracking' + '/ts_f.npy')
            pos_Y = np.load(fdn+'tracking/'+'pos_Y_f.npy')
            pos_X = np.load(fdn+'tracking/'+'pos_X_f.npy')
            tn   = np.load(fdn + 'tracking' + '/tn_f.npy');
            T_tn = np.load(fdn + 'tracking' + '/T_tn_f.npy');

        n_tn = len(tn); 
        T_tn2 = np.empty((0,));  tn2 = np.empty((0,)); 
        T_tn21 = np.empty((0,)); tn21 = np.empty((0,));
        
        total_b4 = total_b4+n_tn
        if n_tn>0:
            for i in np.arange(n_tn):
                TN = int(T_tn[i])
                tn_i = int(tn[i])
                if TN > T_min:
                    T_seg = np.arange(T_min, TN).astype('int')
                    for t in T_seg:
                        j = 0
                        turned = 0
                        while j <= TN-t:
#                             dy = np.absolute(pos_Y[tn_i+TN-j-t] - pos_Y[tn_i+TN-j])
#                             dx = np.absolute(pos_X[tn_i+TN-j-t] - pos_X[tn_i+TN-j])
                            dy = np.absolute(pos_Y[tn_i+j] - pos_Y[tn_i+j+t])
                            dx = np.absolute(pos_X[tn_i+j] - pos_X[tn_i+j+t])
                            if dx >= dmin and dy >= dmin and dx <= dmax and dy <= dmax:
                                R_xy = np.max([dy,dx])/np.min([dy,dx])
                                if R_xy < 1.01:
                                    tn2    = np.append(tn2, tn_i+j);
                                    tn21   = np.append(tn2, tn_i);
                                    T_tn2  = np.append(T_tn2, t);
                                    T_tn21 = np.append(T_tn21, TN);
                                    turned = 1
                                    total = total+1
                                    break
                            j = j+1
                        if turned == 1:
                            break
                else:
                    tn2    = np.append(tn2, tn_i);
                    T_tn2  = np.append(T_tn2, TN);
                    T_tn21 = np.append(T_tn21, TN);
                    total = total+1

        if test:
            np.save(fdn + 'tracking' + '/tn_2.npy', tn2);
            np.save(fdn + 'tracking' + '/T_tn2.npy', T_tn2);  
            np.save(fdn + 'tracking' + '/T_tn21.npy', T_tn21);
        else:
#             print(tn2.shape, T_tn2.shape, T_tn21.shape)
            np.save(fdn + 'tracking' + '/tn_2_f.npy', tn2); 
            np.save(fdn + 'tracking' + '/T_tn2_f.npy', T_tn2);
            np.save(fdn + 'tracking' + '/T_tn21_f.npy', T_tn21);
    print('total shortened turns:', total, 'out of ',total_b4)
    
    
    
def XY_turn_filter(rfdn, epn, t_pre, test = True):

    total_l = 0;total_r = 0
    for k in epn:
        fdn = rfdn + k + '/'
        if test:
            ts_f = np.load(fdn+'tracking'+'/time_frame.npy');XY = np.load(fdn+'tracking/'+'position2D_frame.npy');
            tn   = np.load(fdn+'tracking'+'/tn_2.npy');    
            T_tn = np.load(fdn+'tracking/'+'T_tn2.npy');  T_tn21 = np.load(fdn+'tracking/'+'T_tn21.npy');
        else:
            idx_clean_f   = np.load(fdn+'tracking/'+'idx_clean_f.npy');
            idx_clean_fft = np.load(fdn+'tracking/'+'idx_clean_fft.npy')
            idx_clean   = np.logical_and(idx_clean_f, idx_clean_fft)
            
            ts_f      = np.load(fdn+'tracking'+'/ts_f.npy');

            v_ft_Y = np.load(fdn+'tracking/' + 'spd_Y_f.npy'); pos_Y = np.load(fdn+'tracking/'+'pos_Y_f.npy')
            v_ft_X = np.load(fdn+'tracking/' + 'spd_X_f.npy'); pos_X = np.load(fdn+'tracking/'+'pos_X_f.npy')

            tn    = np.load(fdn+'tracking'+'/tn_2_f.npy');
            T_tn  = np.load(fdn+'tracking/'+'T_tn2_f.npy');
            T_tn21 = np.load(fdn+'tracking/'+'T_tn21_f.npy');
            
            tn_idx_r = np.empty((0,));
            l_t = len(ts_f);
            idx_t_r = (np.arange(l_t))[idx_clean]; 
            idx_t_r = idx_t_r[idx_t_r<l_t-t_pre]

        n_tn = len(tn)
        idx_tn = np.empty((0,));     d_tn = np.empty((0,));   d_pre_tn = np.empty((0,));
        T_tn_idx = np.empty((0,));   tn_idx = np.empty((0,));
        T_tn_idx21 = np.empty((0,));
        if n_tn>0:
            print(tn.shape, T_tn.shape, T_tn21.shape)
            for i in np.arange(n_tn):
                TN   = T_tn[i].astype('int')
                TN21 = T_tn21[i].astype('int')
                tn_i = tn[i].astype('int')
                dy_pre = np.absolute(np.diff(pos_Y[(tn_i-t_pre):tn_i]))
                dx_pre = np.absolute(np.diff(pos_X[(tn_i-t_pre):tn_i]))
                d_pre = np.sum(np.sqrt (np.square(dy_pre)+np.square(dx_pre)))

                v_x1 = np.mean(v_ft_X[(tn_i-5):(tn_i+0)]);   v_x2 = np.mean(v_ft_X[tn_i+TN:tn_i+TN+5])
                v_y1 = np.mean(v_ft_Y[(tn_i-5):(tn_i+0)]);   v_y2 = np.mean(v_ft_Y[tn_i+TN:tn_i+TN+5])
                
                delta, x1, x2 = angularchange(v_x1, v_y1, v_x2, v_y2)
                
                dd = 60
                if delta > (90-dd) and delta< (90+dd):
                    idx_tn = np.append(idx_tn, 1)
                    T_tn_idx = np.append(T_tn_idx, TN)
                    T_tn_idx21 = np.append(T_tn_idx21, TN21)
                    tn_idx   = np.append(tn_idx, tn_i)
                    d_pre_tn = np.append(d_pre_tn, d_pre);
                    total_l = total_l+1
#                     print(np.round(delta,2), 'left')
                elif delta < (-90+dd) and delta>(-90-dd):
                    idx_tn = np.append(idx_tn, 0);
                    T_tn_idx = np.append(T_tn_idx, TN)
                    T_tn_idx21 = np.append(T_tn_idx21, TN21)
                    tn_idx   = np.append(tn_idx, tn_i)
                    d_pre_tn = np.append(d_pre_tn, d_pre);
                    total_r = total_r+1

                
        if test:
            np.save(fdn + 'tracking' +'/d_pre_tn.npy', d_pre_tn);
            np.save(fdn + 'tracking' +'/idx_tn.npy', idx_tn);
            np.save(fdn + 'tracking' +'/T_tn_idx.npy', T_tn_idx);
            np.save(fdn + 'tracking' +'/T_tn_idx21.npy', T_tn_idx21);
            np.save(fdn + 'tracking' +'/tn_idx.npy', tn_idx);
        else:
            tn_idx_clean = tn_idx> -1;
            total = total_r + total_l
            if total>0:
                i = 0
                while i < total*10:
                    t_ref      = np.random.choice(idx_t_r, 1)[0]
                    idx_t_spec = np.arange(t_ref,t_ref+t_pre).astype('int')
                    if np.sum(idx_clean[idx_t_spec])==t_pre:
                        tn_idx_r = np.append(tn_idx_r, t_ref);
                        i = i+1
                j = 0
                for i in tn_idx:
                    idx_t_spec = np.arange(i,i+T_tn_idx[j]).astype('int')
                    if np.sum(idx_clean[idx_t_spec])<t_pre:
                        tn_idx_clean[j] = False
                    j += 1
            np.save(fdn + 'tracking' +'/tn_idx_clean_f.npy', tn_idx_clean);
            np.save(fdn + 'tracking' +'/d_pre_tn_f.npy',     d_pre_tn);
            np.save(fdn + 'tracking' +'/idx_tn_f.npy',       idx_tn);
            np.save(fdn + 'tracking' +'/T_tn_idx_f.npy',     T_tn_idx);
            np.save(fdn + 'tracking' +'/T_tn_idx21_f.npy',   T_tn_idx21);
            np.save(fdn + 'tracking' +'/tn_idx_f.npy',       tn_idx);
            np.save(fdn + 'tracking' +'/tn_idx_r_f.npy',     tn_idx_r);    
    print('total right:',total_r,'total left:',total_l)