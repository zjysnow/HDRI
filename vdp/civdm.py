import numpy as np
from .metric_params import MetricParams
from .multscale import MultScalLPYR
import cv2
import matplotlib.pyplot as plt
import copy
from skimage.morphology import disk

def psych_func(a, contrast, pf):
    P = 1.0 - np.exp( -(np.abs(a*contrast))**pf)
    return P

def civdm(B_T : MultScalLPYR, L_adapt_test, B_R : MultScalLPYR, L_adapt_reference, CSF, metric_par: MetricParams):
    '''
    This is the adaptation of the dynamic range independent quality metric to
    HDR-VDP-2. It is rebranded as a Contrast Invariant Visibility
    Difference Metric to avoid confusion with the original paper.
    '''
    pf = 10**metric_par.psych_func_slope
    a = (-np.log(0.5))**(1/pf)
    
    P_loss = copy.deepcopy(B_T)
    P_ampl = copy.deepcopy(B_T)
    P_rev = copy.deepcopy(B_T)

    for b in range(B_T.band_count()):
        bsize = B_T.band_size()
        log_La_test_rs = np.clip(np.log10(cv2.resize(L_adapt_test, dsize=(bsize[1], bsize[0]), interpolation=cv2.INTER_CUBIC)), CSF.csf_log_la[0], CSF.csf_log_la[-1])
        # np.interp(np.minimum(c_l*2, c_l[-1]), c_l, np.maximum(s_A-s_R, 1e-3))
        CSF_b_test = np.interp(log_La_test_rs, CSF.csf_log_la, CSF.S[:, b])
    
        print("CSF b test minmax: ", CSF_b_test.max(), CSF_b_test.min())

        log_La_ref_rs = np.clip(np.log10(cv2.resize(L_adapt_reference, dsize=(bsize[1], bsize[0]), interpolation=cv2.INTER_CUBIC)), CSF.csf_log_la[0], CSF.csf_log_la[-1])
        CSF_b_ref = np.interp(log_La_ref_rs, CSF.csf_log_la, CSF.S[:,b])

        for o in range(B_T.orient_count(b)):
            if b == B_T.band_count()-1 or B_T.band_freqs[b] <= 2:
                print("Skip base-band and frequencies <= 2cpd", b)
                test = B_T.get_band(b)
                P_loss = P_loss.set_band(b, np.zeros(test.shape))
                P_ampl = P_ampl.set_band(b, np.zeros(test.shape))
                P_rev = P_rev.set_band(b, np.zeros(test.shape))
                continue

            test = B_T.get_band(b)
            ref = B_R.get_band(b)

            # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, [3*2**(b+1)+1, 3*2**(b+1)+1]) # not same with matlab
            test_dil = cv2.dilate(np.abs(test), disk(3*2**b))
            ref_dil = cv2.dilate(np.abs(ref), disk(3*2**b))

            epsilon = 1e-8
            P_thr = 0.5

            P_t_v = psych_func(a, test * CSF_b_test, pf)
            P_t_v[P_t_v < P_thr] = 0
            # P_t_v = np.ascontiguousarray(P_t_v)
            P_t_iv = 1-psych_func(a, test_dil*CSF_b_test, pf)
            P_t_iv[P_t_iv < P_thr] = 0
            # P_t_iv = np.ascontiguousarray(P_t_iv)

            P_r_v = psych_func(a, ref * CSF_b_ref, pf)
            P_r_v[P_r_v < P_thr] = 0
            # P_r_v = np.ascontiguousarray(P_r_v)
            P_r_iv = 1-psych_func(a, ref_dil * CSF_b_ref, pf)
            P_r_iv[P_r_iv < P_thr] = 0
            # P_r_iv = np.ascontiguousarray(P_r_iv)

            P_loss = P_loss.set_band(b, np.log(1 - P_r_v * P_t_iv + epsilon))
            P_ampl = P_ampl.set_band(b, np.log(1 - P_r_iv * P_t_v + epsilon))

            pp_rev = np.zeros(test.shape)
            P_rev = P_rev.set_band(b, np.log(1-pp_rev))


    loss = 1 - np.exp(-np.abs(P_loss.reconstruct()))
    ampl = 1 - np.exp(-np.abs(P_ampl.reconstruct()))
    rev  = 1 - np.exp(-np.abs(P_rev.reconstruct()))
    print(loss.max(), loss.min())
    print(ampl.max(), ampl.min())
    print(rev.max(), rev.min())
    return loss, ampl, rev


