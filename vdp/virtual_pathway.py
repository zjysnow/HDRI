import numpy as np
from .metric_params import MetricParams
from .utils import create_cycdeg_image, load_spectral_resp, joint_rod_cone_sens, rod_sens, pix_per_degree, fast_conv_fft, pupil_d_unified
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.stats import gmean
from numpy.matlib import trapz
from scipy.interpolate import pchip_interpolate
from .local_adapt import local_adapt

from .mtf import mtf

def build_jndspace_from_S(l, S):
    L = 10**l
    dL = S * np.log(10)
    Y = l
    jnd = integrate.cumtrapz(dL, l)
    return Y, jnd

def create_pn_jnd(metric_par : MetricParams):
    c_l = np.logspace(-5, 5, 2048)
    print(c_l.shape)
    s_A = joint_rod_cone_sens(c_l, metric_par)
    
    s_R = rod_sens(c_l, metric_par) * 10**metric_par.rod_sensitivity
    s_C = 0.5 * np.interp(np.minimum(c_l*2, c_l[-1]), c_l, np.maximum(s_A-s_R, 1e-3))

    Y = np.zeros((2, 2048))
    jnd = np.zeros((2, 2048))
    Y[0], jnd[0, 1:] = build_jndspace_from_S( np.log10(c_l), s_C)
    Y[1], jnd[1, 1:] = build_jndspace_from_S( np.log10(c_l), s_R)

    return Y, jnd

    

def visual_pathway( img, name, metric_par : MetricParams, bb_padvalue ):
    width = img.shape[1] # size(img,2);
    height = img.shape[0] # size(img,1);
    img_sz = np.array([height, width]); # image size
    img_ch = img.shape[2] # size(img,3); % number of color channels

    rho2 = create_cycdeg_image( img_sz*2, metric_par.pix_per_deg); # spatial frequency for each FFT coefficient, for 2x image size

    # [lambda, LMSR_S] = load_spectral_resp( fullfile( metric_par.base_dir, 'data', 'log_cone_smith_pokorny_1975.csv' ) );
    l, LMSR_S = load_spectral_resp("vdp/data/log_cone_smith_pokorny_1975.csv")
    LMSR_S[LMSR_S==0] = np.min(LMSR_S)
    LMSR_S = 10**LMSR_S

    # [~, ROD_S] = load_spectral_resp( fullfile( metric_par.base_dir, 'data', 'cie_scotopic_lum.txt' ) );
    _, ROD_S = load_spectral_resp("vdp/data/cie_scotopic_lum.txt")
    # LMSR_S[:,4] = ROD_S
    LMSR_S = np.append(LMSR_S, ROD_S, 1)

    IMG_E = metric_par.spectral_emission

    # =================================
    # Precompute photoreceptor non-linearity
    # =================================

    # pn = hdrvdp_get_from_cache( 'pn', [metric_par.rod_sensitivity metric_par.csf_sa], @() create_pn_jnd( metric_par ) )

    # pn.jnd{1} = pn.jnd{1} * 10.^metric_par.sensitivity_correction
    # pn.jnd{2} = pn.jnd{2} * 10.^metric_par.sensitivity_correction

    Y, jnd = create_pn_jnd(metric_par)
    jnd = jnd * 10**metric_par.sensitivity_correction
    
    L_O = np.zeros_like(img)

    for k in range(img_ch):
        if metric_par.mtf == "none":
            L_O[:,:,k] = img[:,:,k]
        else:
            pad_value = 0 # metric_par.pad_value[k]
            mtf_filter = mtf(rho2, metric_par)
            L_O[:,:,k] = np.clip(fast_conv_fft(img[:,:,k], mtf_filter, pad_value), 1e-5, 1e10)

    print("L_O minmax: ", L_O.max(), L_O.min())

    if metric_par.do_aod:
        lam = np.array([400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580, 590, 600, 610, 620, 630, 640, 650])
        TL1 = np.array([0.600, 0.510, 0.433, 0.377, 0.327, 0.295, 0.267, 0.233, 0.207, 0.187, 0.167, 0.147, 0.133, 0.120, 0.107, 0.093, 0.080, 0.067, 0.053, 0.040, 0.033, 0.027, 0.020, 0.013, 0.007, 0.000])
        TL2 = np.array([1.000, 0.583, 0.300, 0.116, 0.033, 0.005, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        OD_y = TL1 + TL2
    
        if metric_par.age <= 60:
            OD = TL1 * (1 + 0.02 * (metric_par.age - 32)) + TL2
        else:
            OD = TL1 * (1.56 + 0.0667*(metric_par.age - 60)) + TL2

        trans_filter = 10**(-pchip_interpolate(lam, OD-OD_y, np.clip(l, lam[0], lam[-1])))
        IMG_E = IMG_E * trans_filter.reshape(-1,1) # repmat(trans_filter.T, 1, IMG_E.shape[1])

    
    M_img_lmsr = np.zeros((img_ch, 4))
    for i in range(4):
        for c in range(img_ch):
            M_img_lmsr[c, i] = trapz(LMSR_S[:,i]*IMG_E[:,c]*683.002, l)
    
    M_img_lmsr = M_img_lmsr / np.sum(M_img_lmsr[:,0:2])

    R_LMSR = np.clip(np.matmul(L_O, M_img_lmsr), 1e-8, 1e10)
    print("R_LMSR minmax: ", R_LMSR.max(), R_LMSR.min())
    
    L_adapt = local_adapt(R_LMSR[:,:,0] + R_LMSR[:,:,1], metric_par.pix_per_deg) # L + M = Y

    if metric_par.do_slum:
        L_a = gmean(L_adapt.reshape(-1))
        print("L_a: ", L_a)
        print("L_adapt max:", L_adapt.max())
        area = height * width / metric_par.pix_per_deg**2
        d_ref = pupil_d_unified(L_a, area, 28)
        d_age = pupil_d_unified(L_a, area, metric_par.age)

        lum_reduction = d_age**2 / d_ref**2
        R_LMSR = R_LMSR * lum_reduction

    print("R_LMSR minmax: ", R_LMSR.max(), R_LMSR.min())
    P_LMR = np.zeros_like(R_LMSR)
    for k in [0,1,3]: # ignore S - dose not influence luminance
        if k == 3:
            ph_type = 1 # rod
            ii = 2
        else:
            ph_type = 0 # cone
            ii = k
        P_LMR[:,:,ii] = np.interp(np.log10(np.clip(R_LMSR[:,:,k], 10**Y[ph_type][0], 10**Y[ph_type][-1])), Y[ph_type], jnd[ph_type])
    
    P_C = P_LMR[:,:,0]+P_LMR[:,:,1]
    P_R = P_LMR[:,:,2]
    P = P_C + P_R

    print("P minmax: ", P.max(), P.min())
    bands = metric_par.mult_scale.decompose(P, float(metric_par.pix_per_deg))
    BB = bands.get_band(bands.band_count()-1)
    bands = bands.set_band(bands.band_count()-1, BB-np.mean(BB.reshape(-1)))
    return bands, L_adapt, bb_padvalue, P
    


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    img = np.random.randn(112,112,3)
    metric_par = MetricParams()
    metric_par.pix_per_deg = pix_per_degree(30, [112, 112], 0.5)
    visual_pathway(img, "reference", metric_par, -1)
