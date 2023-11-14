from .multscale import MultScalLPYR
from .csf import CSF
from .mtf import MTF
from .utils import load_spectral_resp, create_cycdeg_image, fast_conv_fft, local_adapt, pupil_d_unified

import numpy as np
from numpy.matlib import trapz
from scipy import integrate
from scipy.stats import gmean
from scipy.interpolate import pchip_interpolate
from skimage.morphology import disk

import copy
import cv2
import os

class VDP():
    '''
    Visually significant Differences between an image Pair

    Contrast Invariant Visual Difference Metric
        red - contrast reversal (removed)
        green - contrast loss
        blue - contrast amplification
    Refer to paper 
        Dynamic range independent image quality assessment
    '''
    def __init__(self, task = "civdm", ppd = 36, age = 24):
        self.csf = CSF(age=age)
        self.mtf = MTF(age=age)

        if task == "civdm": 
            self.mult_scale = MultScalLPYR()
        else:
            NotImplemented

        self.age = age
        self.do_aod = True
        self.do_slum = True

        self.pix_per_deg = ppd
        self.sensitivity_correction = 0.2501039518897
        self.rod_sensitivity = 0
        self.psych_func_slope = 0.34

        self.data_path = os.path.join(os.path.dirname(__file__), "data")

    def create_pn_jnd(self):
        def build_jndspace_from_S(l, S):
            # L = 10**l
            dL = S * np.log(10)
            Y = l
            jnd = integrate.cumtrapz(dL, l)
            return Y, jnd
        
        c_l = np.logspace(-5, 5, 2048)
        s_A = self.csf.joint_rod_cone_sens(c_l)
        
        s_R = self.csf.rod_sens(c_l) * 10**self.rod_sensitivity
        s_C = 0.5 * np.interp(np.minimum(c_l*2, c_l[-1]), c_l, np.maximum(s_A-s_R, 1e-3))

        Y = np.zeros((2, 2048))
        jnd = np.zeros((2, 2048))
        Y[0], jnd[0, 1:] = build_jndspace_from_S( np.log10(c_l), s_C)
        Y[1], jnd[1, 1:] = build_jndspace_from_S( np.log10(c_l), s_R)

        return Y, jnd

    def compare(self, test, reference):
        img_channels = test.shape[2]

        _, IMG_E = load_spectral_resp(os.path.join(self.data_path, "emission_spectra_led-lcd-srgb.csv"))
        if img_channels == 1 and IMG_E.shape[1] > 1:
            IMG_E = np.sum(IMG_E, 1)

        if img_channels != IMG_E.shape[1]:
            NotImplemented
        
        self.spectral_emission = IMG_E

        # should translate color space
        # for example all to P3

        B_R, L_adapt_reference, P_ref = self.visual_pathway(reference)
        B_T, L_adapt_test, P_test = self.visual_pathway(test)

        band_freq= B_T.get_freqs()
        # precompute CSF

        # csf_la = np.logspace(-5,5,256)
        # csf_log_la = np.log10(csf_la)
        self.S = np.zeros((256, B_T.band_count()))

        for b in range(B_T.band_count()):
            self.S[:,b] = self.csf.ncsf(band_freq[b])

        res = self.civdm(B_T, L_adapt_test, B_R, L_adapt_reference)

        return res

    def visual_pathway(self, img):
        width = img.shape[1] # size(img,2);
        height = img.shape[0] # size(img,1);
        img_sz = np.array([height, width]); # image size
        img_ch = img.shape[2] # size(img,3); % number of color channels

        rho2 = create_cycdeg_image( img_sz*2, self.pix_per_deg); # spatial frequency for each FFT coefficient, for 2x image size

        # [lambda, LMSR_S] = load_spectral_resp( fullfile( metric_par.base_dir, 'data', 'log_cone_smith_pokorny_1975.csv' ) );
        l, LMSR_S = load_spectral_resp(os.path.join(self.data_path, "log_cone_smith_pokorny_1975.csv"))
        LMSR_S[LMSR_S==0] = np.min(LMSR_S)
        LMSR_S = 10**LMSR_S

        # [~, ROD_S] = load_spectral_resp( fullfile( metric_par.base_dir, 'data', 'cie_scotopic_lum.txt' ) );
        _, ROD_S = load_spectral_resp(os.path.join(self.data_path, "cie_scotopic_lum.txt"))
        # LMSR_S[:,4] = ROD_S
        LMSR_S = np.append(LMSR_S, ROD_S, 1)

        IMG_E = self.spectral_emission # metric_par.spectral_emission

        # =================================
        # Precompute photoreceptor non-linearity
        # =================================

        Y, jnd = self.create_pn_jnd()
        jnd = jnd * 10**self.sensitivity_correction
        
        L_O = np.zeros_like(img)
        for k in range(img_ch):
            # if metric_par.mtf == "none":
            #     L_O[:,:,k] = img[:,:,k]
            # else:
            pad_value = 0 # metric_par.pad_value[k]
            mtf_filter = self.mtf(rho2)
            L_O[:,:,k] = np.clip(fast_conv_fft(img[:,:,k], mtf_filter, pad_value), 1e-5, 1e10)

        if self.do_aod:
            lam = np.array([400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580, 590, 600, 610, 620, 630, 640, 650])
            TL1 = np.array([0.600, 0.510, 0.433, 0.377, 0.327, 0.295, 0.267, 0.233, 0.207, 0.187, 0.167, 0.147, 0.133, 0.120, 0.107, 0.093, 0.080, 0.067, 0.053, 0.040, 0.033, 0.027, 0.020, 0.013, 0.007, 0.000])
            TL2 = np.array([1.000, 0.583, 0.300, 0.116, 0.033, 0.005, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

            OD_y = TL1 + TL2
        
            if self.age <= 60:
                OD = TL1 * (1 + 0.02 * (self.age - 32)) + TL2
            else:
                OD = TL1 * (1.56 + 0.0667*(self.age - 60)) + TL2

            trans_filter = 10**(-pchip_interpolate(lam, OD-OD_y, np.clip(l, lam[0], lam[-1])))
            IMG_E = IMG_E * trans_filter.reshape(-1,1) # repmat(trans_filter.T, 1, IMG_E.shape[1])
       
        M_img_lmsr = np.zeros((img_ch, 4))
        for i in range(4):
            for c in range(img_ch):
                M_img_lmsr[c, i] = trapz(LMSR_S[:,i]*IMG_E[:,c]*683.002, l)
        
        M_img_lmsr = M_img_lmsr / np.sum(M_img_lmsr[:,0:2])

        R_LMSR = np.clip(np.matmul(L_O, M_img_lmsr), 1e-8, 1e10)
  
        L_adapt = local_adapt(R_LMSR[:,:,0] + R_LMSR[:,:,1], self.pix_per_deg) # L + M = Y

        if self.do_slum:
            L_a = gmean(L_adapt.reshape(-1))
            area = height * width / self.pix_per_deg**2
            d_ref = pupil_d_unified(L_a, area, 28)
            d_age = pupil_d_unified(L_a, area, self.age)

            lum_reduction = d_age**2 / d_ref**2
            R_LMSR = R_LMSR * lum_reduction

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

        bands = self.mult_scale.decompose(P, float(self.pix_per_deg))
        BB = bands.get_band(bands.band_count()-1)
        bands = bands.set_band(bands.band_count()-1, BB-np.mean(BB.reshape(-1)))
        return bands, L_adapt, P

    def civdm(self, B_T : MultScalLPYR, L_adapt_test, B_R : MultScalLPYR, L_adapt_reference):
        '''
        This is the adaptation of the dynamic range independent quality metric to
        HDR-VDP-2. It is rebranded as a Contrast Invariant Visibility
        Difference Metric to avoid confusion with the original paper.
        '''
        def psych_func(a, contrast, pf):
            P = 1.0 - np.exp( -(np.abs(a*contrast))**pf)
            return P

        pf = 10**self.psych_func_slope
        a = (-np.log(0.5))**(1/pf)
        
        P_loss = copy.deepcopy(B_T)
        P_ampl = copy.deepcopy(B_T)
        P_rev = copy.deepcopy(B_T)

        for b in range(B_T.band_count()):
            bsize = B_T.band_size()
            log_La_test_rs = np.clip(np.log10(cv2.resize(L_adapt_test, dsize=(bsize[1], bsize[0]), interpolation=cv2.INTER_CUBIC)), self.csf.csf_log_la[0], self.csf.csf_log_la[-1])
            # np.interp(np.minimum(c_l*2, c_l[-1]), c_l, np.maximum(s_A-s_R, 1e-3))
            CSF_b_test = np.interp(log_La_test_rs, self.csf.csf_log_la, self.S[:, b])

            log_La_ref_rs = np.clip(np.log10(cv2.resize(L_adapt_reference, dsize=(bsize[1], bsize[0]), interpolation=cv2.INTER_CUBIC)), self.csf.csf_log_la[0], self.csf.csf_log_la[-1])
            CSF_b_ref = np.interp(log_La_ref_rs, self.csf.csf_log_la, self.S[:,b])

            for _ in range(B_T.orient_count(b)):
                if b == B_T.band_count()-1 or B_T.band_freqs[b] <= 2:
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
        return loss, ampl, rev

