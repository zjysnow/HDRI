import numpy as np
from .multscale import MultScale

class MetricParams:
    def __init__(self):
        self.threshold_p = 0.75
        self.p_obs = 0.8
        self.surround = "none"
        self.ms_decomp = "spyr"
        self.mtf="hdrvdp"
        self.do_new_diff = False
        self.do_sprob_sum = False

        self.pix_per_deg = 0
        self.view_dist = 0.5
        self.do_pixel_threshold = False
        self.age = 24

        self.rgb_display = "led-lcd-srgb"

        self.spectral_emission = np.array([])

        self.do_aesl = True
        self.aesl_slop_freq = -2.711
        self.aesl_base = -0.125539

        # Fitted to LocVisVC (http://dx.doi.org/10.1109/CVPR.2019.00558) using 'lpyr' (08/06/2020)        
        # Manually adjusted
        self.base_sensitivity_correction = 0.2501039518897
        self.mask_self=1.05317173444
        self.mask_xn=-0.43899877503
        self.mask_xo=-50
        self.mask_p=0.3424
        self.do_sprob_sum=True
        self.mask_q=0.108934275615
        self.psych_func_slope=0.34
        self.si_sigma=-0.502280453708
        self.do_robust_pdet = True
        self.do_spatial_total_pooling = False
        self.do_civdm = True
        self.ms_decomp='lpyr'

        self.sensitivity_correction = self.base_sensitivity_correction

        # Achromatic CSF
        self.csf_m1_f_max = 0.425509
        self.csf_m1_s_high = -0.227224
        self.csf_m1_s_low = -0.227224
        self.csf_m1_exp_low = np.log10(2)

        par = [0.061466549455263, 0.99727370023777070]; # old parametrization of MTF
        self.mtf_params_a = np.array([par[1]*0.426, par[1]*0.574, (1-par[1])*par[0], (1-par[1])*(1-par[0])])
        self.mtf_params_b = np.array([0.028, 0.37, 37, 360])

        # metric_par.quality_band_freq = [15 7.5 3.75 1.875 0.9375 0.4688 0.2344];
        self.quality_band_freq = [60, 30, 15, 7.5, 3.75, 1.875, 0.9375, 0.4688, 0.2344, 0.1172]

        # metric_par.quality_band_w = [0.2963    0.2111    0.1737    0.0581   -0.0280    0.0586    0.2302];

        # New quality calibration: LDR + HDR datasets - paper to be published
        # metric_par.quality_band_w = [0.2832    0.2142    0.2690    0.0398    0.0003    0.0003    0.0002];
        self.quality_band_w = np.array([0, 0.2832, 0.2832, 0.2142, 0.2690, 0.0398, 0.0003, 0.0003, 0, 0])

        self.quality_logistic_q1 = 3.455
        self.quality_logistic_q2 = 0.8886

        # HDR-VDP-csf refitted on 19/04/2020
        self.csf_params = np.array([
            [0.699404,   1.26181,   4.27832,   0.361902,   3.11914],
            [1.00865,   0.893585,   4.27832,   0.361902,   2.18938],
            [1.41627,   0.84864,  3.57253,   0.530355,   3.12486],
            [1.90256,   0.699243,   3.94545,   0.68608,   4.41846],
            [2.28867,   0.530826,   4.25337,   0.866916,   4.65117],
            [2.46011,   0.459297,   3.78765,   0.981028,   4.33546],
            [2.5145,   0.312626,   4.15264,   0.952367,   3.22389]
        ])

        self.csf_lums = np.array([ 0.0002, 0.002, 0.02, 0.2, 2, 20, 150])

        self.csf_sa = np.array([315.98, 6.7977, 1.6008, 0.25534])
        self.csf_sr = np.array([1.1732, 1.32, 1.095, 0.5547, 2.9899, 1.8]) # rod sensitivity function

        self.mult_scale  = MultScale()

        self.rod_sensitivity = 0
        self.cvi_sens_drop_rod = -0.58342

        self.pad_value = 0
        
        self.do_aod = True
        self.do_slum = True

