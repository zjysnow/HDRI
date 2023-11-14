import numpy as np
from .mtf import MTF
from .aesl import AESL


class CSF:
    def __init__(self, age = 24, do_aesl : bool = True):
        self.csf_la = np.logspace(-5,5,256)
        self.csf_log_la = np.log10(self.csf_la)
        
        self.do_aesl = do_aesl
        self.aesl = AESL(age=age)

        self.csf_lums = np.array([ 0.0002, 0.002, 0.02, 0.2, 2, 20, 150])

        self.csf_sa = np.array([315.98, 6.7977, 1.6008, 0.25534])
        self.csf_sr = np.array([1.1732, 1.32, 1.095, 0.5547, 2.9899, 1.8]) # rod sensitivity function

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

        self.mtf = MTF(age=age)


    def rod_sens(self, la):
        S = np.zeros_like(la)
        peak_l = self.csf_sr[0]
        low_s = self.csf_sr[1]
        low_exp = self.csf_sr[2]
        high_s = self.csf_sr[3]
        high_exp = self.csf_sr[4]
        rod_sens = self.csf_sr[5]

        ss = la>peak_l
        S[ss] = np.exp(-np.abs(np.log10(la[ss]/peak_l))**high_exp/high_s)
        S[~ss] = np.exp(-np.abs(np.log10(la[~ss]/peak_l))**low_exp/low_s)

        return S * 10 ** rod_sens
    
    def joint_rod_cone_sens(self, la):
        cvi_sens_drop = self.csf_sa[1]
        cvi_trans_slope = self.csf_sa[2]
        cvi_low_slope = self.csf_sa[3]
        # S = metric_par.csf_sa(1) * ( (cvi_sens_drop./la).^cvi_trans_slope+1).^-cvi_low_slope;
        return self.csf_sa[0] * ((cvi_sens_drop/la)**cvi_trans_slope+1) ** -cvi_low_slope

    def ncsf(self, rho):
        lum = np.logspace(-5,5,256)
        csf_pars = self.csf_params
        lum_lut = np.log10(self.csf_lums)
        log_lum = np.log10(lum)
        par = np.zeros((lum.shape[0],4))
        for k in range(4):
            par[:,k] = np.interp(np.clip(log_lum, lum_lut[0], lum_lut[-1]), lum_lut, csf_pars[:,k+1])

        # S = np.piecewise(rho, [rho <= 1e-4], [
        #     lambda rho: 0,
        #     lambda rho: par[:,3] / ((1+(par[:,0]*rho)**par[:,1])/(1-np.exp(-(-rho/7)**2))**par[:,2])**0.5
        # ])
        S = par[:,3] / ((1+(par[:,0]*rho)**par[:,1])/(1-np.exp(-(-rho/7)**2))**par[:,2])**0.5
        S = S / self.mtf(rho)

        S[rho<=1e-4] = 0

        if self.do_aesl:
            S = S * self.aesl(rho)
        
        return S

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    la = np.logspace(-4,4,4096)
    S = CSF().rod_sens(la)

    plt.loglog(la, S)
    plt.show()