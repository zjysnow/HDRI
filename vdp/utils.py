import numpy as np
from numpy.fft import fft2, ifft2

from scipy.interpolate import pchip_interpolate

import matplotlib.pyplot as plt
from .metric_params import MetricParams

from .mtf import mtf

def pix_per_degree(display_diagonal, resolution, viewing_distance):
    '''
    display_diagonal: inch
    resolution: [height, width]
    viewing_distance: in meters
    '''
    ar = resolution[1] / resolution[0] # width / height
    height_mm = np.sqrt((display_diagonal*25.4)**2/(1+ar**2))
    height_deg = 2 * np.arctan( 0.5*height_mm/(viewing_distance*1000) ) / np.pi * 180
    return resolution[0]/height_deg

def gain_offset(Y, Y_peak, contrast, E_ambient, k_refl):
    Y_refl = E_ambient / np.pi * k_refl
    Y_black = Y_refl + Y_peak / contrast
    return (Y_peak - Y_black) * Y + Y_black

def create_cycdeg_image( im_size, pix_per_deg ):
    nyquist_freq = 0.5 * pix_per_deg

    KX0 = np.mod(0.5 + np.array(range(im_size[1]))/im_size[1], 1) - 0.5 # [np.mod(1/2 + (0:(im_size[2]-1))/im_size[2], 1) - 1/2]
    KX1 = KX0 * nyquist_freq * 2
    KY0 = np.mod(0.5 + np.array(range(im_size[0]))/im_size[0], 1) - 0.5 # (np.mod(1/2 + (0:(im_size[1]-1))/im_size[1], 1) - 1/2)
    KY1 = KY0 * nyquist_freq * 2

    XX, YY = np.meshgrid(KX1, KY1)

    D = np.sqrt( XX**2 + YY**2 )
    return D

def load_spectral_resp(file_name):
    with open(file_name) as f:
        D = np.loadtxt(f, delimiter = ",")

    # l_minmax = [360, 780]
    l_min = 360
    l_max = 780
    l_step = 1
    
    l = np.linspace( l_min, l_max, int((l_max-l_min)/l_step) )

    R = np.zeros((l.shape[0], D.shape[1]-1)) # R = zeros( length(lambda_), size(D,2)-1 );
    for k in range(1,D.shape[1]): # 2:size(D,2)
        R[:,k-1] = pchip_interpolate(D[:,0], D[:,k], l) 
    return l, R

def joint_rod_cone_sens(la, metric_par: MetricParams):
    cvi_sens_drop = metric_par.csf_sa[1]
    cvi_trans_slope = metric_par.csf_sa[2]
    cvi_low_slope = metric_par.csf_sa[3]
    # S = metric_par.csf_sa(1) * ( (cvi_sens_drop./la).^cvi_trans_slope+1).^-cvi_low_slope;
    return metric_par.csf_sa[0] * ((cvi_sens_drop/la)**cvi_trans_slope+1) ** -cvi_low_slope

def rod_sens(la, metric_par: MetricParams):
    S = np.zeros_like(la)
    peak_l = metric_par.csf_sr[0]
    low_s = metric_par.csf_sr[1]
    low_exp = metric_par.csf_sr[2]
    high_s = metric_par.csf_sr[3]
    high_exp = metric_par.csf_sr[4]
    rod_sens = metric_par.csf_sr[5]

    ss = la>peak_l
    S[ss] = np.exp(-np.abs(np.log10(la[ss]/peak_l))**high_exp/high_s)
    S[~ss] = np.exp(-np.abs(np.log10(la[~ss]/peak_l))**low_exp/low_s)

    return S * 10 ** rod_sens
    
def fast_conv_fft(X, fH, pad_value):
    pad_size = np.array(fH.shape) - np.array(X.shape)
    
    padX = np.ones(X.shape + pad_size) * pad_value

    padX[0:X.shape[0], 0:X.shape[1]] = X
    fX = fft2(padX)
    Y1 = np.real(ifft2(fX*fH, [fX.shape[0], fX.shape[1]]))
    Y = Y1[0:X.shape[0], 0:X.shape[1]]
    return Y

def aesl(rho, metric_par: MetricParams):
    gamma = 10**metric_par.aesl_base
    S_corr = 10**(-(10**metric_par.aesl_slop_freq * np.log2(rho+gamma)) * np.maximum(0, metric_par.age - 24))
    return S_corr

def ncsf(rho, lum, metric_par: MetricParams):
    csf_pars = metric_par.csf_params
    lum_lut = np.log10(metric_par.csf_lums)
    log_lum = np.log10(lum)
    par = np.zeros((lum.shape[0],4))
    for k in range(4):
        par[:,k] = np.interp(np.clip(log_lum, lum_lut[0], lum_lut[-1]), lum_lut, csf_pars[:,k+1])

    # S = np.piecewise(rho, [rho <= 1e-4], [
    #     lambda rho: 0,
    #     lambda rho: par[:,3] / ((1+(par[:,0]*rho)**par[:,1])/(1-np.exp(-(-rho/7)**2))**par[:,2])**0.5
    # ])
    S = par[:,3] / ((1+(par[:,0]*rho)**par[:,1])/(1-np.exp(-(-rho/7)**2))**par[:,2])**0.5
    S = S / mtf(rho, metric_par)

    S[rho<=1e-4] = 0

    if metric_par.do_aesl:
        S = S * aesl(rho, metric_par)
    
    return S

def pupil_d_stanley_davies(L, area):
    La = L * area
    return 7.75 - 5.75 * ((La/846)**0.41 / ((La/846)**0.41 + 2))

def pupil_d_unified(L, area, age):
    y0 = 28.58 # referenceage from the paper
    y = np.clip(age, 20, 83)
    d_sd = pupil_d_stanley_davies(L, area)

    return d_sd + (y-y0)*(0.02132-0.009562*d_sd)


if __name__ == "__main__":
    # 30" 4K monitor seen from 0.5 meters
    ppd = pix_per_degree(30, [2160,3840], 0.5) # 52.722441427310380
    print(ppd)

    D = create_cycdeg_image([1536,2048], ppd)

    l, R = load_spectral_resp("vdp/data/emission_spectra_led-lcd-srgb.csv")
    
    # plt.plot(l, R)
    # plt.show()

    img = np.array([
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 4, 3, 2, 1, 3, 3, 1], 
        [2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 5, 6, 6, 4, 4, 5],
        [3, 4, 5, 6, 7, 8, 9, 3, 3, 4, 5, 6, 7, 3, 4, 5],
        [2, 4, 6, 7, 3, 2, 4, 2, 7, 8, 0, 4, 5, 6, 7, 8],
        [4, 5, 6, 7, 3, 2, 1, 2, 4, 5, 7, 8, 9, 0, 3, 1],
        [3, 1, 2, 3, 4, 8, 6, 4, 3, 1, 4, 6, 8, 9, 0, 3],
        [1, 3, 4, 2, 5, 4, 3, 5, 6, 8, 9, 4, 5, 2, 5, 6],
        [6, 3, 4, 1, 5, 3, 6, 7, 9, 3, 9, 7, 0, 8, 5, 3],
        [1, 2, 5, 3, 2, 5, 7, 8, 4, 2, 4, 2, 1, 3, 4, 6],
        [0, 9, 5, 3, 1, 3, 1, 2, 1, 2, 3, 3, 4, 3, 2, 1],
        [7, 7, 3, 4, 3, 3, 2, 1, 2, 3, 5, 7, 6, 5, 0, 0],
        [3, 4, 5, 6, 4, 3, 3, 4, 4, 5, 5, 5, 7, 8, 9, 0],
        [6, 7, 8, 7, 8, 6, 7, 6, 7, 6, 7, 6, 5, 4, 3, 2],
        [4, 5, 6, 7, 8, 9, 0, 2, 1, 2, 3, 4, 5, 6, 7, 8],
        [3, 2, 1, 3, 2, 2, 2, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [3, 4, 5, 3, 2, 1, 3, 4, 5, 6, 7, 8, 9, 0, 7, 5]
    ])

    Y = fast_conv_fft(img, img, 0)
    print(Y)