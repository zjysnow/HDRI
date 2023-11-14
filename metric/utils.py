import numpy as np
from numpy.fft import fft2, ifft2
from scipy import signal
from scipy.interpolate import pchip_interpolate

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

def fspecial_gaussian(size, sigma):
    x,y = np.mgrid[-size//2 + 1: size//2+1, -size//2+1:size//2+1]
    g = np.exp(-((x**2+y**2)/(2.0*sigma**2)))
    return g/g.sum()

def fast_gauss(X, sigma, do_norm = True, pad_value = "replicate"):
    # not fast
    ksize = np.round(sigma*6).astype(np.int32)
    ksize = ksize + 1 - np.mod(ksize,2) # Make sure the kernel size is always an odd number
    h = fspecial_gaussian(ksize, sigma)
    
    padX = np.pad(X, (ksize//2,), "edge")
    Y = signal.convolve2d(padX,h, "valid")
    return Y

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

def fast_conv_fft(X, fH, pad_value):
    pad_size = np.array(fH.shape) - np.array(X.shape)
    
    padX = np.ones(X.shape + pad_size) * pad_value

    padX[0:X.shape[0], 0:X.shape[1]] = X
    fX = fft2(padX)
    Y1 = np.real(ifft2(fX*fH, [fX.shape[0], fX.shape[1]]))
    Y = Y1[0:X.shape[0], 0:X.shape[1]]
    return Y

def local_adapt(L_otf, ppd):
    sigma = 10**-0.781367 * ppd
    L_la = np.exp(fast_gauss(np.log(np.maximum(L_otf, 1e-6)), sigma))
    return L_la

def pupil_d_stanley_davies(L, area):
    La = L * area
    return 7.75 - 5.75 * ((La/846)**0.41 / ((La/846)**0.41 + 2))

def pupil_d_unified(L, area, age):
    y0 = 28.58 # referenceage from the paper
    y = np.clip(age, 20, 83)
    d_sd = pupil_d_stanley_davies(L, area)

    return d_sd + (y-y0)*(0.02132-0.009562*d_sd)