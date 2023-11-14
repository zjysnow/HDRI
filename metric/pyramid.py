import numpy as np
from scipy import signal

def gaussian_2D(X,K):
    pad_size = K.shape[0]//2
    res=np.pad(X, (pad_size,pad_size), mode="reflect")
    res = signal.convolve2d(res, K.T, "valid")
    res = signal.convolve2d(res, K, "valid")
    return res

def gaussian_pyramid(img, levels:np.int32, kernel_a = 0.4):
    '''
    img - image in grey scale
    '''
    default_levels = np.floor(np.log2(np.min(img.shape))).astype(np.int32)
    # print("levels:", levels)
    # print("default_levels:", default_levels)

    if levels == -1:
        levels = default_levels
    
    K = np.array([0.25 - kernel_a/2, 0.25, kernel_a, 0.25, 0.25 - kernel_a/2]).reshape(5,1)

    res = np.zeros((levels, img.shape[0], img.shape[1]), dtype=np.float64)
    res[0] = img
    for i in range(1, levels):
        dilatedK = np.zeros((5+4*(2**(i-1)-1), 1), dtype=np.float64)
        dilatedK[::2**(i-1)] = K[:]
        res[i] = gaussian_2D(res[i-1], dilatedK)
    return res

def laplacian_pyramid(img, levels, kernel_a=0.4):
    gpyr = gaussian_pyramid(img, levels, kernel_a)
    lpyr = np.zeros_like(gpyr)
    for i in range(gpyr.shape[0] - 1):
        lpyr[i,:,:] = gpyr[i,:,:] - gpyr[i+1,:,:]
    lpyr[-1,:,:] = gpyr[-1,:,:]
    return lpyr, gpyr