import numpy as np
from scipy import signal

def fspecial_gaussian(size, sigma):
    x,y = np.mgrid[-size//2 + 1: size//2+1, -size//2+1:size//2+1]
    print(sigma)
    g = np.exp(-((x**2+y**2)/(2.0*sigma**2)))
    return g/g.sum()

def fast_gauss(X, sigma, do_norm = True, pad_value = "replicate"):
    # not fast
    ksize = np.round(sigma*6).astype(np.int32)
    ksize = ksize + 1 - np.mod(ksize,2) # Make sure the kernel size is always an odd number
    h = fspecial_gaussian(ksize, sigma)
    
    padX = np.pad(X, (ksize//2,), "edge")
    Y = signal.convolve2d(padX,h, "valid")
    print(X.shape, Y.shape)
    return Y


if __name__ == "__main__":
    X = np.array([
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
    Y = fast_gauss(X, 1)
    print(Y)