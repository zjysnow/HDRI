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


if __name__ == "__main__":
    import matplotlib.pyplot as plt
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

    res = gaussian_pyramid(img, 1)


    for i in range(res.shape[0]):
        print("layer", i)
        print(res[i])

    # plt.subplot(2,4,1)
    # plt.imshow(img)
    # plt.subplot(2,4,5)
    # plt.imshow(res[0])
    # plt.subplot(2,4,6)
    # plt.imshow(res[1])
    # plt.subplot(2,4,7)
    # plt.imshow(res[2])
    # plt.subplot(2,4,8)
    # plt.imshow(res[3])
    # plt.show()
    
