import numpy as np
from .gaussian_pyramid import gaussian_pyramid

def laplacian_pyramid(img, levels, kernel_a=0.4):
    gpyr = gaussian_pyramid(img, levels, kernel_a)
    lpyr = np.zeros_like(gpyr)
    for i in range(gpyr.shape[0] - 1):
        lpyr[i,:,:] = gpyr[i,:,:] - gpyr[i+1,:,:]
    lpyr[-1,:,:] = gpyr[-1,:,:]

    return lpyr, gpyr
    

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

    res = laplacian_pyramid(img, 4)
    print(type(res[0]))
    # for i in range(res(0).shape[0]):
    #     print("layer", i)
    #     print(lpyr[i])