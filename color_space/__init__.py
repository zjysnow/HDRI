from .gamut import *
from .yuv import *
from .xyz import *
from .white_points import *
from .cct import *
from .convert import *

import numpy as np
import matplotlib.pyplot as plt

def imshow_in_gamut(image, gamut, white_point, label="", EOTF=(lambda x: x**2.2), gain=1):
    M = getMatrixRGB2XYZ(gamut, white_point)
    rgb = EOTF(image) * gain
    xyz = np.matmul(rgb, M.T)
    # when X = Y = Z = 0, set the value to referene white
    xyz[np.sum(xyz, axis=2)==0,:] = np.matmul(M, np.array([1,1,1]))
    xyz = np.clip(xyz, 0, 1)
    x = xyz[:,:,0] / np.sum(xyz, axis=2)
    y = xyz[:,:,1] / np.sum(xyz, axis=2)

    plt.scatter(x,y, s=20, alpha=xyz[:,:,1], c=image.reshape(-1,3), linewidths=0)
    plt.fill(gamut[:,0], gamut[:,1], color="b", fill=False, label=label)
    plt.fill(DCI_P3[:,0], DCI_P3[:,1], label="DCI_P3", color="r", fill=False)
    plt.fill(BT709[:,0], BT709[:,1], label="sRGB", color="g", fill=False)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()