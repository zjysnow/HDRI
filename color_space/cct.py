import numpy as np
from .white_points import *
from .gamut import *
from .xyz import *

M_vonKries = np.array([
    [0.3897, 0.6890, -0.0787],
    [-0.2298, 1.1834, 0.0464],
    [0, 0, 1]
])

M_Bradford = np.array([
    [0.8951, 0.2664, -0.1614],
    [-0.7502, 1.71135, 0.0367],
    [0.0389, -0.0685, 1.0296]
])

M_cat02 = np.array([
    [0.7328, 0.4296, -0.1624],
    [-0.7036, 1.6975, 0.0061],
    [0.0030, 0.0136, 0.9834]
])

def getMatrixCCT(M_lms, Ts, Td):
    def getWhitePointXYZ(T):
        x, y = WhitePoint[T]
        # rgb = np.matmul(np.linalg.inv(M), np.array([[x/y],[1],[(1-x-y)/y]]))
        # rgb = rgb + np.maximum(0, -np.min(rgb))
        # rgb = rgb / np.max(rgb)
        # return np.matmul(M, rgb)
        return np.array([[x/y],[1],[(1-x-y)/y]])

    scale = np.eye(3) * np.matmul(M_lms, getWhitePointXYZ(Td)) / np.matmul(M_lms, getWhitePointXYZ(Ts))
    return np.matmul(np.linalg.inv(M_lms), np.matmul(scale, M_lms))