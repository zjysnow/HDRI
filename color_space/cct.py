import numpy as np
from .white_points import *
from .gamut import *
from .xyz import *

M_cat02 = np.array([
    [0.7328, 0.4296, -0.1624],
    [-0.7036, 1.6975, 0.0061],
    [0.0030, 0.0136, 0.9834]
])

def getMatrixCCT(M_lms, M, Ts, Td):
    def getWhitePoint(M, T):
        x, y = WhitePoint[T]
        rgb = np.matmul(np.linalg.inv(M), np.array([[x/y],[1],[(1-x-y)/y]]))
        rgb = rgb + np.maximum(0, -np.min(rgb))
        rgb = rgb / np.max(rgb)
        return np.matmul(M, rgb)
    
    scale = np.eye(3) * np.matmul(M_lms, getWhitePoint(M, Td)) / np.matmul(M_lms, getWhitePoint(M, Ts))
    return np.matmul(np.linalg.inv(M_lms), np.matmul(scale, M_lms))