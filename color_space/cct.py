import numpy as np

M_cat02 = np.array([
    [0.7328, 0.4296, -0.1624],
    [-0.7036, 1.6975, 0.0061],
    [0.0030, 0.0136, 0.9834]
])

def getMatrixCCT(M_lms, M_s, M_d):
    scale = np.eye(3) * np.matmul(M_lms, np.matmul(M_d, np.ones((3,1)))) / np.matmul(M_lms, np.matmul(M_s, np.ones((3,1))))
    return np.matmul(np.linalg.inv(M_lms), np.matmul(scale, M_lms))