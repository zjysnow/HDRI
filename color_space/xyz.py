import numpy as np

def xyY2XYZ(x,y,Y):
    X = x*Y/y
    Z = (1-x-y)*Y/y
    return X,Y,Z

def getMatrixRGB2XYZ(primary_color, white_point = np.array([0.31271, 0.32902])):
    '''
    primary_color: 3x2 matrix
    white_point: use D65 as default white point
    '''
    M = np.array([
        primary_color[:,0] / primary_color[:,1],
        [1, 1, 1],
        (1 - primary_color[:,0] - primary_color[:,1])/primary_color[:,1]
    ])

    White = np.array([white_point[0] / white_point[1], 1, (1 - white_point[0] - white_point[1]) / white_point[1]])
    Minv = np.linalg.inv(M)
    S = np.matmul(Minv, White)
    return M * S