import numpy as np
from .gamut import *
from .white_points import *
from .xyz import *

def xyY2XYZ(x,y,Y):
    X = x*Y/y
    Z = (1-x-y)*Y/y
    return X,Y,Z

def RGB2XYZ(R,G,B, primary_color = BT709, white_point = WhitePoint[D65]):
    '''
    default is sRGB(BT709) D65
    '''
    M = getMatrixRGB2XYZ(primary_color, white_point)
    X,Y,Z = np.matmul(M, (R,G,B))
    return X,Y,Z

def XYZ2RGB(X,Y,Z, primary_color = BT709, white_point = WhitePoint[D65]):
    M = getMatrixRGB2XYZ(primary_color, white_point)
    R,G,B = np.matmul(np.linalg.inv(M), (X,Y,Z))
    return R,G,B


def RGB2HSV(R,G,B):
    Cmax = np.max([R,G,B])
    Cmin = np.min([R,G,B])
    delta = Cmax - Cmin

    # H = 0
    if delta == 0:
        H = 0
    else:
        if Cmax == R:
            H = ((G-B)/delta+0)/6
        if Cmax == G:
            H = ((B-R)/delta+2)/6
        if Cmax == B:
            H = ((R-G)/delta+4)/6
    # H = 0 if delta == 0 else ((G-B)/delta+0)/6 if Cmax == R else ((B-R)/delta+2)/6 if Cmax == G else ((R-G)/delta+4)/6 if Cmax == B else 0
    # H = np.piecewise(0.0, [delta==0, Cmax == R, Cmax == G, Cmax == B], [
    #     lambda x: 0,
    #     lambda x: ((G-B)/delta+0.0)/6,
    #     lambda x: ((B-R)/delta+2.0)/6,
    #     lambda x: ((R-G)/delta+4.0)/6
    # ])
    
    S = 0 if Cmax == 0 else delta/Cmax
    # S = np.piecewise(0.0, [Cmax == 0], [
    #     lambda x: 0,
    #     lambda x: delta/Cmax
    # ])
    V = Cmax
    return H, S, V

def HSV2RGB(H,S,V):
    h = np.floor(H * 6)
    f = H * 6 - h
    p = V * (1-S)
    q = V * (1-f*S)
    t = V * (1-(1-f)*S)

    if h == 0:
        return V,t,p
    if h == 1:
        return q,V,p
    if h == 2:
        return p,V,t
    if h == 3:
        return p,q,V
    if h == 4:
        return t,p,V
    if h == 5:
        return V,p,q
    
