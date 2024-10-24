import numpy as np
from .gamut import *
from .white_points import *
from .xyz import *

def xyY2XYZ(x,y,Y):
    X = x*Y/y
    Z = (1-x-y)*Y/y
    return X,Y,Z

def XYZ2xyY(X,Y,Z):
    x = X/(X+Y+Z)
    y = Y/(X+Y+Z)
    return x,y,Y

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


def XYZ2Lab(X,Y,Z, white_point = WhitePoint[D65]):
    Xw, Yw, Zw = xyY2XYZ(white_point[0], white_point[1], 1)
    ep = 216/24389
    kp = 24389/27
    x = X / Xw
    y = Y / Yw
    z = Z / Zw
    fx = x**(1/3) if x > ep else (kp*x+16)/116
    fy = y**(1/3) if y > ep else (kp*y+16)/116
    fz = z**(1/3) if z > ep else (kp*z+16)/116
    L = 116*fy-16
    a = 500*(fx-fy)
    b = 200*(fy-fz)
    return L, a, b

def Lab2XYZ(L,a,b,white_point=WhitePoint[D65]):
    Xw, Yw, Zw = xyY2XYZ(white_point[0], white_point[1], 1)
    ep = 216/24389
    kp = 24389/27
    fy = (L + 16)/116
    fx = a/500+fy
    fz = fy-b/200
    x = fx**3 if fx**3>ep else (116*fx-16)/kp
    y = ((L+16)/116)**3 if L > kp*ep else L/kp
    z = fz**3 if fz**3>ep else (116*fz-16)/kp
    X = x*Xw
    Y = y*Yw
    Z = z*Zw
    return X,Y,Z

def RGB2HSV(R,G,B):
    Cmax = np.max([R,G,B])
    Cmin = np.min([R,G,B])
    delta = Cmax - Cmin

    # H = 0
    if delta == 0:
        H = 0
    else:
        if Cmax == R and G >= B:
            H = ((G-B)/delta+0)/6
        if Cmax == R and G < B:
            H = ((G-B)/delta+6)/6
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
