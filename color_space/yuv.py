from .xyz import getMatrixRGB2XYZ
import numpy as np

def getMatrixRGB2YPbPr(Yr, Yb):
    Yg = 1 - Yr - Yb
    return np.array([
        [Yr, Yg, Yb],
        [-0.5*Yr/(1-Yb), -0.5*Yg/(1-Yb),0.5],
        [0.5, -0.5*Yg/(1-Yr), -0.5*Yb/(1-Yr)]
    ])

def getMatrixYPbPr2RGB(Yr, Yb):
    Yg = 1 - Yr - Yb
    return np.array([
        [1, 0, 2-2*Yr],
        [1, -Yb/Yg*(2-2*Yb), -Yr/Yg*(2-2*Yr)],
        [1, 2-2*Yb, 0]
    ])

def getMatrixRGB2YUV(gamut, white_point= np.array([0.31271, 0.32902]), is_narrow=False, weight_bits=8, offset_bits=8):
    Yr, _, Yb = getMatrixRGB2XYZ(gamut, white_point)[1]
    M = getMatrixRGB2YPbPr(Yr, Yb)
    
    scale = (np.array([[219],[224],[224]])<<(offset_bits-8)) / (2**offset_bits-1) if is_narrow else np.ones((3,1))
    offset = np.array([[16],[128],[128]]) if is_narrow else np.array([[0],[128],[128]])

    return np.round((2**weight_bits) * scale * M), offset << (offset_bits - 8)


def getMatrixYUV2RGB(gamut, white_point= np.array([0.31271, 0.32902]), is_narrow=False, weight_bits=8, offset_bits=8):
    Yr, _, Yb = getMatrixRGB2XYZ(gamut, white_point)[1]
    M = getMatrixYPbPr2RGB(Yr, Yb)

    scale = (2**offset_bits-1)/(np.array([219,224,224])<<(offset_bits-8)) if is_narrow else np.ones((3,1))
    offset = np.array([[16],[128],[128]]) if is_narrow else np.array([[0],[128],[128]])

    return np.round((2**weight_bits)*scale*M), np.round(np.matmul(scale*M, -(offset << (offset_bits-8))))